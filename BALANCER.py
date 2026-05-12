#!/usr/bin/env python3
import argparse
import os
import json
import gc
import hashlib
import heapq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

DEFAULT_MIN_Q_BITS = {"q": 3, "k": 3, "v": 3, "o": 4, "up": 4, "gate": 4, "down": 4}
DEFAULT_RANK_CAP = {2: 32, 3: 64, 4: 128, 5: 256}

def _stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "little")

@torch.no_grad()
def quantize_2bit_lsq(W, q2_mode="zero", q2_iters=8, eps=1e-12):
    Wf = W.float()
    codebooks = {
        "sym": (torch.tensor([-2., -1., 1., 2.], device=Wf.device), 2.0),
        "pm3": (torch.tensor([-3., -1., 1., 3.], device=Wf.device), 3.0),
        "zero": (torch.tensor([-2., -1., 0., 1.], device=Wf.device), 2.0),
    }
    code, qmax = codebooks.get(q2_mode, codebooks["zero"])
    s = torch.amax(Wf.abs(), dim=1, keepdim=True).clamp_min(eps) / qmax
    for _ in range(max(0, int(q2_iters))):
        ratio = (Wf / s).unsqueeze(-1)
        idx = torch.argmin((ratio - code)**2, dim=-1)
        qf = code[idx]
        denom = (qf * qf).sum(dim=1, keepdim=True).clamp_min(eps)
        s = ((Wf * qf).sum(dim=1, keepdim=True) / denom).clamp_(min=1e-6, max=1e3)
    ratio = (Wf / s).unsqueeze(-1)
    idx = torch.argmin((ratio - code)**2, dim=-1)
    Wq = code[idx] * s
    R = Wf - Wq
    return Wq.to(W.dtype), R.to(W.dtype), s.squeeze(1).to(W.dtype)

@torch.no_grad()
def quantize_symmetric(W, bits=4, eps=1e-8):
    m, n = W.shape
    qmax = 2.0 if bits == 2 else float((2 ** (bits - 1)) - 1)
    scale = W.abs().amax(dim=1, keepdim=True).clamp_min(eps) / qmax
    if bits == 2:
        t = (W / scale).to(W.dtype)
        Wint = torch.where(t >= 0,
            torch.where(t > 1.5, torch.full_like(t, 2.), torch.full_like(t, 1.)),
            torch.where(t < -1.5, torch.full_like(t, -2.), torch.full_like(t, -1.)))
    else:
        Wint = torch.round(W / scale).clamp(-qmax, qmax)
    Wq = Wint * scale
    R = W - Wq
    return Wq, R, scale.squeeze(1), Wint

@torch.no_grad()
def quantize_metric(W, sx, bits=4, eps=1e-8):
    m, n = W.shape
    qmax = 2.0 if bits == 2 else float((2 ** (bits - 1)) - 1)
    sx = sx.to(W.device, W.dtype).clamp_min(eps)
    W_tilde = W * sx.unsqueeze(0)
    scale = W_tilde.abs().amax(dim=1, keepdim=True).clamp_min(eps) / qmax
    if bits == 2:
        t = (W_tilde / scale).to(W.dtype)
        Wint = torch.where(t >= 0,
            torch.where(t > 1.5, torch.full_like(t, 2.), torch.full_like(t, 1.)),
            torch.where(t < -1.5, torch.full_like(t, -2.), torch.full_like(t, -1.)))
    else:
        Wint = torch.round(W_tilde / scale).clamp(-qmax, qmax)
    Wq_tilde = Wint * scale
    Wq = Wq_tilde / sx.unsqueeze(0)
    R = W - Wq
    return Wq, R, scale.squeeze(1), Wint

@torch.no_grad()
def quantize_2bit_metric(W, sx, q2_mode="zero", q2_iters=8):
    sx_safe = sx.to(W.dtype).clamp_min(1e-12)
    Wt = W * sx_safe.unsqueeze(0)
    Wt_q, _, _ = quantize_2bit_lsq(Wt, q2_mode=q2_mode, q2_iters=q2_iters)
    Wq = (Wt_q / sx_safe.unsqueeze(0)).to(W.dtype)
    R = (W - Wq).to(W.dtype)
    return Wq, R

def fuse_signals(*signals, eps=1e-12):
    valid = [s.float() for s in signals if s is not None]
    if not valid:
        return None
    normalized = []
    for s in valid:
        s_min, s_max = s.min(), s.max()
        normalized.append((s - s_min) / (s_max - s_min + eps))
    return sum(normalized) / len(normalized)

def build_metric(m, n, diag_cx, G=None, H_diag=None, device="cpu", eps=1e-12):
    diag_cx = diag_cx.to(device)
    if G is not None:
        G = G.to(device)
    if H_diag is not None:
        H_diag = H_diag.to(device)
    col_signals = []
    if diag_cx is not None and diag_cx.numel() == n:
        col_signals.append(torch.sqrt(diag_cx.float().clamp_min(1e-12)))
    if G is not None and G.shape == (m, n):
        col_signals.append(G.float().abs().mean(dim=0))
    fused_col = fuse_signals(*col_signals, eps=eps) if col_signals else None
    sx = 1.0 + fused_col if fused_col is not None else torch.ones(n, device=device)
    row_signals = []
    if G is not None and G.shape == (m, n):
        row_signals.append(G.float().abs().mean(dim=1))
    if H_diag is not None and H_diag.numel() == m:
        row_signals.append(H_diag.float().clamp_min(1e-12))
    fused_row = fuse_signals(*row_signals, eps=eps) if row_signals else None
    sy = 1.0 + fused_row if fused_row is not None else torch.ones(m, device=device)
    return sx.clamp_min(eps), sy.clamp_min(eps)

def svd_lowrank_diag(R, sx, sy, k_max=256, device='cpu'):
    m, n = R.shape
    k_max = min(k_max, m, n)
    R = R.to(device)
    sx = sx.to(device).to(R.dtype).clamp_min(1e-8)
    sy = sy.to(device).to(R.dtype).clamp_min(1e-8)
    R_tilde = (sy.unsqueeze(1) * R) * sx.unsqueeze(0)
    Rt = R_tilde.float()
    try:
        U, S, V = torch.svd_lowrank(Rt, q=k_max, niter=4)
    except:
        U_f, S_f, Vh_f = torch.linalg.svd(Rt, full_matrices=False)
        k = min(k_max, len(S_f))
        U, S, V = U_f[:, :k], S_f[:k], Vh_f[:k, :].T
    gain = S ** 2
    A = (U.to(R.dtype) * S.to(R.dtype).unsqueeze(0)) / sy.unsqueeze(1)
    B = V.to(R.dtype) / sx.unsqueeze(1)
    return A, B, gain

class DiagCovCollector:
    def __init__(self, module, dtype=torch.float32):
        self.mod = module
        self.dtype = dtype
        self.m2 = None
        self.count = 0
        self._handle = None

    def _hook(self, module, inputs, output):
        x = inputs[0].detach()
        x2d = x.reshape(-1, x.shape[-1]).to(self.dtype)
        m2_batch = (x2d * x2d).sum(dim=0)
        if self.m2 is None:
            self.m2 = torch.zeros_like(m2_batch)
        self.m2 += m2_batch
        self.count += x2d.shape[0]

    def start(self):
        self._handle = self.mod.register_forward_hook(self._hook)

    def stop(self):
        if self._handle:
            self._handle.remove()
            self._handle = None

    def finalize(self, eps=1e-6):
        if self.m2 is None or self.count == 0:
            return None
        return (self.m2 / self.count).clamp_min(eps).cpu()

def load_signal(path, layer_idx, tag, device='cpu'):
    if not path or not os.path.exists(path):
        return None
    fpath = os.path.join(path, f"{layer_idx}_{tag}.pt")
    if not os.path.exists(fpath):
        return None
    try:
        data = torch.load(fpath, map_location=device)
        if isinstance(data, dict):
            for k in ['grad', 'gradient', 'G', 'hess', 'hessian', 'H', 'diag']:
                if k in data:
                    return data[k]
            for v in data.values():
                if torch.is_tensor(v):
                    return v
        elif torch.is_tensor(data):
            return data
    except:
        pass
    return None

def parse_blocks(s):
    full = ["q", "k", "v", "o", "up", "down", "gate"]
    if s.lower() == "all":
        return full
    return [x.strip() for x in s.split(',') if x.strip() in full]

def tag2mod(layer):
    d = {}
    if hasattr(layer, "self_attn"):
        attn = layer.self_attn
        for t in ["q", "k", "v", "o"]:
            if hasattr(attn, f"{t}_proj"):
                d[t] = getattr(attn, f"{t}_proj")
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        for t in ["up", "down", "gate"]:
            if hasattr(mlp, f"{t}_proj"):
                d[t] = getattr(mlp, f"{t}_proj")
    return d

def collect_covariances(model, tokenizer, calib_text, blocks, nsamples=128, seqlen=512):
    tokens = tokenizer(calib_text, return_tensors="pt").input_ids
    device = model.get_input_embeddings().weight.device
    total = tokens.shape[1]
    required = nsamples * seqlen
    if total < required:
        nsamples = max(1, total // seqlen)
    tokens = tokens[:, :nsamples * seqlen].reshape(-1, seqlen)[:nsamples].to(device)
    covariances = {}
    model.eval()
    for layer_idx, layer in enumerate(tqdm(model.model.layers, desc="Collecting")):
        for tag, mod in tag2mod(layer).items():
            if tag not in blocks:
                continue
            name = f"L{layer_idx:02d}_{tag}"
            collector = DiagCovCollector(mod, dtype=torch.float32)
            collector.start()
            for i in range(tokens.shape[0]):
                with torch.no_grad():
                    model(tokens[i:i+1])
            collector.stop()
            covariances[name] = collector.finalize()
    return covariances

def compute_gains(model, blocks, covariances, grad_dir=None, hess_dir=None, q_bits=4, k_max=256):
    layer_info = []
    for layer_idx, layer in enumerate(tqdm(model.model.layers, desc="Computing")):
        for tag, mod in tag2mod(layer).items():
            if tag not in blocks:
                continue
            name = f"L{layer_idx:02d}_{tag}"
            W = mod.weight.data.cpu().float()
            m, n = W.shape
            Cx = covariances.get(name)
            if Cx is None:
                continue
            diag_cx = Cx.diag() if Cx.ndim == 2 else Cx
            G = load_signal(grad_dir, layer_idx, tag, device='cpu')
            if G is not None and G.shape != (m, n):
                G = None
            H_diag = load_signal(hess_dir, layer_idx, tag, device='cpu')
            if H_diag is not None:
                if H_diag.ndim == 2:
                    H_diag = H_diag.diag()
                if H_diag.numel() != m:
                    H_diag = None
            sx, sy = build_metric(m, n, diag_cx, G, H_diag)
            if int(q_bits) == 2:
                Wq, R = quantize_2bit_metric(W, sx)
            else:
                Wq, R, _, _ = quantize_metric(W, sx, bits=q_bits)
            A, B, gain = svd_lowrank_diag(R, sx, sy, k_max, device='cpu')
            layer_info.append({
                'name': name,
                'layer_idx': layer_idx,
                'tag': tag,
                'shape': (m, n),
                'gains': gain.cpu(),
                'q_bits': int(q_bits),
            })
            del A, B, gain, Wq, R, W
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return layer_info

def allocate_ranks(layer_info, budget_bits, q_bits, lr_bits, scale_bits):
    total_params = sum(info['shape'][0] * info['shape'][1] for info in layer_info)
    baseline_bits = total_params * q_bits
    target_bits = budget_bits * total_params
    allocations = {i: 0 for i in range(len(layer_info))}
    current_bits = baseline_bits
    heap = []
    for i, info in enumerate(layer_info):
        m, n = info['shape']
        gains = info['gains']
        if len(gains) == 0:
            continue
        cost_1 = (m + n) * lr_bits + scale_bits
        if current_bits + cost_1 <= target_bits:
            density = float(gains[0]) / cost_1
            heapq.heappush(heap, (-density, i, 1))
    while heap:
        neg_density, i, k = heapq.heappop(heap)
        info = layer_info[i]
        m, n = info['shape']
        gains = info['gains']
        cost = (m + n) * lr_bits + (scale_bits if k == 1 else 0)
        if current_bits + cost > target_bits:
            break
        allocations[i] = k
        current_bits += cost
        if k < len(gains):
            cap = DEFAULT_RANK_CAP.get(info['q_bits'], 128)
            if k + 1 <= cap:
                next_cost = (m + n) * lr_bits
                if current_bits + next_cost <= target_bits:
                    next_density = float(gains[k]) / next_cost
                    heapq.heappush(heap, (-next_density, i, k + 1))
    realized = current_bits / total_params
    return allocations, realized

def reconstruct_model(model, layer_info, allocations, covariances, grad_dir, hess_dir, q_bits):
    for i, info in enumerate(tqdm(layer_info, desc="Reconstructing")):
        k = allocations[i]
        layer_idx = info['layer_idx']
        tag = info['tag']
        module = model.model.layers[layer_idx]
        attr_map = {'q': 'self_attn.q_proj', 'k': 'self_attn.k_proj', 'v': 'self_attn.v_proj',
                    'o': 'self_attn.o_proj', 'up': 'mlp.up_proj', 'gate': 'mlp.gate_proj', 'down': 'mlp.down_proj'}
        parts = attr_map[tag].split('.')
        target = module
        for p in parts[:-1]:
            target = getattr(target, p)
        linear = getattr(target, parts[-1])
        W_orig = linear.weight.data
        m, n = W_orig.shape
        dst_device = W_orig.device
        W = W_orig.detach().cpu().float()
        name = info['name']
        Cx = covariances.get(name, torch.ones(n))
        diag_cx = Cx.diag() if Cx.ndim == 2 else Cx
        G = load_signal(grad_dir, layer_idx, tag, device='cpu')
        if G is not None and G.shape != (m, n):
            G = None
        H_diag = load_signal(hess_dir, layer_idx, tag, device='cpu')
        if H_diag is not None:
            if H_diag.ndim == 2:
                H_diag = H_diag.diag()
            if H_diag.numel() != m:
                H_diag = None
        sx, sy = build_metric(m, n, diag_cx, G, H_diag, device="cpu")
        if int(q_bits) == 2:
            Wq, R = quantize_2bit_metric(W, sx)
        else:
            Wq, R, _, _ = quantize_metric(W, sx, bits=q_bits)
        if k > 0:
            A, B, _ = svd_lowrank_diag(R, sx, sy, k_max=k, device='cpu')
            W_compressed = Wq + (A @ B.T)
        else:
            W_compressed = Wq
        orig_norm = W.norm().clamp_min(1e-12)
        reco_norm = W_compressed.norm().clamp_min(1e-12)
        W_compressed = W_compressed * (orig_norm / reco_norm)
        linear.weight.data.copy_(W_compressed.to(dst_device, dtype=W_orig.dtype))
        del W, Wq, R
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--blocks", default="q,k,v,o,up,down,gate")
    parser.add_argument("--budget_bits", type=float, default=4.0)
    parser.add_argument("--q_bits", type=int, default=4)
    parser.add_argument("--lr_bits", type=int, default=4)
    parser.add_argument("--scale_bits", type=int, default=16)
    parser.add_argument("--k_max", type=int, default=256)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--grad_dir", default=None)
    parser.add_argument("--hess_dir", default=None)
    args = parser.parse_args()
    blocks = parse_blocks(args.blocks)
    print("=" * 80)
    print("BALANCER: Bi-diagonal Adaptive Low-rank Allocation with Neural Compression")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Budget: {args.budget_bits} bits/param")
    print(f"Signals: Activation" + (f" + Gradient ({args.grad_dir})" if args.grad_dir else "") + (f" + Curvature ({args.hess_dir})" if args.hess_dir else ""))
    print("=" * 80)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map={"": 0} if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_text = "\n\n".join(data["text"][:5000])
    covariances = collect_covariances(model, tokenizer, calib_text, blocks, args.nsamples, args.seqlen)
    layer_info = compute_gains(model, blocks, covariances, args.grad_dir, args.hess_dir, args.q_bits, args.k_max)
    allocations, realized = allocate_ranks(layer_info, args.budget_bits, args.q_bits, args.lr_bits, args.scale_bits)
    print(f"\nRealized: {realized:.4f} bits/param")
    print(f"Layers with LR: {sum(1 for k in allocations.values() if k > 0)}/{len(allocations)}")
    reconstruct_model(model, layer_info, allocations, covariances, args.grad_dir, args.hess_dir, args.q_bits)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    meta = {
        'method': 'BALANCER',
        'budget_bits': float(args.budget_bits),
        'realized_bits': float(realized),
        'q_bits': int(args.q_bits),
        'lr_bits': int(args.lr_bits),
        'total_layers': len(layer_info),
        'layers_with_lr': sum(1 for k in allocations.values() if k > 0),
    }
    with open(os.path.join(args.output_dir, "quant_meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved to {args.output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
