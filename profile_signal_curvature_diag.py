#!/usr/bin/env python3
import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def compute_hessian_diagonal(model, tokens, batch_size=1):
    hessians = {}
    nsamples = tokens.shape[0]
    
    for idx in tqdm(range(0, nsamples, batch_size), desc="Computing Hessian"):
        batch = tokens[idx:idx+batch_size].cuda()
        
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None and 'weight' in name:
                if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']):
                    grad_sq = param.grad.data.pow(2)
                    
                    if name not in hessians:
                        hessians[name] = torch.zeros(grad_sq.shape[0], device='cpu', dtype=torch.float32)
                    
                    if len(grad_sq.shape) == 2:
                        hessians[name] += grad_sq.mean(dim=1).cpu()
                    else:
                        hessians[name] += grad_sq.cpu()
        
        model.zero_grad()
        del batch, outputs, loss
        torch.cuda.empty_cache()
    
    for name in hessians:
        hessians[name] /= (nsamples / batch_size)
    
    return hessians

def save_hessian_by_layer(hessian_diag, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    component_map = {
        'self_attn.q_proj': 'q', 'self_attn.k_proj': 'k',
        'self_attn.v_proj': 'v', 'self_attn.o_proj': 'o',
        'mlp.up_proj': 'up', 'mlp.down_proj': 'down', 'mlp.gate_proj': 'gate'
    }
    
    saved = 0
    for name, hess in hessian_diag.items():
        if 'layers.' not in name:
            continue
        
        parts = name.split('.')
        layer_idx = None
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts):
                layer_idx = int(parts[i + 1])
                break
        
        if layer_idx is None:
            continue
        
        component = None
        for key, tag in component_map.items():
            if key in name:
                component = tag
                break
        
        if component is None:
            continue
        
        layer_file = os.path.join(save_path, f"{layer_idx}_{component}.pt")
        torch.save(hess, layer_file)
        saved += 1
    
    print(f"Saved {saved} Hessian diagonal files to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--output_dir", required=True, help="Output directory for Hessians")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Computing Hessian Diagonal")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Samples: {args.nsamples}, SeqLen: {args.seqlen}")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", use_cache=False
    )
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(data["text"][:5000])
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.nsamples * args.seqlen).input_ids
    ns = min(tokens.shape[1] // args.seqlen, args.nsamples)
    tokens = tokens[0, :ns * args.seqlen].view(ns, args.seqlen)
    
    print(f"Processing {ns} samples...")
    hessian_diag = compute_hessian_diagonal(model, tokens, args.batch_size)
    save_hessian_by_layer(hessian_diag, args.output_dir)
    
    print("Done!")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
