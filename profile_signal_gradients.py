#!/usr/bin/env python3
import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def tag2mod(layer):
    return {
        "q": layer.self_attn.q_proj, "k": layer.self_attn.k_proj,
        "v": layer.self_attn.v_proj, "o": layer.self_attn.o_proj,
        "up": layer.mlp.up_proj, "down": layer.mlp.down_proj,
        "gate": getattr(layer.mlp, 'gate_proj', None),
    }

@torch.no_grad()
def teacher_probs(teacher, input_ids, T=1.0):
    out = teacher(input_ids=input_ids, use_cache=False)
    return torch.softmax(out.logits / T, dim=-1)

def kd_loss(student, input_ids, p_t, T=1.0):
    out = student(input_ids=input_ids, use_cache=False)
    log_p_s = torch.log_softmax(out.logits / T, dim=-1)
    return torch.nn.functional.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model (used as both teacher and student)")
    parser.add_argument("--output_dir", required=True, help="Output directory for gradients")
    parser.add_argument("--nsamples", type=int, default=64, help="Number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=512, help="Sequence length")
    parser.add_argument("--temperature", type=float, default=1.0, help="KD temperature")
    parser.add_argument("--blocks", type=str, default="q,k,v,o,up,down,gate", help="Target blocks")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
    
    print("=" * 60)
    print("Generating Teacher Gradients (KD)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Samples: {args.nsamples}, SeqLen: {args.seqlen}")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    
    teacher.eval()
    student.eval()
    student.config.use_cache = False
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
    
    for layer in student.model.layers:
        mods = tag2mod(layer)
        for tag, mod in mods.items():
            if mod is not None:
                mod.weight.requires_grad_(tag in blocks)
    
    buffers = {}
    for i, layer in enumerate(student.model.layers):
        for tag, mod in tag2mod(layer).items():
            if tag in blocks and mod is not None:
                buffers[(i, tag)] = torch.zeros_like(mod.weight, dtype=torch.float32, device="cpu")
    
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(data["text"][:5000])
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.nsamples * args.seqlen).input_ids
    ns = min(tokens.shape[1] // args.seqlen, args.nsamples)
    tokens = tokens[0, :ns * args.seqlen].view(ns, args.seqlen)
    
    print(f"Processing {ns} samples...")
    for b in tqdm(range(ns), desc="KD-Grad"):
        input_ids = tokens[b:b+1].cuda()
        
        with torch.no_grad():
            p_t = teacher_probs(teacher, input_ids, args.temperature)
        
        for p in student.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad = None
        
        loss = kd_loss(student, input_ids, p_t, args.temperature)
        loss.backward()
        
        for i, layer in enumerate(student.model.layers):
            for tag, mod in tag2mod(layer).items():
                if tag in blocks and mod is not None and mod.weight.grad is not None:
                    buffers[(i, tag)] += mod.weight.grad.detach().cpu()
        
        del input_ids, p_t, loss
        torch.cuda.empty_cache()
    
    print("Saving gradients...")
    for (i, tag), G in buffers.items():
        G /= float(ns)
        torch.save(G, os.path.join(args.output_dir, f"{i}_{tag}.pt"))
    
    print(f"Saved {len(buffers)} gradient files to {args.output_dir}")
    print("Done!")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
