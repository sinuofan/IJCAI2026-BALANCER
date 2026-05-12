#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

@torch.no_grad()
def evaluate_ppl(model_path, dataset="wikitext2", seqlen=2048, stride=512):
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if dataset == "wikitext2":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(data["text"])
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")
    enc = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    nsamples = (enc.shape[1] - seqlen) // stride + 1
    print(f"Tokens: {enc.shape[1]}, Samples: {nsamples}")
    model.eval()
    nlls = []
    for i in tqdm(range(nsamples), desc="Evaluating"):
        start = i * stride
        end = start + seqlen
        if end > enc.shape[1]:
            break
        batch = enc[:, start:end]
        out = model(batch, labels=batch)
        nlls.append(out.loss)
    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"\nPerplexity: {ppl:.4f}")
    return float(ppl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    args = parser.parse_args()
    evaluate_ppl(args.model, args.dataset, args.seqlen, args.stride)
