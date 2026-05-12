#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def eval_hellaswag(model, tokenizer, num_samples=None):
    data = load_dataset("hellaswag", split="validation")
    if num_samples:
        data = data.select(range(min(num_samples, len(data))))
    correct = 0
    for item in tqdm(data, desc="HellaSwag"):
        ctx = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])
        losses = []
        for end in endings:
            text = ctx + " " + end
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            losses.append(out.loss.item())
        pred = losses.index(min(losses))
        if pred == label:
            correct += 1
    return correct / len(data)

def eval_winogrande(model, tokenizer, num_samples=None):
    data = load_dataset("winogrande", "winogrande_xl", split="validation")
    if num_samples:
        data = data.select(range(min(num_samples, len(data))))
    correct = 0
    for item in tqdm(data, desc="WinoGrande"):
        sent = item["sentence"]
        opt1, opt2 = item["option1"], item["option2"]
        label = int(item["answer"]) - 1
        losses = []
        for opt in [opt1, opt2]:
            text = sent.replace("_", opt)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            losses.append(out.loss.item())
        pred = losses.index(min(losses))
        if pred == label:
            correct += 1
    return correct / len(data)

def eval_arc(model, tokenizer, split="easy", num_samples=None):
    subset = "ARC-Easy" if split == "easy" else "ARC-Challenge"
    data = load_dataset("ai2_arc", subset, split="test")
    if num_samples:
        data = data.select(range(min(num_samples, len(data))))
    correct = 0
    for item in tqdm(data, desc=f"ARC-{split}"):
        question = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer = item["answerKey"]
        answer_idx = labels.index(answer)
        losses = []
        for choice in choices:
            text = f"Question: {question}\nAnswer: {choice}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            losses.append(out.loss.item())
        pred = losses.index(min(losses))
        if pred == answer_idx:
            correct += 1
    return correct / len(data)

def eval_piqa(model, tokenizer, num_samples=None):
    data = load_dataset("piqa", split="validation")
    if num_samples:
        data = data.select(range(min(num_samples, len(data))))
    correct = 0
    for item in tqdm(data, desc="PIQA"):
        goal = item["goal"]
        sol1, sol2 = item["sol1"], item["sol2"]
        label = item["label"]
        losses = []
        for sol in [sol1, sol2]:
            text = f"{goal} {sol}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            losses.append(out.loss.item())
        pred = losses.index(min(losses))
        if pred == label:
            correct += 1
    return correct / len(data)

def eval_boolq(model, tokenizer, num_samples=None):
    data = load_dataset("boolq", split="validation")
    if num_samples:
        data = data.select(range(min(num_samples, len(data))))
    correct = 0
    for item in tqdm(data, desc="BoolQ"):
        passage = item["passage"]
        question = item["question"]
        label = item["answer"]
        losses = []
        for ans in ["no", "yes"]:
            text = f"{passage}\nQuestion: {question}\nAnswer: {ans}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            losses.append(out.loss.item())
        pred = losses.index(min(losses)) == 1
        if pred == label:
            correct += 1
    return correct / len(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", default="hellaswag,winogrande,piqa,arc_e,arc_c,boolq")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit samples per task (for quick testing)")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    tasks = [t.strip() for t in args.tasks.split(",")]
    results = {}
    
    for task in tasks:
        print(f"\nEvaluating {task}...")
        if task == "hellaswag":
            results[task] = eval_hellaswag(model, tokenizer, args.num_samples)
        elif task == "winogrande":
            results[task] = eval_winogrande(model, tokenizer, args.num_samples)
        elif task == "piqa":
            results[task] = eval_piqa(model, tokenizer, args.num_samples)
        elif task == "arc_e":
            results[task] = eval_arc(model, tokenizer, "easy", args.num_samples)
        elif task == "arc_c":
            results[task] = eval_arc(model, tokenizer, "challenge", args.num_samples)
        elif task == "boolq":
            results[task] = eval_boolq(model, tokenizer, args.num_samples)
        print(f"{task}: {results[task]*100:.2f}%")
    
    print("\n" + "="*40)
    print("Results:")
    for task, acc in results.items():
        print(f"  {task}: {acc*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
