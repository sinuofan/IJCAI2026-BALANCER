 BALANCER

This repository contains the code for reproducing the main experiments in our paper. The method compresses LLMs by combining quantization with metric-induced low-rank decomposition, using three signals (activation statistics, teacher gradients, and Hessian diagonal) to guide the compression.

 Dependencies

We tested on Python 3.10 with PyTorch 2.1. Install the required packages:

bash
pip install torch transformers datasets tqdm


You'll need a GPU with at least 24GB memory for LLaMA-2-7B (40GB+ recommended for LLaMA-3.1-8B).

 Quick Start

The simplest way to run the method is with activation statistics only:

bash
python BALANCER.py \
    --model meta-llama/Llama-2-7b-hf \
    --output_dir ./compressed_llama2_7b \
    --budget_bits 4.0


This will compress the model to ~4 bits per parameter and save it to `./compressed_llama2_7b`. You can then evaluate:

bash
 Perplexity
python eval_perplexity.py --model ./compressed_llama2_7b

 Downstream tasks
python eval_downstream_tasks.py --model ./compressed_llama2_7b --tasks hellaswag,winogrande,piqa,arc_e,arc_c,boolq


 Using All Three Signals

For the full method described in the paper, you need to pre-compute the gradient and curvature signals first. This takes some extra time but generally gives better results, especially at lower bit budgets.

 Step 1: Profile Gradient Signal

The gradient signal captures which weights are important for preserving the model's output distribution. We use self-distillation (the model is both teacher and student) with KL divergence loss.

bash
python profile_signal_gradients.py \
    --model meta-llama/Llama-2-7b-hf \
    --output_dir ./signals/llama2_7b/gradients \
    --nsamples 64 \
    --seqlen 512


This takes about 30-60 minutes on an A100. The script saves one `.pt` file per layer per block type, e.g., `0_q.pt` for layer 0's Q projection, `15_down.pt` for layer 15's MLP down projection, etc.

You can adjust `--nsamples` (more samples = more accurate but slower) and `--seqlen` (longer sequences need more memory). We found 64 samples with seqlen 512 works well in practice.

 Step 2: Profile Curvature Signal

The curvature signal approximates the diagonal of the Hessian matrix, telling us which output dimensions are most sensitive to perturbation.

bash
python profile_signal_curvature_diag.py \
    --model meta-llama/Llama-2-7b-hf \
    --output_dir ./signals/llama2_7b/curvature \
    --nsamples 128 \
    --seqlen 2048


This takes about 60-90 minutes. We use more samples here because the Hessian estimate can be noisy. The output files have the same naming convention as the gradients, but each file contains a 1D tensor (the diagonal) rather than a 2D matrix.

 Step 3: Run Compression with All Signals

Now you can run the full method:

bash
python BALANCER.py \
    --model meta-llama/Llama-2-7b-hf \
    --output_dir ./compressed_llama2_7b_3signal \
    --budget_bits 4.0 \
    --grad_dir ./signals/llama2_7b/gradients \
    --hess_dir ./signals/llama2_7b/curvature


The method will automatically load and fuse the three signals when computing the metric for each layer.

 Verifying Signal Files

If something seems off, you can check that the signal files were generated correctly:

python
import torch
import os

 Check a gradient file
g = torch.load("signals/llama2_7b/gradients/0_q.pt")
print(f"Gradient shape: {g.shape}")   [4096, 4096] for LLaMA-2-7B
print(f"Gradient mean abs: {g.abs().mean():.6f}")

 Check a curvature file  
h = torch.load("signals/llama2_7b/curvature/0_q.pt")
print(f"Curvature shape: {h.shape}")   [4096] - it's the diagonal
print(f"Curvature mean: {h.mean():.6f}")


The gradient tensors should have shape `[out_dim, in_dim]` matching the weight matrix. The curvature tensors should be 1D with length `out_dim`.

 Different Compression Budgets

For more aggressive compression (e.g., 3.6 bits), you might need to adjust some parameters:

bash
python BALANCER.py \
    --model meta-llama/Llama-2-7b-hf \
    --output_dir ./compressed_llama2_7b_36bit \
    --budget_bits 3.6 \
    --q_bits 4 \
    --k_max 128 \
    --grad_dir ./signals/llama2_7b/gradients \
    --hess_dir ./signals/llama2_7b/curvature


At very low budgets, reducing `--k_max` helps because high-rank corrections can become numerically unstable with aggressive quantization.

 Running on LLaMA-3.1-8B

Same procedure, just change the model path:

bash
 Profile signals
python profile_signal_gradients.py --model meta-llama/Llama-3.1-8B --output_dir ./signals/llama31_8b/gradients
python profile_signal_curvature_diag.py --model meta-llama/Llama-3.1-8B --output_dir ./signals/llama31_8b/curvature

 Compress
python BALANCER.py \
    --model meta-llama/Llama-3.1-8B \
    --output_dir ./compressed_llama31_8b \
    --budget_bits 4.0 \
    --grad_dir ./signals/llama31_8b/gradients \
    --hess_dir ./signals/llama31_8b/curvature


LLaMA-3.1-8B has a larger vocabulary and slightly different architecture but the code handles it automatically.

 Output Files

After compression, you'll get a standard HuggingFace model directory:


compressed_model/
├── config.json
├── generation_config.json
├── model.safetensors (or pytorch_model.bin)
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── quant_meta.json           our metadata (budget, realized bits, etc.)


The model can be loaded directly with `transformers`:

python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("./compressed_model")


 Troubleshooting

**Out of memory during signal profiling**: Try reducing `--nsamples` or `--seqlen`. You can also add `--batch_size 1` to the curvature script if needed.

**Compression takes too long**: The bottleneck is usually the SVD computation. You can speed it up by reducing `--k_max` (e.g., 128 instead of 256) at some cost to quality.

**Very high perplexity**: Make sure the signal files exist and have the right shapes. Also check that `--grad_dir` and `--hess_dir` point to the correct directories.
