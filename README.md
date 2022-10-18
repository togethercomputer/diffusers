# Diffusers + FlashAttention

This is a branch of [HuggingFace Diffusers](https://github.com/huggingface/diffusers) to incorporate FlashAttention, optimized for high throughput.

## Installation

The FlashAttention implementation in this repo depends on the experimental [cutlass](https://github.com/HazyResearch/flash-attention/tree/cutlass) branch.
FlashAttention requires CUDA 11, NVCC, and a Turing or Ampere GPU.

To install FlashAttention:
```
git clone https://github.com/HazyResearch/flash-attention.git
cd flash-attention
git checkout cutlass
git submodule init
git submodule update
python setup.py install
cd ..
```

To install diffusers:
```
git clone https://github.com/HazyResearch/diffusers.git
cd diffusers
pip install -e .
```

## Running

A sample benchmark, following HuggingFace's [benchmark](https://twitter.com/Nouamanetazi/status/1576959648912973826) of diffusers:
```Python
import time
import torch
from diffusers import StableDiffusionPipeline
import functools

# torch disable grad
torch.set_grad_enabled(False)

torch.manual_seed(1231)
torch.cuda.manual_seed(1231)

prompt = "a photo of an astronaut riding a horse on mars"

# cudnn benchmarking
torch.backends.cudnn.benchmark = True

# make sure you're logged in with `huggingface-cli login`
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16
).to("cuda")

batch_size = 10

# warmup
with torch.inference_mode():
    image = pipe([prompt] * batch_size, num_inference_steps=5).images[0]

for _ in range(3):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.inference_mode():
        image = pipe([prompt] * batch_size, num_inference_steps=50).images[0]
    torch.cuda.synchronize()
    print(f"Pipeline inference took {time.time() - start_time:.2f} seconds")
```
