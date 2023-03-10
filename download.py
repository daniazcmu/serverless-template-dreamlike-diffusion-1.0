# In this file, we define download_model
# It runs during container build time to get model weights built into the container
import os
import time
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionKDiffusionPipeline

def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time
    t1 = time.time()
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"
    model = StableDiffusionKDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    t2 = time.time()
    print("Download took - ",t2-t1,"seconds")

if __name__ == "__main__":
    download_model()
