import os
import time
import torch
import base64
from io import BytesIO
from torch import autocast
from diffusers import StableDiffusionPipeline, KDPM2DiscreteScheduler

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    t1 = time.time()
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"
    device = "cuda" 
    model = StableDiffusionPipeline.from_pretrained(model_id, custom_pipeline="lpw_stable_diffusion")
    model.scheduler = KDPM2DiscreteScheduler.from_config(model.scheduler.config)
    model = model.to(device)
    t2 = time.time()
    print("Init took - ",t2-t1,"seconds")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 512)
    width = model_inputs.get('width', 768)
    negative = model_inputs.get('negative', None)
    steps = model_inputs.get('num_inference_steps', 50)
    guidance = model_inputs.get('guidance_scale', 7)
    
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    t1 = time.time()
    with autocast("cuda"):
        image = pipe(prompt, height=height, width=width, negative_prompt=negative, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=guidance, max_embeddings_multiples=4).images[0]
    t2 = time.time()
    print("Inference took - ",t2-t1,"seconds")
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}
