from show import show_samples, show_images
import torch.nn as nn
import torch
import os
from train import Mini_UNet, EMA, sample_cfg_ddim
import matplotlib.pyplot as plt
import torchvision
from transformers import CLIPTokenizer, CLIPTextModel # CLIP for text encoding
from upscale import Mini_UpscalerUNet, sample_upscaler_cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIPmodel = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

model = Mini_UNet(3).to(device)
upModel = Mini_UpscalerUNet(input_channel=3).to(device)
model.load_state_dict(torch.load("models/diffusion_model.pth", map_location=device))
upModel.load_state_dict(torch.load("models/upscale_model2.pth", map_location=device))

model_path = torch.load("models/diffusion_checkpointCIFAR.pth", map_location=device)
upscaler_path = torch.load("models/upscale_checkpointCIFAR.pth", map_location=device)
epoch = model_path['epoch']
print(f"Model with Epoch {epoch+1} loaded successfully.")

upEpoch = upscaler_path['epoch']
print(f"Upscaler Model with Epoch {upEpoch+1} loaded successfully.")

while True:
    print("If you want to exit, type -1")
    prompt = input("Enter your prompt to generate images: a photo of a ")
    if prompt.strip() == "-1":
        break
    prompt = "a photo of a " + prompt.strip()
    empty_prompt = ""

    inputs = tokenizer([prompt, empty_prompt], padding="max_length", max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        context = CLIPmodel(**inputs)
        context = context.last_hidden_state

    n_samples = 4
    cond_context = context[0:1]
    null_context = context[1:2]

    if n_samples > 1:
        cond_context = cond_context.repeat(n_samples, 1, 1)
        null_context = null_context.repeat(n_samples, 1, 1)

    ema = EMA(model)
    ema.shadow = model_path['ema_shadow']

    ema.apply_shadow(model)

    with torch.no_grad():
        samples = sample_cfg_ddim(model, context=cond_context, null_context=null_context, n=n_samples, channels=3, size=32, steps=100, eta=0.0, cfg_scale=7.5) # Using DDIM sampling (eta=0.0: deterministic sampling)

    ema.restore(model)

    ema_upscaler = EMA(upModel)
    ema_upscaler.shadow = upscaler_path['ema_shadow']

    ema_upscaler.apply_shadow(upModel)

    with torch.no_grad():
        low_res_imgs = samples
        low_res_imgs = low_res_imgs * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        low_res_imgs = torchvision.transforms.functional.resize(low_res_imgs, (96, 96), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        upscaled_samples = sample_upscaler_cfg(upModel, cond_context=cond_context, null_context=null_context, n=n_samples, steps=100, cfg_scale=5.0, low_res_imgs=low_res_imgs)

    grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
    up_grid = torchvision.utils.make_grid(upscaled_samples, nrow=4, normalize=True)
    show_images(None, None, axises=[(grid, "Diffusion Samples (32x32)"), (up_grid, "Upscale Diffusion Samples (96x96)")])

