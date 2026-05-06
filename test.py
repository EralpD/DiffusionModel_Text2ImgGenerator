import torch
import torchvision
import os
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer, CLIPTextModel

from train import Mini_UNet, EMA, sample_cfg_ddim
from show import show_images

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading CLIP model...")
    clip_model_id = "openai/clip-vit-large-patch14"
    CLIPmodel = CLIPTextModel.from_pretrained(clip_model_id).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_id)

    print("Loading Diffusion Model...")
    model = Mini_UNet(3).to(device)
    
    model_weights_path = "models/diffusion_model.pth"
    checkpoint_path = "models/diffusion_checkpointCIFAR.pth"

    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    else:
        print(f"Warning: {model_weights_path} not found. Attempting to proceed with initialized weights.")

    if os.path.exists(checkpoint_path):
        model_path = torch.load(checkpoint_path, map_location=device)
        epoch = model_path.get('epoch', 0)
        print(f"Model checkpoint with Epoch {epoch+1} loaded successfully.")
    else:
        print(f"Error: Checkpoint {checkpoint_path} not found. Please ensure the path is correct.")
        return

    while True:
        print("\n" + "="*50)
        print("If you want to exit, type -1")
        prompt_input = input("Enter your prompt to generate images: a photo of a ")
        
        if prompt_input.strip() == "-1":
            print("Exiting generator...")
            break
            
        prompt = "a photo of a " + prompt_input.strip()
        empty_prompt = ""

        inputs = tokenizer([prompt, empty_prompt], padding="max_length", max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            context = CLIPmodel(**inputs).last_hidden_state

        n_samples = 4
        cond_context = context[0:1]
        null_context = context[1:2]

        if n_samples > 1:
            cond_context = cond_context.repeat(n_samples, 1, 1)
            null_context = null_context.repeat(n_samples, 1, 1)

        ema = EMA(model)
        if 'ema_shadow' in model_path:
            ema.shadow = model_path['ema_shadow']
            ema.apply_shadow(model)
        else:
            print("Warning: No 'ema_shadow' found in checkpoint. Proceeding without EMA.")

        # Sample Images
        print(f"Generating {n_samples} samples (32x32)...")
        with torch.no_grad():
            samples = sample_cfg_ddim(
                model, 
                context=cond_context, 
                null_context=null_context, 
                n=n_samples, 
                channels=3, 
                size=32, 
                steps=100, 
                eta=0.0, # Deterministic sampling
                cfg_scale=7.5
            )

        if 'ema_shadow' in model_path:
            ema.restore(model)

        grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), interpolation='bicubic')
        plt.title("Diffusion Samples (32x32)")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()