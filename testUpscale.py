import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision import transforms
from upscale import Mini_UpscalerUNet, sample_upscaler_cfg, BlurImage, test_ds, CLIPmodel as clip_model, tokenizer
# from upscale import Mini_UpscalerUNet, sample_upscaler_cfg, BlurImage, test_ds, CLIPmodel, tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)

def test_model(model_path="models/upscale_model2.pth"):
    model = Mini_UpscalerUNet(input_channel=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=4, shuffle=True)
    imgs, labels = next(iter(test_loader))
    imgs = imgs.to(device)
    n = imgs.size(0)

    prompts = [f"a photo of a {test_ds.classes[l]}" for l in labels]
    null_prompts = [""] * n
    
    with torch.no_grad():
        tokens_cond = tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=77).to(device)
        cond_c = clip_model(**tokens_cond).last_hidden_state
        
        tokens_null = tokenizer(null_prompts, return_tensors="pt", padding="max_length", max_length=77).to(device)
        null_c = clip_model(**tokens_null).last_hidden_state

    low_res_imgs = BlurImage(imgs)
    low_res_imgs = F.interpolate(low_res_imgs, size=(96, 96), mode='bilinear', align_corners=False)
    low_res_imgs = low_res_imgs * 2.0 - 1.0  # [0, 1] -> [-1, 1] (Model waits for inputs in [-1, 1])

    samples = sample_upscaler_cfg(
        model=model, 
        n=n, 
        low_res_imgs=low_res_imgs, 
        cond_context=cond_c, 
        null_context=null_c, 
        steps=50,
        cfg_scale=0.0
    )

    fig, axes = plt.subplots(3, n, figsize=(15, 10))
    fig.suptitle("Diffusion Upscaler Results", fontsize=16)

    for i in range(n):
        orig_img = imgs[i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"Original: {test_ds.classes[labels[i]]}")
        axes[0, i].axis('off')

        lr_img = (low_res_imgs[i] + 1) / 2  # Denormalize to [0, 1] for visualization
        lr_img = lr_img.permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(lr_img)
        axes[1, i].set_title("Input (Agressively Blurred)")
        axes[1, i].axis('off')

        gen_img = samples[i].permute(1, 2, 0).cpu().numpy()
        axes[2, i].imshow(gen_img)
        axes[2, i].set_title("Model Output (Upscaled)")
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_model("models/upscale_model.pth")