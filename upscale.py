import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from torchvision.datasets import STL10
from train import SinusoidalPosEmb, ResBlock, SelfAttention, CrossAttention, predict_v, CLIPmodel, tokenizer, save_checkpoint, EMA, sample_cfg_ddim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

path = "upscaler_checkpoint.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 50
def cosine_beta_schedule(T, s=.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    This schedule produces samples and smoother denoising.
    """
    x = torch.linspace(0, T, T+1)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999)

T = 200

betas = cosine_beta_schedule(T).to(device)
alphas = 1. - betas.to(device)
alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)

def forward_diffusion_sample(x_0, t, noise=None):
    """
    Add noise to the image x_0 at timestep t
    """

    if noise is None:
        noise = torch.randn_like(x_0)
    
    alpha_bar_t = alphas_cumprod.gather(0, t).view(-1,1,1,1)
    sqrt_one_minus_alphas_bar_t = torch.sqrt(1 - alpha_bar_t)

    return torch.sqrt(alpha_bar_t) * x_0 + sqrt_one_minus_alphas_bar_t * noise, noise

class FiLM(nn.Module):
    def __init__(self, cond_dim, out_dim):
        super(FiLM, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(cond_dim, out_dim * 2)
        )

    def forward(self, h, cond):
        gamma_beta = self.linear(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * h + beta
    
class SpatialFiLM(nn.Module):
    def __init__(self, in_channels, cond_channels):
        super().__init__()
        self.conv = nn.Conv2d(cond_channels, in_channels * 2, 1)
        self.res_proj = nn.Conv2d(cond_channels, in_channels, 1)

    def forward(self, h, cond_feat):
        gamma, beta = self.conv(cond_feat).chunk(2, dim=1)
        cond_proj = self.res_proj(cond_feat)
        h = h + cond_proj # Adding the residual connection for better conditioning
        return (1 + gamma) * h + beta

    
class LowResFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

    def forward(self, x):
        f1 = F.gelu(self.conv1(x))   # (B, 64, H, W)
        f2 = F.gelu(self.conv2(f1))  # (B, 128, H/2, W/2)
        f3 = F.gelu(self.conv3(f2))  # (B, 256, H/4, W/4)

        return f1, f2, f3

class Mini_UpscalerUNet(nn.Module):
    def __init__(self, input_channel,time_emb_dim=256):
        super(Mini_UpscalerUNet, self).__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(), # x ∗ Φ(x) (Φ(x) is Gaussian CDF (Cumulative Distribution Function))
            nn.Linear(time_emb_dim, time_emb_dim) 
        )

        self.low_res_encoder = LowResFeatureExtractor()

        # Global (time)
        self.t_film1 = FiLM(256, 64)
        self.t_film2 = FiLM(256, 128)
        self.t_film3 = FiLM(256, 256)
        self.t_film4 = FiLM(256, 768)

        # Spatial (low-res)
        self.s_film1 = SpatialFiLM(64, 64)
        self.s_film2 = SpatialFiLM(128, 128)
        self.s_film3 = SpatialFiLM(256, 256)
        self.s_film4 = SpatialFiLM(768, 256)
                
        self.rb1 = ResBlock(input_channel, 64, time_emb_dim)
        self.cr_attn = CrossAttention(64, context_dim=768)
        self.rb2 = ResBlock(64, 64, time_emb_dim)
        self.down1 = nn.Conv2d(64, 64, 4, stride=2, padding=1)

        self.rb3 = ResBlock(64, 128, time_emb_dim)
        self.cr_attn2 = CrossAttention(128, context_dim=768)
        self.rb4 = ResBlock(128, 128, time_emb_dim)
        self.down2 = nn.Conv2d(128, 128, 4, stride=2, padding=1)

        self.rb5 = ResBlock(128, 256, time_emb_dim)
        self.cr_attn3 = CrossAttention(256, context_dim=768)
        self.rb6 = ResBlock(256, 256, time_emb_dim)
        self.down3 = nn.Conv2d(256, 256, 4, stride=2, padding=1)

        self.rb7 = ResBlock(256, 768, time_emb_dim)
        self.cr_attn4 = CrossAttention(768, context_dim=768)
        self.attn = SelfAttention(768, 16)
        self.rb8 = ResBlock(768, 768, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(768, 256, 2, stride=2)
        self.rb9 = ResBlock(512, 256, time_emb_dim)
        self.rb10 = ResBlock(256, 256, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rb11 = ResBlock(256, 128, time_emb_dim)
        self.rb12 = ResBlock(128, 128, time_emb_dim)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rb13 = ResBlock(128, 64, time_emb_dim)
        self.rb14 = ResBlock(64, 64, time_emb_dim)

        self.out = nn.Conv2d(64, input_channel, 1)


    def forward(self, x_t, x_low_res, context, t):

        t_emb = self.time_mlp(t)

        # x_low_res = F.interpolate(x_low_res, size=(128, 128), mode='bilinear', align_corners=False)

        # Encoder
        if x_low_res.shape[-2:] != x_t.shape[-2:]:
            x_low_res = F.interpolate(
                x_low_res, 
                size=(x_t.shape[2], x_t.shape[3]), 
                mode='bilinear', 
                align_corners=False
            )

        f1, f2, f3 = self.low_res_encoder(x_low_res)

        # x = torch.cat([x_t, x_low_res], dim=1)
        x = x_t
        h1 = self.rb1(x, t_emb)
        h1 = self.t_film1(h1, t_emb)
        f1 = F.interpolate(f1, size=h1.shape[-2:])
        h1 = self.s_film1(h1, f1)
        h1 = h1 + self.cr_attn(h1, context)
        h1 = self.rb2(h1, t_emb)
        h = self.down1(h1)

        h2 = self.rb3(h, t_emb)
        h2 = self.t_film2(h2, t_emb)
        f2 = F.interpolate(f2, size=h2.shape[-2:])
        h2 = self.s_film2(h2, f2)
        h2 = h2 + self.cr_attn2(h2, context)
        h2 = self.rb4(h2, t_emb)
        h = self.down2(h2)

        h3 = self.rb5(h, t_emb)
        h3 = self.t_film3(h3, t_emb)
        f3 = F.interpolate(f3, size=h3.shape[-2:])
        h3 = self.s_film3(h3, f3)
        h3 = h3 + self.cr_attn3(h3, context)
        h3 = self.rb6(h3, t_emb)
        h = self.down3(h3)

        # Bottleneck
        f3_down = F.interpolate(f3, size=h.shape[-2:], mode='bilinear', align_corners=False)
        
        h = self.rb7(h, t_emb)
        h = self.t_film4(h, t_emb)
        h = self.s_film4(h, f3_down)
        h = self.attn(h)
        h = h + self.cr_attn4(h, context)

        h = self.rb8(h, t_emb)

        # Decoder
        h = self.up1(h)
        h = torch.cat([h, h3], dim=1)
        h = self.rb9(h, t_emb)
        h = self.s_film3(h, f3)
        h = self.rb10(h, t_emb)

        h = self.up2(h)
        h = torch.cat([h, h2], dim=1)
        h = self.rb11(h, t_emb)
        h = self.s_film2(h, f2)
        h = self.rb12(h, t_emb)

        h = self.up3(h)
        h = torch.cat([h, h1], dim=1)
        h = self.rb13(h, t_emb)
        h = self.s_film1(h, f1)
        h = self.rb14(h, t_emb)

        return self.out(h)
    
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*vgg[:4])
        self.slice2 = nn.Sequential(*vgg[4:9])
        self.slice3 = nn.Sequential(*vgg[9:16])

        for param in self.parameters():
            param.requires_grad = False
        
    def forward(self, x, y):

        h1_x = self.slice1(x)
        h1_y = self.slice1(y)
        h2_x = self.slice2(h1_x)
        h2_y = self.slice2(h1_y)
        h3_x = self.slice3(h2_x)
        h3_y = self.slice3(h2_y)

        loss = torch.mean((h1_x - h1_y) ** 2) + torch.mean((h2_x - h2_y) ** 2) + torch.mean((h3_x - h3_y) ** 2)
        return loss
    
train_transform = transforms.Compose([
    # transforms.Resize((128, 128)),
    # transforms.RandomCrop((64,64)),
    # 96x96 resolution will be used, uncomment when 128x128 is needed
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    # transforms.Resize((128, 128)),
    # 96x96 resolution will be used, uncomment when 128x128 is needed
    transforms.ToTensor()
])

train_ds = STL10(root='../../Datasets/data', split="train", download=True, transform=train_transform) 
test_ds = STL10(root='../../Datasets/data', split="test", download=True, transform=test_transform)

vgg_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def NormalizeForVGG(img):
    
    img = img.clamp(0, 1)

    img = F.interpolate(img, size=(96, 96), mode='bilinear', align_corners=False)

    img = vgg_normalization(img)
    return img

def BlurImage(img):
    sigma = torch.empty(1).uniform_(0.1, 2.0).item()
    img = transforms.functional.gaussian_blur(img, kernel_size=5, sigma=sigma)
    
    img = transforms.functional.resize(img, (32, 32), 
                                       interpolation=transforms.InterpolationMode.BILINEAR)
    
    img = img + torch.randn_like(img) * 0.05     
    
    return img.clamp(0, 1)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.to(a.device))
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def predict_x0_from_v(x_t, t, v_pred):
    alpha_bar_t = extract(alphas_cumprod, t, x_t.shape)
    return torch.sqrt(alpha_bar_t) * x_t - torch.sqrt(1 - alpha_bar_t) * v_pred

def train(model, dataloader, optimizer, ema, scheduler, device, epochs):
    model.train()
    CLIPmodel.eval().to(device)
    vgg_criterion = PerceptualLoss().to(device)

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for img, labels in dataloader:
            img = img.to(device)
            b = img.size(0)
            low_res_img = torch.stack([BlurImage(img[i]) for i in range(b)])
            low_res_img = low_res_img * 2.0 - 1.0
            x_0 = img * 2 - 1

            if low_res_img.shape[-2:] != x_0.shape[-2:]:
                low_res_img = F.interpolate(
                    low_res_img, 
                    size=(x_0.shape[2], x_0.shape[3]), 
                    mode='bilinear', 
                    align_corners=False
                )

            t = torch.randint(0, T, (b,), device=device)
            prompts = [f"a photo of a {test_ds.classes[l]}" for l in labels]

            mask = torch.rand(b, device=device) < 0.1
            prompts = ["" if m else p for p, m in zip(prompts, mask)]

            with torch.no_grad():
                text_emb = tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=77).to(device)
                out = CLIPmodel(**text_emb)
                context = out.last_hidden_state

            noisy_img, noise = forward_diffusion_sample(x_0, t)
            alpha_bar_t = extract(alphas_cumprod, t, x_0.shape)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

            v_target = sqrt_alpha_bar_t * noise - sqrt_one_minus_alpha_bar_t * x_0 # v-target for v-prediction parameterization
            with autocast():
                pred = model(noisy_img, low_res_img, context, t)

                # MSE Loss
                loss = F.mse_loss(pred, v_target)

                # Perceptual Loss (using VGG16)
                # if epoch > 7:
                x0_pred_norm = predict_x0_from_v(noisy_img, t, pred)

                x0_pred_vgg = (x0_pred_norm + 1) / 2  # Denormalize to [0, 1] for VGG
                x0_pred_vgg = NormalizeForVGG(x0_pred_vgg)
                target_img_vgg = NormalizeForVGG(img)

                perceptual_loss = vgg_criterion(x0_pred_vgg, target_img_vgg)
                total_loss = (
                    loss
                    + 0.05 * perceptual_loss
                    + 0.5 * F.l1_loss(x0_pred_norm, x_0)
                )

                # else:
                    # total_loss = loss

            optimizer.zero_grad()

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

        scheduler.step()

        if total_loss is None:
            raise ValueError("Total loss is None. Check the loss computation in the training loop.")

        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}, Loss: {total_loss.item()}, Current Learning Rate (LR): {current_lr}")
            save_checkpoint(model, ema, optimizer, scheduler, epoch=epoch, loss=total_loss, path="models/upscale_checkpointCIFAR.pth")

    return total_loss.item()

@torch.no_grad()
def sample_upscaler_cfg(model, n, low_res_imgs, cond_context, null_context, steps=50, cfg_scale=7.5):
    model.eval()
    ts = torch.linspace(0, T-1, steps).long().to(device)
    x = torch.randn((n, 3, 96, 96)).to(device)
    
    for i in tqdm(reversed(range(steps)), desc="Upscaler Sampling"):
        t = ts[i]
        t_prev = ts[i-1] if i > 0 else torch.tensor(-1).to(device)
        t_tensor = torch.full((n,), t, device=device, dtype=torch.long)
        
        # Classifier-Free Guidance (CFG)
        v_cond = model(x, low_res_imgs, cond_context, t_tensor)
        v_uncond = model(x, low_res_imgs, null_context, t_tensor)
        
        v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        # DDIM / v-prediction
        alpha_bar = alphas_cumprod[t]
        alpha_bar_prev = alphas_cumprod[t_prev] if i > 0 else torch.tensor(1.0).to(device)
        
        noise_pred = torch.sqrt(alpha_bar) * v_pred + torch.sqrt(1 - alpha_bar) * x
        x0_pred = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
        
        dir_xt = torch.sqrt(1 - alpha_bar_prev) * noise_pred
        x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt

    return (x.clamp(-1, 1) + 1) / 2

if __name__ == "__main__":
    model = Mini_UpscalerUNet(input_channel=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ema_decay = .999
    ema = EMA(model, decay=ema_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    scaler = GradScaler()
    loss = train(model, train_dataloader, optimizer, ema, scheduler, device, epochs)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    imgs,labels = next(iter(test_loader))
    imgs = imgs.to(device)
    n = imgs.size(0)

    prompts = [f"a photo of a {test_ds.classes[l]}" for l in labels]
    null_prompts = [""] * n # Empty list for null prompts (unconditional)

    with torch.no_grad():
        tokens_cond = tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=77).to(device)
        cond_c = CLIPmodel(**tokens_cond).last_hidden_state
        
        tokens_null = tokenizer(null_prompts, return_tensors="pt", padding="max_length", max_length=77).to(device)
        null_c = CLIPmodel(**tokens_null).last_hidden_state


    low_res_imgs = BlurImage(imgs)
    low_res_imgs = F.interpolate(low_res_imgs, size=(96, 96), mode='bilinear', align_corners=False)

    low_res_imgs = low_res_imgs * 2.0 - 1.0 # Normalize to [-1, 1]

    ema.apply_shadow(model)
    samples = sample_upscaler_cfg(
        model=model, 
        n=n, 
        low_res_imgs=low_res_imgs, 
        cond_context=cond_c, 
        null_context=null_c, 
        steps=100, 
        cfg_scale=5.0 # 2.0-5.0 range is usually good for CFG scale, adjust based on your preference for fidelity vs diversity
    )
    ema.restore(model)
    grid = torchvision.utils.make_grid(samples, nrow=4)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(f"Generated Images for Prompts: {[test_ds.classes[l] for l in labels]}")
    plt.axis('off')

    plt.figure(figsize=(8, 8))
    plt.title("Sampled Images from the Diffusion Model")
    for i in range(16):
        ax, fig = plt.subplot(4,4,i+1), plt.gcf()
        index = torch.randint(0, len(test_ds), (1,)).item()
        img, label = test_ds[index]
        # img = (img + 1) / 2
        img = img.permute(1, 2, 0)
        ax.imshow(img.numpy())
        ax.axis('off')

    plt.show()

    torch.save(model.state_dict(), "models/upscale_model2.pth")
    print("Model weights saved to models/upscale_model2.pth")
