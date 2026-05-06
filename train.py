import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
from collections import OrderedDict
from torchinfo import summary
from textToImage import CrossAttention, CLIPmodel, tokenizer
import os
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# def linear_beta_schedule(timesteps):

    # beta_start = 1e-4
    # beta_end = .02
    # return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(T, s=.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    This schedule produces samples and smoother denoising.
    """
    x = torch.linspace(0, T, T+1) 
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2 # f(t)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0] # α¯t
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]) # βt = 1 - α¯t / α¯t-1
    return betas.clamp(0.0001, 0.9999)

T = 1000

betas = cosine_beta_schedule(T).to(device)
alphas = 1. - betas.to(device)
alphas_cumprod = torch.cumprod(alphas, axis=0)

def forward_diffusion_sample(x_0, t, noise=None):
    """
    Add noise to the image x_0 at timestep t
    """

    if noise is None:
        noise = torch.randn_like(x_0)
    
    alpha_bar_t = alphas_cumprod.gather(0, t).view(-1,1,1,1).to(device)
    sqrt_one_minus_alphas_bar_t = torch.sqrt(1 - alpha_bar_t)

    return torch.sqrt(alpha_bar_t) * x_0 + sqrt_one_minus_alphas_bar_t * noise, noise

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2

        emb = torch.exp(
            torch.arange(half_dim, device=device) *
            -(math.log(10000) / (half_dim - 1))
        )

        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        return emb  # [B, dim]
    
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = x + attention_value

        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(32, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(32, out_channels)

        self.time_mlp = nn.Linear(time_dim, out_channels * 2)

        self.residual = (
             nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t_emb):
        x_residual = self.residual(x)
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.gn2(h)

        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        scale = scale.view(scale.shape[0], scale.shape[1], 1, 1) # [B, C, 1, 1]
        shift = shift.view(shift.shape[0], shift.shape[1], 1, 1) # [B, C, 1, 1]
        h = h * (scale + 1) + shift
        h = F.silu(h)

        return h + x_residual

class Mini_UNet(nn.Module):
    def __init__(self, input_channel,time_emb_dim=256):
        super(Mini_UNet, self).__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(), # x ∗ Φ(x) (Φ(x) is Gaussian CDF (Cumulative Distribution Function))
            nn.Linear(time_emb_dim, time_emb_dim) # must be stable 8, as it is proper with encoder channel size
        )
        
        self.rb1 = ResBlock(input_channel, 64, time_emb_dim)
        self.rb2 = ResBlock(64, 64, time_emb_dim)
        self.down1 = nn.Conv2d(64, 64, 4, stride=2, padding=1)

        self.rb3 = ResBlock(64, 128, time_emb_dim)
        self.attn = SelfAttention(128, 16)
        self.cr_attn = CrossAttention(128, context_dim=768)
        self.rb4 = ResBlock(128, 128, time_emb_dim)
        self.down2 = nn.Conv2d(128, 128, 4, stride=2, padding=1)

        self.rb5 = ResBlock(128, 256, time_emb_dim)
        self.cr_attn2 = CrossAttention(256, context_dim=768)
        self.rb6 = ResBlock(256, 256, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rb7 = ResBlock(256, 128, time_emb_dim)
        self.cr_attn3 = CrossAttention(128, context_dim=768)
        self.rb8 = ResBlock(128, 128, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rb9 = ResBlock(128, 64, time_emb_dim)
        self.rb10 = ResBlock(64, 64, time_emb_dim)

        self.out = nn.Conv2d(64, input_channel, 1)


    def forward(self, x, context, t):
        """
        Mini U-Net Time Embedded Architecture with ADM structure
        """

        # Get time embedding: [batch_size, time_emb_dim]
        t_emb = self.time_mlp(t)

        # Encoder
        h1 = self.rb1(x, t_emb)
        h1 = self.rb2(h1, t_emb)
        h = self.down1(h1)

        h2 = self.rb3(h, t_emb)
        h2 = self.attn(h2) # Self-Attention Layer (it is more effective on 16x16 feature maps)
        h2 = h2 + self.cr_attn(h2, context)
        h2 = self.rb4(h2, t_emb)
        h = self.down2(h2)

        # Bottleneck
        h = self.rb5(h, t_emb)
        h = h + self.cr_attn2(h, context)
        h = self.rb6(h, t_emb)

        # Decoder
        h = self.up1(h)
        h = torch.cat([h, h2], dim=1)
        h = self.rb7(h, t_emb)
        h = h + self.cr_attn3(h, context)
        h = self.rb8(h, t_emb)

        h = self.up2(h)
        h = torch.cat([h, h1], dim=1)
        h = self.rb9(h, t_emb)
        h = self.rb10(h, t_emb)

        return self.out(h)

    

transform = transforms.Compose([
        #! Open the Rotataion and Affine for more easier datasets, to generate CIFAR-10 like images, the generated images can be generated rotated or affined.
        # transforms.RandomRotation(15),
        # transforms.RandomAffine(0, translate=(0.1,0.1)),

        ## For ImageNet dataset, resizing to 128x128 might be better, but for faster experimentation, 32x32 is used here.
        # transforms.Resize(128),
        # transforms.CenterCrop(128), # make exact square images ()
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        lambda x: x*2-1
])

# dataset = FashionMNIST(root="./data", train=True, transform=transform, download=True)
dataset = torchvision.datasets.CIFAR10(
    root='../../Datasets/data',
    train=True, # CIFAR-10
    download=True, # CIFAR-10
    # split='val', # ImageNet
    transform=transform
)

def get_data():
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    return dataloader

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])

epochs = 100

def predict_v(x0, t, noise):
    """
    This helper function is a alternative loss function formulation for training the diffusion model.
    Why predict v? Because the intitution is related to signal scaling during the diffusion process.
    Predicting v can be interpreted as predicting a target that has a more consistent variance across different timesteps, potentially making the learning task easier or more stable.
    https://apxml.com/courses/advanced-diffusion-architectures/chapter-4-advanced-diffusion-training/advanced-loss-functions
    """
    alpha_bar = alphas_cumprod.gather(0, t).view(-1,1,1,1).to(device)
    return torch.sqrt(alpha_bar) * noise - torch.sqrt(1 - alpha_bar) * x0

def train(model, clip_model, dataloader, optimizer, ema, scheduler, epochs=epochs):

    loss = None
    model.train()
    clip_model.eval().to(device) 
    scaler = GradScaler()

    for epoch in tqdm(range(epochs)):
        for step, (x, labels) in enumerate(dataloader):
            x = x.to(device)
            b = x.shape[0]
            prompts = [f"a photo of a {dataset.classes[l]}" for l in labels]
            mask = torch.rand(b, device=device) < 0.1 # 10% of the time, use null prompts for classifier-free guidance (unconditional training)
            prompts = ["" if m else p for p, m in zip(prompts, mask)]

            with torch.no_grad():
                text_inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=77).to(device)
                out = clip_model(**text_inputs)
                context = out.last_hidden_state # [B, seq_len, 768]

            t = torch.randint(0, T, (b,), device=device).long()
            x_noisy, noise = forward_diffusion_sample(x, t)

            optimizer.zero_grad()

            with autocast():

                v_target = predict_v(x, t, noise)
                v_pred = model(x_noisy, context, t)
                loss = F.mse_loss(v_pred, v_target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update() 

            ema.update(model)
        
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}, Current Learning Rate (LR): {current_lr}")
            save_checkpoint(model, ema, optimizer, scheduler, epoch=epoch, loss=loss, path="models/diffusion_checkpointCIFAR.pth") # Saving periodically
            
    return loss.item()
    
# @torch.no_grad()
# def sample_ddim(model, n, channels=3, size=32, steps=50, eta=0.0):
#     """
#     DDIM Sampling: Generates images in fewer steps (e.g., 50) 
#     instead of the full T (e.g., 1000).
#     """
#     model.eval()
    
#     # Create a linear schedule of timesteps (e.g., [0, 20, 40... 980])
#     ts = torch.linspace(0, T-1, steps).long().to(device)
    
#     x = torch.randn((n, channels, size, size)).to(device)
    
#     for i in tqdm(reversed(range(steps)), desc="DDIM Sampling"):
#         t = ts[i] # Current timestep
#         t_prev = ts[i-1] if i > 0 else torch.tensor(-1).to(device) # Previous timestep
        
#         t_tensor = torch.full((n,), t, device=device, dtype=torch.long)
#         v_pred = model(x, t_tensor) # Predict v
        
#         alpha_bar = alphas_cumprod[t]
#         noise_pred = torch.sqrt(alpha_bar) * v_pred + torch.sqrt(1 - alpha_bar) * x # Convert v to noise ϵ
#         alpha_bar_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(device)
        
#         sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))
        
#         x0_pred = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
        
#         dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * noise_pred
        
#         if i > 0:
#             noise = torch.randn_like(x)
#         else:
#             noise = 0
            
#         x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise
#     if channels == 3:
#         return (x.clamp(-1, 1) + 1) / 2
#     elif channels == 1:
#         return x.clamp(-1, 1)
    
@torch.no_grad()
def sample_cfg_ddim(model, n, context, null_context, channels=3, size=32, steps=50, eta=0.0, cfg_scale=7.5):
    model.eval()
    # Create a linear schedule of timesteps (e.g., [0, 20, 40... 980])
    ts = torch.linspace(0, T-1, steps).long().to(device)
    x = torch.randn((n, channels, size, size)).to(device)
    
    for i in tqdm(reversed(range(steps)), desc="CFG DDIM Sampling"):
        t = ts[i] # Current timestep
        t_prev = ts[i-1] if i > 0 else torch.tensor(-1).to(device) # Previous timestep

        t_tensor = torch.full((n,), t, device=device, dtype=torch.long)
        v_cond = model(x, context, t_tensor) # Conditional prediction
        v_uncond = model(x, null_context, t_tensor) # Unconditional prediction

        v_pred = v_uncond + cfg_scale * (v_cond - v_uncond) # Classifier-free guidance formula

        alpha_bar = alphas_cumprod[t]
        noise_pred = torch.sqrt(alpha_bar) * v_pred + torch.sqrt(1 - alpha_bar) * x # Convert v to noise ϵ (DDIM formulation)
        alpha_bar_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(device)
        
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))
        
        x0_pred = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
        
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * noise_pred
        
        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0
            
        x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

    return (x.clamp(-1, 1) + 1) / 2


def save_checkpoint(model, ema, optimizer, scheduler, epoch, loss, path="diffusion_checkpointCIFAR.pth"):
    """
    Saving all of the model states to a file.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_shadow': ema.shadow, # EMA weights are crucial!
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint has been saved: {path} (Epoch: {epoch+1})")

def load_checkpoint(model, ema, optimizer, scheduler, path="diffusion_checkpointCIFAR.pth"):
    """
    Loads the model states from saved file.
    """
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema.shadow = checkpoint['ema_shadow']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Checkpoint loaded! Resuming from Epoch {checkpoint['epoch']+1}")
        return checkpoint['epoch'] + 1
    else:
        print("No saved model found, starting from scratch.")
        return 0
    
if __name__ == "__main__":
    model = Mini_UNet(3).to(device)

    # batch_size = 128
    # summary(model, input_size=[(batch_size, 3, 32, 32), (batch_size,)], dtypes = [torch.float, torch.long], device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) # Decreasing lr with cosine annealing schedule for better convergence
    dataloader = get_data()
    ema_decay = .999
    ema = EMA(model, decay=ema_decay)
    loss = train(model, CLIPmodel, dataloader, optimizer, ema, scheduler, epochs=epochs)

    prompt = "a photo of a car"
    empty_prompt = ""
    text_inputs = tokenizer([prompt, empty_prompt], return_tensors="pt", padding="max_length", max_length=77).to(device)
    with torch.no_grad():
        embeddings = CLIPmodel(text_inputs.input_ids).last_hidden_state

    cond_context = embeddings[0:1].repeat(16, 1, 1)
    null_context = embeddings[1:2].repeat(16, 1, 1)

    ema.apply_shadow(model)
    samples = sample_cfg_ddim(model, n=16, context=cond_context, null_context=null_context, channels=3, size=32, steps=50, eta=0.0, cfg_scale=7.5)
    ema.restore(model)
    grid = torchvision.utils.make_grid(samples, nrow=4)
    plt.imshow(grid.detach().cpu().permute(1, 2, 0))
    plt.title(f"Generated Images for Prompt: '{prompt}'")
    plt.axis('off')

    plt.figure(figsize=(8, 8))
    plt.title("Sampled Images from the Diffusion Model")
    for i in range(16):
        ax, fig = plt.subplot(4,4,i+1), plt.gcf()
        index = torch.randint(0, len(dataset), (1,)).item()
        img, label = dataset[index]
        img = (img + 1) / 2 # Scale back to [0, 1] for visualization
        img = img.permute(1, 2, 0) 
        ax.imshow(img.numpy(), interpolation='bicubic')
        ax.axis('off')

    plt.show()

    torch.save(model.state_dict(), "models/diffusion_model.pth")


