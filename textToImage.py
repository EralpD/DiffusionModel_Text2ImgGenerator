"""
We will use U-Net structure to build the ControlNet model for text-to-image generation.
"""

from transformers import CLIPTokenizer, CLIPTextModel
import torch
import torch.nn as nn
import torch.nn.functional as F

CLIPmodel = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=768, n_heads=8, d_head=64):
        super().__init__()
        inner_dim = n_heads * d_head
        self.n_heads = n_heads
        self.scale = d_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.1)
        )
        self.norm = nn.GroupNorm(32, query_dim)

    def forward(self, x, context):
        res = x
        x = self.norm(x)

        b,c,h,w = x.shape
        x = x.view(b, c, h*w).permute(0, 2, 1)
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.view(b, -1, self.n_heads, q.shape[-1] // self.n_heads).permute(0, 2, 1, 3)
        k = k.view(b, -1, self.n_heads, k.shape[-1] // self.n_heads).permute(0, 2, 1, 3)
        v = v.view(b, -1, self.n_heads, v.shape[-1] // self.n_heads).permute(0, 2, 1, 3)

        sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, -1, self.n_heads * (q.shape[-1]))
        out = self.to_out(out)
        return out.permute(0, 2, 1).view(b, c, h, w) + res
        