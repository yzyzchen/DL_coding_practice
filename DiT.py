import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class adaLNzero(nn.Module):
    def __init__(self, dim, cond_dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.linear = nn.Linear(cond_dim, dim * 3)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x, cond):
        B, T, C = x.shape
        assert C == self.dim
        assert cond.shape[0] == B
        assert cond.shape[1] == self.cond_dim
        params = self.linear(cond)
        gamma, beta, gate = params.chunk(3, dim = -1)
        gamma, beta, gate = gamma[:, None, :], beta[:, None, :], gate[:, None, :]
        x = self.norm(x)
        x = (1 + gamma) * x + beta
        return x, gate
    
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        out = self.linear1(x)
        swish = out * nn.Sigmoid()(out)
        swiglu = swish * self.linear2(x)
        return self.proj_out(swiglu)
    
class MHA(nn.Module):
    def __init__(self, dim, nheads, cross = False):
        super().__init__()
        self.dim = dim
        self.nheads = nheads
        if cross:
            self.q_proj = nn.Linear(dim, dim)
            self.kv_proj = nn.Linear(dim, 2 * dim)
        else:
            self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def selfAttn(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim = -1) #(B, T, C)
        assert self.dim % self.nheads == 0
        d_heads = self.dim // self.nheads
        q = q.view(B, T, self.nheads, d_heads).transpose(1, 2)
        k = k.view(B, T, self.nheads, d_heads).transpose(1, 2)
        v = v.view(B, T, self.nheads, d_heads).transpose(1, 2)
        post_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(post_attn.transpose(1, 2).contiguous().view(B, T, C))
    
    def crossAttn(self, x_q, x_kv):
        B, T_q, C = x_q.shape
        _, T_kv, _ = x_kv.shape
        assert self.dim % self.nheads == 0
        d_heads = self.dim // self.nheads
        q = self.q_proj(x_q)
        k, v = self.kv_proj(x_kv).chunk(2, dim = -1)
        q = q.view(B, T_q, self.nheads, d_heads).transpose(1, 2)
        k = k.view(B, T_kv, self.nheads, d_heads).transpose(1, 2)
        v = v.view(B, T_kv, self.nheads, d_heads).transpose(1, 2)
        post_attn = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(post_attn.transpose(1, 2).contiguous().view(B, T_kv, C))
    
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.act = SwiGLU(dim, hidden_dim)
    
    def forward(self, x):
        return self.act(x)
    
class DiT_block(nn.Module):
    def __init__(self, dim, cond_dim, hidden_dim, nheads):
        super().__init__()
        self.mlp = MLP(dim, hidden_dim)
        self.adaLN1 = adaLNzero(dim, cond_dim)
        self.adaLN2 = adaLNzero(dim, cond_dim)
        self.attn = MHA(dim, nheads)
    
    def forward(self, x, cond):
        normed, gate = self.adaLN1(x, cond)
        post_attn = x + gate * self.attn.selfAttn(normed)
        normed, gate = self.adaLN2(post_attn, cond)
        out = post_attn + gate * self.mlp(normed)
        return out
    
class MDT_block(nn.Module):
    def __init__(self, dim, cond_dim, hidden_dim, nheads):
        super().__init__()
        self.mlp = MLP(dim, hidden_dim)
        self.adaLN1 = adaLNzero(dim, cond_dim)
        self.adaLN2 = adaLNzero(dim, cond_dim)
        self.attn = MHA(dim, nheads)
        self.cross_attn = MHA(dim, nheads, cross=True)
    
    def forward(self, x, image_emb, cond):
        normed, gate = self.adaLN1(x, cond)
        post_self_attn = x + gate * self.attn.selfAttn(normed)

        post_attn = post_self_attn + self.cross_attn.crossAttn(x, image_emb)
        
        normed, gate = self.adaLN2(post_attn, cond)
        out = post_attn + gate * self.mlp(normed)
        return out
        