# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

device = "cuda" if torch.cuda.is_avlable() else "cpu"

class SwiGLU(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, ff_dim * 2, bias=True)
        self.w2 = nn.Linear(ff_dim, dim, bias=True)

    def forward(self, x):
       
        u = self.w1(x)  
        a, b = u.chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)

class Expert(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.ffn = SwiGLU(dim, ff_dim)

    def forward(self, x):
        return self.ffn(x)

class Top2Router(nn.Module):
    def __init__(self, dim, n_experts, k=2):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.linear = nn.Linear(dim, n_experts, bias=False)

    def forward(self, x):
     
        logits = self.linear(x) 
        topk = torch.topk(logits, self.k, dim=-1)  
        topk_vals = topk.values 
        topk_idx = topk.indices  
        gates = F.softmax(topk_vals, dim=-1)  
        return topk_idx, gates, logits

class MoEFFN(nn.Module):
    def __init__(self, dim, ff_dim, n_experts=4, k=2, capacity_factor=1.25):
        super().__init__()
        self.dim = dim
        self.ff_dim = ff_dim
        self.n_experts = n_experts
        self.k = k
        self.router = Top2Router(dim, n_experts, k)
        self.experts = nn.ModuleList([Expert(dim, ff_dim) for _ in range(n_experts)])
        self.capacity_factor = capacity_factor

    def forward(self, x):
     
        B, T, D = x.shape
        device = x.device

        topk_idx, gates, logits = self.router(x)  
        flat_x = x.view(B * T, D)  
        flat_idx = topk_idx.view(B * T, self.k) 
        flat_gates = gates.view(B * T, self.k)  

        
        outputs = torch.zeros_like(flat_x, device=device)  
        
        for choice in range(self.k):
           
            expert_ids = flat_idx[:, choice]  
            gate_vals = flat_gates[:, choice]  
            for e in range(self.n_experts):
                mask = (expert_ids == e)
                if not mask.any():
                    continue
                idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)  # indices into flat_x
                inputs = flat_x[idxs]  # [Ne, D]
                out = self.experts[e](inputs)  # [Ne, D]
                g = gate_vals[idxs].unsqueeze(-1)  # [Ne, 1]
                outputs[idxs] += out * g  # weighted sum
        # reshape back
        y = outputs.view(B, T, D)
        # extremely small auxiliary load-balance etc. omitted here (can be added)
        return y

# ---- Attention + MoE transformer block ----
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, attn_dropout=0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.register_buffer("mask", None, persistent=False)

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x)  
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.n_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)
       
        att = torch.einsum("b h i d, b h j d -> b h i j", q, k) / (self.head_dim ** 0.5)
        # causal mask
        i, j = att.shape[-2], att.shape[-1]
        if self.mask is None or self.mask.size(0) < i:
            mask = torch.tril(torch.ones((i, j), device=x.device)).unsqueeze(0).unsqueeze(0)
            self.mask = mask
        att = att.masked_fill(self.mask[:, :, :i, :j] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", att, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ff_dim, n_experts=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.moe = MoEFFN(dim, ff_dim, n_experts=n_experts, k=2)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x

# ---- Full model ----
class MoETransformer(nn.Module):
    def __init__(self, vocab_size=50257, dim=384, n_layers=11, n_heads=8,
                 ff_dim=1536, n_experts=4, context_len=1024):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(context_len, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads, ff_dim, n_experts=n_experts)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        # input_ids: [B, T]
        B, T = input_ids.shape
        device = input_ids.device
        tok = self.token_emb(input_ids)  # [B, T, D]
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos = self.pos_emb(pos_ids)
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, V]
        return logits





