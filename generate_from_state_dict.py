+#!/usr/bin/env python3
# generate_from_state_dict.py
# Usage:
#   python generate_from_state_dict.py --ckpt /path/to/pytorch_model.bin --prompt "Hello" --device cuda --max_len 100

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model_moe_small import MoETransformer

def sample_logits(logits, top_k=None, top_p=None, temperature=1.0):
    # logits: (V,)
    logits = logits / max(1e-8, temperature)
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, top_k)
        minv = v[-1]
        logits = torch.where(logits < minv, torch.tensor(-1e10, device=logits.device), logits)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        # mask tokens above top_p
        cutoff = cumulative > top_p
        # keep first token above cutoff
        cutoff_idx = torch.argmax(cutoff.int()).item()
        mask = torch.ones_like(sorted_logits, dtype=torch.bool)
        mask[cutoff_idx+1:] = False
        # map mask back to original order
        mask_orig = torch.zeros_like(mask)
        mask_orig[sorted_idx] = mask
        logits = torch.where(mask_orig, logits, torch.tensor(-1e10, device=logits.device))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

def generate(model, tokenizer, prompt, device, max_len=128, temperature=1.0, top_k=50, top_p=0.95):
    model.eval()
    toks = tokenizer.encode(prompt, add_special_tokens=False)
    if hasattr(toks, 'ids'): toks = toks.ids
    input_ids = torch.tensor([toks], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_ids)  # [B, T, V]
            next_logits = logits[0, -1]  # [V]
            next_id = sample_logits(next_logits, top_k=top_k, top_p=top_p, temperature=temperature)
            next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            # optional: stop on eos
            if next_id == tokenizer.eos_token_id:
                break
    out_ids = input_ids[0].tolist()
    text = tokenizer.decode(out_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="path to pytorch_model.bin (state_dict)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=11)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1536)
    parser.add_argument("--n_experts", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    print("Loading tokenizer (gpt2)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    print("Building model architecture...")
    model = MoETransformer(vocab_size=args.vocab_size, dim=args.dim,
                           n_layers=args.n_layers, n_heads=args.n_heads,
                           ff_dim=args.ff_dim, n_experts=args.n_experts)
    # load state dict
    print("Loading checkpoint:", args.ckpt)
    sd = torch.load(args.ckpt, map_location="cpu")
    # if sd is a dict with 'state_dict' key, extract it
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    # try to be permissive on key names (remove leading module. if present)
    new_sd = {}
    for k,v in sd.items():
        newk = k
        if k.startswith("module."):
            newk = k[len("module."):]
        new_sd[newk] = v
    model.load_state_dict(new_sd, strict=False)
    model.to(device)
    print("Running generation...")
    out = generate(model, tokenizer, args.prompt, device, max_len=args.max_len,
                   temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    print("\n=== GENERATED ===\n")
    print(out)
    print("\n=== END ===\n")

if __name__ == "__main__":
    main()

