import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import time
import math
from datasets import load_dataset, load_from_disk
import glob
import random
import numpy as np
import os
import pandas as pd
import ast

def set_seed(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  print(f"--- Seed set to {seed} ---")

class Dataset(Dataset):
  def __init__(self, tokenizer, block_size=512, split_path=r"D:\custom llm from scratch"):
    self.tokenizer = tokenizer
    self.block_size = block_size
    self.data = []

    print(f"Loading from disk: {split_path}")
    hf_ds = load_from_disk(split_path)

    print("Tokenizing and packing ...")
    count = 0

    for sample in hf_ds:
      text = sample["text"]

      try:
        tokens = tokenizer.encode(text, truncation=True, max_length=self.block_size)
        if len(tokens) > 5:
          self.data.extend(tokens)
          self.data.append(tokenizer.eos_token_id)
          count += 1
      except:
        continue

      if count % 2000 == 0:
        print(f" Packed {count} conversations...")

    print(f"Packed dataset: {len(self.data):,} tokens total")

  def __len__(self):
    # Returns the number of available blocks
    return len(self.data) // self.block_size - 1

  def __getitem__(self, idx):
    start = idx * self.block_size
    end = start + self.block_size + 1
    chunk = self.data[start:end]

    # Handle edge cases if data isnt perfectly divisible
    if len(chunk) < self.block_size + 1:
      padding = [self.tokenizer.pad_token_id] * (self.block_size + 1 - len(chunk))
      chunk = chunk + padding

    x = torch.tensor(chunk[:-1], dtype=torch.long)
    y = torch.tensor(chunk[1:], dtype=torch.long)

    return x, y

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.query = nn.Linear(n_embed, n_embed)
        self.key = nn.Linear(n_embed, n_embed)
        self.value = nn.Linear(n_embed, n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        att_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0 if not self.training else 0.1,
            is_causal=True,
        )
        attn_output = att_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(attn_output)

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        self.attn = MultiHeadAttention(n_embed, n_head, block_size)
        self.ff = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class SmallLLM(nn.Module):
    def __init__(self, vocab_size, n_embed, n_head, n_layer, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)

        # Create sinusodial position encoding
        self.register_buffer('pos_encoding', self.create_sinusodial_encoding(block_size, n_embed))

        self.blocks = nn.Sequential(*[TransformerBlock(n_embed, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.block_size = block_size

    def create_sinusodial_encoding(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1)
        div_terms = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_terms)
        pos_encoding[:, 1::2] = torch.cos(position * div_terms)

        return pos_encoding

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embed = self.token_embedding(idx)

        pos_embed = self.pos_encoding[:T] # Shape: (T, n_embed)
        pos_embed = pos_embed.unsqueeze(0) # Shape: (1, T, n_embed) for broadcasting

        x = tok_embed + pos_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        logits = logits.view(B * T, self.lm_head.out_features)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def evaluate_model(model, val_loader, device, num_batches=None):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if num_batches is not None and i >= num_batches:
                break

            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item() * x.size(0)

    if num_batches is None:
        avg_loss = total_loss / len(val_loader.dataset)
    else:
        avg_loss = total_loss / (num_batches * val_loader.batch_size)

    return avg_loss

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    train_dataset = Dataset(
    tokenizer=tokenizer,
    split_path=r"D:\custom llm from scratch\filtered_tinystories\train",
    block_size=512
    )

    val_dataset = Dataset(
    tokenizer=tokenizer,
    split_path=r"D:\custom llm from scratch\filtered_tinystories\val",
    block_size=512
    )

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    block_size = 512
    n_embed = 640
    n_head = 10
    n_layer = 12
    learning_rate = 3e-4
    max_epochs = 10
    eval_interval = 700
    grad_clip = 1.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    model = SmallLLM(vocab_size, n_embed, n_head, n_layer, block_size).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    print("\nStarting training...")
    start_time = time.time()
    step = 0
    best_val_loss = float('inf')

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            step += 1

            if step % 100 == 0:
                print(f"Step {step}: loss {loss.item():.4f}")

            if step % eval_interval == 0:
                val_loss = evaluate_model(model, val_loader, device, num_batches=50)
                print(f" Step {step}: train {loss.item():.4f}, val {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), r'D:\custom llm from scratch\small_llm_best.pth')
                    print(f"  New best model saved with val loss: {val_loss:.4f}")

                if early_stopping(val_loss):
                    print('Early stopping triggered')
                    break

                model.train()

        avg_epoch_loss = epoch_loss / epoch_steps
        print(f"Epoch {epoch+1}/{max_epochs} completed - Avg loss: {avg_epoch_loss:.4f}")

        if early_stopping.early_stop:
            break

    if os.path.exists(r'D:\custom llm from scratch\small_llm_best.pth'):
        model.load_state_dict(torch.load(r'D:\custom llm from scratch\small_llm_best.pth'))
        print("Loaded best model for generation")

    final_val_loss = evaluate_model(model, val_loader, device)
    print(f"\nFinal validation loss: {final_val_loss:.4f}")

    torch.save(model.state_dict(), r'D:\custom llm from scratch\small_llm_final.pth')
    print(f"\nModel saved to small_llm_final.pth")

    print("\nGenerating sample text...")
    model.eval()
    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=100, temperature=0.8)

    print(tokenizer.decode(generated[0].tolist(), skip_special_tokens=True))

    end_time = time.time()
    print(f"\nTraining completed in {(end_time - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()