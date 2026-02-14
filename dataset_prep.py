import pandas as pd
from datasets import load_dataset, Dataset, DownloadConfig
from transformers import AutoTokenizer
import os

MIN_TOKENS = 30
MAX_TOKENS = 1024
MIN_WORDS = 20
OUTPUT_FOLDER = r"D:\custom llm from scratch\cleaned_tinystories"
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def filter_story(example):
    text = example['text']
    tokens = len(tokenizer.encode(text, add_special_tokens=False))
    words = len(text.split())
    return tokens >= MIN_TOKENS and tokens <= MAX_TOKENS and words >= MIN_WORDS

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("Loading TinyStories dataset...")
    full_dataset = load_dataset("roneneldan/TinyStories", split="train")
    print(f"Original count: {len(full_dataset):,}")

    print("Computing approximate token lengths for filtering...")
    clean_dataset = full_dataset.filter(filter_story, num_proc=4)  # multiprocessing now safe

    print(f"After filtering: {len(clean_dataset):,} stories kept")

    train_n = int(0.96 * len(clean_dataset))
    train_ds = clean_dataset.select(range(train_n))
    val_ds = clean_dataset.select(range(train_n, len(clean_dataset)))

    train_ds.save_to_disk(f"{OUTPUT_FOLDER}/train")
    val_ds.save_to_disk(f"{OUTPUT_FOLDER}/val")
    print(f"Saved to {OUTPUT_FOLDER}")