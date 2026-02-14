from datasets import load_from_disk

train_full = load_from_disk(r"D:\custom llm from scratch\cleaned_tinystories\train")
val_full   = load_from_disk(r"D:\custom llm from scratch\cleaned_tinystories\val")

# Subsample
train_small = train_full.shuffle(seed=42).select(range(150000))
val_small   = val_full.shuffle(seed=42).select(range(20000))

train_small.save_to_disk(r"D:\custom llm from scratch\filtered_tinystories\train")
val_small.save_to_disk(r"D:\custom llm from scratch\filtered_tinystories\val")