# prepare_science_data.py
from datasets import load_dataset

print("Loading MetaMathQA dataset...")
# This dataset is real and highly effective for reasoning calibration
dataset = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)

print("Formatting data...")
with open("science_calibration.txt", "w", encoding="utf-8") as f:
    count = 0
    for row in dataset:
        if count > 20000: break # 20000 examples is still plenty
        
        # MetaMathQA stores the Question in 'query' and the Chain-of-Thought in 'response'
        # We format it to look like a standard instruction interaction.
        text = f"User: {row['query']}\nAssistant: {row['response']}\n\n"
        
        f.write(text)
        count += 1
        if count % 100 == 0: print(f"Processed {count}...")

print("Done! science_calibration.txt created.")
