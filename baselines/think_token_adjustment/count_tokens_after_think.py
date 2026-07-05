import json
from tqdm import tqdm
import sys
import os

MODEL_NAME = sys.argv[1]
DATASET_PATH = sys.argv[2]
OUTPUT_PATH = sys.argv[3]

from transformers import AutoTokenizer

print(f"Loading tokenizer for: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Tokenizer loaded.")

# Load data and count tokens after </think>
print("Processing dataset...")
results = []

with open(DATASET_PATH, "r") as f:
    lines = f.readlines()
    for line in tqdm(lines, desc="Counting tokens after </think>"):
        item = json.loads(line)
        generated_text = item["generated_text"]
        
        # Find everything after </think>
        if "</think>" in generated_text:
            after_think = generated_text.split("</think>", 1)[1]
            tokens = tokenizer.encode(after_think, add_special_tokens=False)
            num_tokens = len(tokens)
        else:
            tokens = tokenizer.encode(generated_text, add_special_tokens=False)
            # If no </think> and generated_text < 2048 then count entire text
            if len(tokens) < 30000:
                after_think = generated_text
                num_tokens = len(tokens)
            else: # If no </think> and generated_text >= 2048 then the reasoning didn't end ==> no answer
                after_think = ""
                num_tokens = 0
            
        
        results.append({
            "question": item["question"],
            "answer": item.get("answer", ""),
            "tokens_after_think": num_tokens,
            "text_after_think": after_think
        })

# Save results
print(f"Saving results to {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "w") as f_out:
    for result in results:
        json.dump(result, f_out, ensure_ascii=False)
        f_out.write("\n")

print(f"Done! Processed {len(results)} questions.")