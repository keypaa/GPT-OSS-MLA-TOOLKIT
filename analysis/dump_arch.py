import torch
from transformers import AutoModelForCausalLM
import os

def dump_architecture():
    model_id = "openai/gpt-oss-20b"
    # Save into the logs folder
    output_file = "analysis/logs/architecture.txt"
    
    # Ensure folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Loading config for {model_id}...")
    
    # Load on Meta device (Instant, 0 memory usage)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="meta", 
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Writing architecture to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"--- FULL ARCHITECTURE FOR {model_id} ---\n\n")
        f.write(str(model))
        f.write("\n\n--- DETAILED MODULE LIST ---\n")
        
        for name, module in model.named_modules():
            # Write the name and the module type/details
            f.write(f"[{name}]: {module}\n")
            
            # If it's a Linear layer, print its shape specifically
            if isinstance(module, torch.nn.Linear):
                f.write(f"    -> Shape: {module.in_features} x {module.out_features}\n")
            f.write("-" * 40 + "\n")

    print(f"Done. Saved to {output_file}")

if __name__ == "__main__":
    dump_architecture()