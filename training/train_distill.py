import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
import json
import argparse
from tqdm import tqdm

# --- CONFIG ---
ACCUMULATION_STEPS = 4 
LEARNING_RATE = 1e-5
MAX_LENGTH = 2048 

class JsonlDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Loading {file_path}...")
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line)['text'])

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.data[idx], truncation=True, max_length=self.max_length, 
                             padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0)}

def train_distillation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--student", type=str, required=True, help="Path to MLA converted model")
    parser.add_argument("--data", type=str, default="data_pipeline/calib_mixed_golden.jsonl")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=ACCUMULATION_STEPS)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Load Teacher (4-bit Frozen)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, quantization_config=bnb, device_map={"": accelerator.device}, trust_remote_code=True)
    teacher.eval()

    # Load Student (BF16 Trainable)
    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16, device_map={"": accelerator.device}, trust_remote_code=True, use_cache=False)
    student.gradient_checkpointing_enable()
    student.train()

    # Optimizer (Only train MLA Projectors)
    params = [p for n, p in student.named_parameters() if "proj" in n or "norm" in n]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)

    dataset = JsonlDataset(args.data, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    student, optimizer, dataloader = accelerator.prepare(student, optimizer, dataloader)

    print("Starting Distillation...")
    for step, batch in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(student):
            with torch.no_grad(): t_logits = teacher(**batch).logits
            s_logits = student(**batch).logits
            
            # KL Loss (Temp=2.0)
            loss = F.kl_div(F.log_softmax(s_logits/2.0, dim=-1), F.softmax(t_logits/2.0, dim=-1), reduction='batchmean') * 4.0
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
        if step % 100 == 0: print(f"Step {step} Loss: {loss.item():.4f}")

    student.save_pretrained("gpt-oss-20b-mla-healed")

if __name__ == "__main__":
    train_distillation()