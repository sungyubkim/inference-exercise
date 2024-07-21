import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
parser.add_argument("--prompt", type=str, default="Can you explain Tensor Parallelism?")
args = parser.parse_args()
dtype = torch.bfloat16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    trust_remote_code=True,
    )
tokenizer = AutoTokenizer.from_pretrained(args.model)

model.to(device=device, dtype=dtype)
model.eval()

messages = [{"role": "user", "content": f"{args.prompt}"}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device=device)

outputs = model.generate(inputs, max_new_tokens=1024)
text = tokenizer.batch_decode(outputs)[0]
print(text)