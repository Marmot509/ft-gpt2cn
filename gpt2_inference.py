
import argparse
from transformers import BertTokenizer, GPT2LMHeadModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="uer/gpt2-distil-chinese-cluecorpussmall", help="model weights")
parser.add_argument("--tokenizer", type=str, default=None, help="model tokenizer")

args = parser.parse_args()

if args.tokenizer is None:
    args.tokenizer = args.model

tokenizer = BertTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
model = GPT2LMHeadModel.from_pretrained(args.model, trust_remote_code=True)

while True:
    prompt = input("Prompt:")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to("cuda")
    response = model.generate(input_ids=inputs["input_ids"], max_length=inputs["input_ids"].shape[-1] + args.max_new_tokens)
    response = response[0, inputs["input_ids"].shape[-1]:]
    print("Response:", tokenizer.decode(response, skip_special_tokens=True))