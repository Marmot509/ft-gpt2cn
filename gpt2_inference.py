
import argparse
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="uer/gpt2-distil-chinese-cluecorpussmall", help="model weights")
parser.add_argument("--tokenizer", type=str, default=None, help="model tokenizer")
parser.add_argument("--max_new_tokens", type=str, default=500, help="max new tokens to generate")

args = parser.parse_args()

if args.tokenizer is None:
    args.tokenizer = args.model

tokenizer = BertTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
model = GPT2LMHeadModel.from_pretrained(args.model, trust_remote_code=True)

while True:
    prompt = input("Prompt:")
    text_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    text_generator(prompt, max_length=args.max_new_tokens, do_sample=True, temperature=0.9)