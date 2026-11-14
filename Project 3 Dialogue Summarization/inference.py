"""Summarize a chat with the fine-tuned BERT2GPT2 SAMSum model.

Usage:
    python inference.py --model models/bert2gpt2-samsum-final --file chat.txt
    python inference.py --model your-username/bert2gpt2-samsum --file chat.txt
    cat chat.txt | python inference.py --model models/bert2gpt2-samsum-final

The chat should contain one message per line in "Name: message" format.
"""
import argparse
import os
import re
import sys

import torch
from transformers import AutoTokenizer, EncoderDecoderModel

MAX_INPUT_LEN = 512


def clean_text(t):
    """Same normalization as the training notebook."""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"<file_(\w+)>", r"[\1]", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n+", "\n", t)
    return t.strip()


def load(model_path, device):
    """Load the model and its two tokenizers from a local folder or a Hub repo."""
    if os.path.isdir(model_path):
        enc_tok = AutoTokenizer.from_pretrained(os.path.join(model_path, "encoder_tokenizer"))
        dec_tok = AutoTokenizer.from_pretrained(os.path.join(model_path, "decoder_tokenizer"))
    else:
        enc_tok = AutoTokenizer.from_pretrained(model_path, subfolder="encoder_tokenizer")
        dec_tok = AutoTokenizer.from_pretrained(model_path, subfolder="decoder_tokenizer")
    model = EncoderDecoderModel.from_pretrained(model_path).to(device).eval()
    return model, enc_tok, dec_tok


@torch.no_grad()
def summarize(dialogues, model, enc_tok, dec_tok, num_beams=4, length_penalty=1.0, max_new_tokens=128):
    enc = enc_tok([clean_text(d) for d in dialogues], max_length=MAX_INPUT_LEN, truncation=True,
                  padding=True, return_token_type_ids=False, return_tensors="pt").to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, num_beams=num_beams)
    if num_beams > 1:
        kwargs.update(no_repeat_ngram_size=3, early_stopping=True, length_penalty=length_penalty)
    ids = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], **kwargs)
    return [s.strip() for s in dec_tok.batch_decode(ids, skip_special_tokens=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="local model folder or Hugging Face Hub repo id")
    parser.add_argument("--file", help="text file with the chat (default: read stdin)")
    parser.add_argument("--beams", type=int, default=4, help="beam size; 1 = greedy decoding")
    parser.add_argument("--length-penalty", type=float, default=1.0, help="beam search length penalty")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            chat = f.read()
    else:
        chat = sys.stdin.read()
    if not chat.strip():
        sys.exit("No chat text provided.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, enc_tok, dec_tok = load(args.model, device)
    print(summarize([chat], model, enc_tok, dec_tok, args.beams, args.length_penalty)[0])


if __name__ == "__main__":
    main()
