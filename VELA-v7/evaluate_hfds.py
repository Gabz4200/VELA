import os

os.environ["RWKV_JIT_ON"] = "1"

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoImageProcessor

from Vela7.src.dataset import (
    DEFAULT_STOP_TOKEN,
    STOP_TOKEN_INDEX,
    preprocess,
    process_image_tokens_in_conversations,
)
from Vela7.src.utils import Conversation
from Vela7.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER


def get_input_image_tensor(image_list, image_processor):
    crop_size = image_processor.size
    from Vela7.src.utils import image_to_regions

    all_regions = []
    for image in image_list:
        image = image.convert("RGB")
        regions = image_to_regions(image, (crop_size["width"], crop_size["height"]))
        all_regions.extend(regions)
    if len(all_regions) == 0:
        return None, 0
    images = image_processor.preprocess(all_regions, return_tensors="pt")["pixel_values"]
    return images.unsqueeze(0), len(all_regions)


def prepare_conversations(line):
    if "question" in line:
        input_text = line["question"]

        conv = Conversation(id=line["sample_id"], roles=["human", "gpt"], conversations=[])
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], "")

        conversations = conv.conversations
    elif "conversations" in line:
        conv = Conversation(id=line["sample_id"], roles=["human", "gpt"], conversations=[])
        for msg in line["conversations"]:
            if msg["from"] == "human":
                conv.append_message(conv.roles[0], msg["value"])
            elif msg["from"] == "gpt":
                conv.append_message(conv.roles[1], msg["value"])
            else:
                raise ValueError(f"Unknown role {msg['from']}")
        # if last message is from human, add an empty message from gpt
        if msg["from"] == "human":
            conv.append_message(conv.roles[1], "")
        conversations = conv.conversations
    else:
        raise ValueError("Invalid input line, no 'question' or 'conversations' field")
    return conversations


def eval_model(args):
    from Vela7.src.model import VELA

    model_path = Path(args.model_path)
    model_name = model_path.stem
    if getattr(args, "vision_tower_path", None) is None:
        if getattr(args, "vision_tower_dir", None) is not None:
            args.vision_tower_path = str(
                Path(args.vision_tower_dir) / "timm/ViT-SO400M-14-SigLIP-384"
            )
        else:
            args.vision_tower_path = "google/siglip-so400m-patch14-384"
    # Model
    model = VELA(args)
    msg = model.load_state_dict(torch.load(model_path), strict=False)
    print("msg of loading model: ", msg)
    model = model.bfloat16().to(args.device)
    tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")
    image_processor = AutoImageProcessor.from_pretrained(args.vision_tower_path, use_fast=True)

    dataset = load_from_disk(args.dataset_path)
    if isinstance(dataset, dict):
        dataset = dataset[args.split]
    # output to the same dir of the model
    dataset_path = Path(args.dataset_path)
    output_dir = model_path.parent / dataset_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}.jsonl"

    out_file = open(output_file, "w")
    pbar = tqdm(total=len(dataset))
    update_every = len(dataset) // 100 if len(dataset) >= 100 else 1
    for i, line in enumerate(dataset):
        idx = line["sample_id"]
        image_keys = [k for k in line.keys() if "image" in k]
        image_list = [line[k] for k in image_keys if line[k] is not None]
        images, num_regions = get_input_image_tensor(image_list, image_processor)
        if images is not None:
            images = images.bfloat16().to(args.device)

        conversations = prepare_conversations(line)
        if images is not None:
            conversations = process_image_tokens_in_conversations(
                conversations, num_regions=num_regions
            )

        data_dict = preprocess(
            conversations,
            tokenizer,
            has_image=(images is not None),
            ctx_len=args.ctx_len,
            num_token_per_image=args.num_token_per_image,
            pad_token_id=0,
            do_pad_to_max_length=False,
        )

        input_ids = data_dict["input_ids"].unsqueeze(0).to(args.device)
        cur_prompt = data_dict["input_text"]
        if i == 0:
            print("input_ids.shape: ", input_ids.shape)
            print("input_ids: ", input_ids)
            print("cur_prompt: ", cur_prompt)
            print("num_input_images: ", len(image_list))
            if images is not None:
                print(f"images.shape: {images.shape}")

        with torch.inference_mode():
            output_ids, output_logits, output_probs = model.generate(
                input_ids,
                images=images,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=args.max_new_tokens,
                stop_token_idx=STOP_TOKEN_INDEX,
            )

        output = tokenizer.decode(output_ids).split(DEFAULT_STOP_TOKEN)[0].strip()
        # avg logit
        avg_logit = sum(output_logits) / len(output_logits)
        # geometric mean of probs
        avg_prob = np.prod(output_probs) ** (1.0 / len(output_probs))

        out_str = json.dumps(
            {
                "question_id": idx,
                "prompt": cur_prompt,
                "text": output,
                "avg_logit": str(round(avg_logit, 3)),
                "avg_prob": str(round(avg_prob, 3)),
                "model_id": model_name,
                "metadata": {
                    "sub_task": line.get("sub_task", None),
                    "question_type": line.get("question_type", None),
                    "answer": line.get("answer", None),
                },
            },
            ensure_ascii=False,
        )
        out_file.write(out_str + "\n")
        # update progress bar
        if i % update_every == 0 and i != 0:
            pbar.update(update_every)
        out_file.flush()
    out_file.close()
    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # arguments to init model
    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--vocab_size", default=65536, type=int)
    parser.add_argument("--ctx_len", default=256, type=int)
    parser.add_argument("--n_layer", default=24, type=int)
    parser.add_argument("--n_embd", default=2048, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument(
        "--pre_ffn", default=0, type=int
    )  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_size_a", default=64, type=int)
    parser.add_argument("--head_size_divisor", default=8, type=int)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument(
        "--vision_tower_dir",
        type=str,
        help="Path to the directory containing the vision tower checkpoints",
    )
    parser.add_argument("--vision_tower_path", type=str, default=None)
    parser.add_argument(
        "--grad_cp", default=0, type=int
    )  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--proj_type", default="linear", type=str, choices=["linear", "mlp"])
    parser.add_argument("--num_token_per_image", type=int, default=16)
    # arguments for evaluation
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    #
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)  # default = 3.5x emb size

    eval_model(args)
