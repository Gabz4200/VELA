import os
os.environ["RWKV_JIT_ON"] = "1"

import json
from PIL import Image
import pandas as pd
import numpy as np
import math
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from Vela7.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from Vela7.src.dataset import DEFAULT_IMAGE_TOKEN, DEFAULT_STOP_TOKEN, STOP_TOKEN_INDEX
from Vela7.src.dataset import process_image_tokens_in_conversations, preprocess
from Vela7.src.utils import Conversation
from transformers import AutoImageProcessor


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False


def load_questions(file_path):
    file_path = Path(file_path)
    suffix = file_path.suffix
    if suffix == ".jsonl":
        questions = [json.loads(q) for q in open(file_path)]
    elif suffix == ".json":
        questions = json.load(open(file_path))
    elif suffix == ".tsv":
        questions = pd.read_table(file_path).to_dict("records")
    else:
        raise ValueError("Unsupported file type: {}".format(suffix))
    return questions


def get_question_id(line):
    if "question_id" in line:
        return line["question_id"]
    elif "id" in line:
        return line["id"]
    elif "index" in line:
        return line["index"]
    else:
        raise ValueError("Cannot find question id in line: {}".format(line))


def get_input_text(line, num_images, dataset_name):
    input_text = line["text"] if "text" in line else line["conversations"][0]["value"]
    # remove DEFAULT_IMAGE_TOKEN
    input_text = input_text.replace(DEFAULT_IMAGE_TOKEN, "").strip()
    # add <image> tokens
    image_prifix = "\n".join(num_images * [DEFAULT_IMAGE_TOKEN])
    input_text = image_prifix + "\n" + input_text
    if 'conversations' in line and dataset_name == 'scienceqa': # special case for scienceqa
        input_text += "\n" + "Answer with the option's letter from the given choices directly."
    return input_text


def get_single_image_tensor(line, image_folder, image_processor):
    if "image" in line:
        image = Image.open(image_folder / line["image"]).convert("RGB")
        crop_size = image_processor.size
        from Vela7.src.utils import image_to_regions
        regions = image_to_regions(image, (crop_size['width'], crop_size['height']))
        images = image_processor.preprocess(regions, return_tensors='pt')['pixel_values']
        return images.unsqueeze(0), len(regions)
    else:
        raise ValueError("no key 'image' in line: {}".format(line))
    




def get_video_image_tensor(line, image_folder, image_processor):
    if "video" in line:
        video_folder = image_folder / line["video"]
        video_frames = sorted(video_folder.rglob("*.jpg"))
        num_frames = line['text'].count(DEFAULT_IMAGE_TOKEN)
        # uniform sampling
        if len(video_frames) <= num_frames:
            sampled_frames = video_frames
        else:
            indices = np.linspace(0, len(video_frames) - 1, num_frames)
            rounded_indices = np.round(indices).astype(int)
            sampled_frames = [video_frames[i] for i in rounded_indices]
        
        all_regions = []
        crop_size = image_processor.size
        from Vela7.src.utils import image_to_regions
        for frame in sampled_frames:
            image = Image.open(frame).convert("RGB")
            regions = image_to_regions(image, (crop_size['width'], crop_size['height']))
            all_regions.extend(regions)
        if len(all_regions) == 0:
            return None, 0
        images = image_processor.preprocess(all_regions, return_tensors='pt')['pixel_values']
        return images.unsqueeze(0), len(all_regions)
    else:
        raise ValueError("no key 'video' in line: {}".format(line))


def eval_model(args):
    from Vela7.src.model import VELA
    model_path = Path(args.model_path)
    model_name = model_path.parent.name
    if getattr(args, 'vision_tower_path', None) is None:
        if getattr(args, 'vision_tower_dir', None) is not None:
            args.vision_tower_path = str(Path(args.vision_tower_dir) / "timm/ViT-SO400M-14-SigLIP-384")
        else:
            args.vision_tower_path = "google/siglip-so400m-patch14-384"
    # Model
    model = VELA(args)
    msg = model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    print("msg of loading model: ", msg)
    model = model.bfloat16().to(args.device)
    tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")
    image_processor = AutoImageProcessor.from_pretrained(args.vision_tower_path, use_fast=True)

    questions = load_questions(args.question_file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image_folder = Path(args.image_folder) if args.image_folder is not None else None

    out_file = open(output_file, "w")
    pbar = tqdm(total=len(questions))
    update_every = len(questions) // 100 if len(questions) >= 100 else 1
    for i, line in enumerate(questions):
        # skip if no image or video line
        if 'image' not in line and 'video' not in line:
            continue
        idx = get_question_id(line)
        #
        if 'image' in line:
            images, num_regions = get_single_image_tensor(line, image_folder, image_processor)
        else:
            images, num_regions = get_video_image_tensor(line, image_folder, image_processor)
        
        images = images.bfloat16().to(args.device)
        
        input_text = get_input_text(line, num_images=num_regions, dataset_name=args.dataset_name)

        conv = Conversation(id=idx, roles=["human", "gpt"], conversations=[])
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], "")

        conversations = process_image_tokens_in_conversations(
            conv.conversations,
            num_regions=num_regions)

        data_dict = preprocess(
            conversations,
            tokenizer,
            has_image=True,
            ctx_len=args.ctx_len,
            num_token_per_image=args.num_token_per_image,
            pad_token_id=0,
            do_pad_to_max_length=False)
        
        input_ids = data_dict['input_ids'].unsqueeze(0).to(args.device)
        cur_prompt = data_dict['input_text']
        if i == 0:
            print("input_ids.shape: ", input_ids.shape)
            print("input_ids: ", input_ids)
            print("cur_prompt: ", cur_prompt)
            print(f"images.shape: {images.shape}")

        with torch.inference_mode():
            output_ids, output_logits, output_probs = model.generate(
                input_ids,
                images=images,
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                stop_token_idx=STOP_TOKEN_INDEX)

        output = tokenizer.decode(output_ids).split(DEFAULT_STOP_TOKEN)[0].strip()
        # avg logit
        avg_logit = sum(output_logits) / len(output_logits)
        # geometric mean of probs
        avg_prob = np.prod(output_probs) ** (1.0 / len(output_probs))

        out_str = json.dumps({"question_id": idx,
                              "prompt": cur_prompt,
                              "text": output,
                              "avg_logit": str(round(avg_logit, 3)),
                              "avg_prob": str(round(avg_prob, 3)),
                              "model_id": model_name,
                              "metadata": {
                                  **line.get("metadata", {}),
                                  "image_file": line.get("image", None),
                              }}, ensure_ascii=False)
        out_file.write(out_str + "\n")
        # update progress bar
        if i % update_every == 0 and i != 0:
            pbar.update(update_every)
        torch.cuda.empty_cache()
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
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_size_a", default=64, type=int)
    parser.add_argument("--head_size_divisor", default=8, type=int)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--vision_tower_dir",type=str, help="Path to the directory containing the vision tower checkpoints")
    parser.add_argument("--vision_tower_path", type=str, default=None)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--proj_type", default='mlp', type=str, choices=['linear', 'mlp'])
    parser.add_argument("--num_token_per_image", type=int, default=16)
    # arguments for evaluation
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--question_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset_name", type=str, default="default")
    args = parser.parse_args()
    #
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    eval_model(args)