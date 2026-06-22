"""Unit tests for Vela7 early fusion, ChatML formatting, and MHC MoE routing."""

from argparse import Namespace

import pytest
import torch

from Vela7.src.dataset import (
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    preprocess,
    process_image_tokens_in_conversations,
)
from Vela7.src.model import VELA, Sinkhorn_Knopp
from Vela7.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER


def test_chatml_formatting():
    """Verify that ChatML formatting and dynamic target masking work correctly."""
    tokenizer = TRIE_TOKENIZER("VELA-v7/tokenizer/rwkv_vocab_v20230424.txt")

    # Test conversation with custom metadata
    conversations = [
        {"from": "system", "value": "You are a helpful assistant.", "metadata": "developer"},
        {"from": "human", "value": "Hello! <image>", "metadata": "alice"},
        {"from": "gpt", "value": "Hi! How can I help you today?", "metadata": "bob"},
    ]

    # 1. Process image tokens
    num_regions = 2
    conversations = process_image_tokens_in_conversations(conversations, num_regions=num_regions)

    # Verify in-place image tokens are wrapped with <img_start> and <img_end>
    user_turn = conversations[1]["value"]
    assert "<img_start>" in user_turn
    assert "<img_end>" in user_turn
    assert f"<img_start>{DEFAULT_IMAGE_TOKEN}{DEFAULT_IMAGE_TOKEN}<img_end>" in user_turn

    # 2. Run preprocess
    ctx_len = 128
    num_token_per_image = 4
    data_dict = preprocess(
        conversations,
        tokenizer,
        has_image=True,
        ctx_len=ctx_len,
        num_token_per_image=num_token_per_image,
        do_pad_to_max_length=True,
    )

    input_ids = data_dict["input_ids"]
    labels = data_dict["labels"]
    input_text = data_dict["input_text"]

    # Check input_text contains ChatML format tags
    assert "<im_start>system:developer\n" in input_text
    assert "<im_start>user:alice\n" in input_text
    assert "<im_start>assistant:bob\n" in input_text
    assert "<im_end>\n" in input_text

    # Check label masking
    # Non-assistant tokens should be masked
    # Let's decode tokens that are not masked
    unmasked_indices = (labels != IGNORE_INDEX).nonzero().squeeze(-1)
    unmasked_tokens = [input_ids[idx].item() for idx in unmasked_indices]
    decoded_unmasked = tokenizer.decode(unmasked_tokens)

    # Only assistant's content and closing tags should be unmasked
    assert "Hi! How can I help you today?" in decoded_unmasked
    assert "user:alice" not in decoded_unmasked
    assert "system:developer" not in decoded_unmasked


def test_sinkhorn_knopp():
    """Verify that Sinkhorn-Knopp algorithm produces doubly stochastic matrices."""
    B, T = 2, 8
    X = torch.randn(B, T, 4, 4)
    M = Sinkhorn_Knopp(X, tmax=20)

    # Rows must sum to 1
    row_sums = M.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    # Columns must sum to 1
    col_sums = M.sum(dim=-2)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-5)

    assert (M >= 0).all(), "Sinkhorn-Knopp output should be non-negative"
    assert not torch.isnan(M).any(), "NaN in Sinkhorn-Knopp output"
    assert not torch.isinf(M).any(), "Inf in Sinkhorn-Knopp output"


def test_moe_forward_backward():
    """Verify full forward/backward flow through the MHC MoE model layers."""
    args = Namespace()
    args.n_layer = 4
    args.n_embd = 128
    args.vocab_size = 65536
    args.dim_att = 128
    args.dim_ffn = 512
    args.head_size_a = 32
    args.head_size_divisor = 8
    args.dropout = 0.0
    args.grad_cp = 0
    args.ctx_len = 32
    args.my_pos_emb = 0
    args.my_pile_stage = 1
    args.pre_ffn = 0
    args.head_size = 32
    args.load_model = ""
    args.n_attnres_blocks = 2
    args.vision_tower_path = "tiiuae/siglino-30M"
    args.n_vtc_layer = 1
    args.num_token_per_image = 4

    # Instantiate VELA model
    model = VELA(args).bfloat16()
    model.train()

    # Create dummy batch
    B, T = 2, 32
    input_ids = torch.randint(0, 65530, (B, T), dtype=torch.long)
    # Inject some image token indexes
    input_ids[0, 5:9] = 65535
    labels = torch.randint(0, 65530, (B, T), dtype=torch.long)

    # Generate synthetic image input
    B_N, L_patches = 1, 16
    patch_dim = 3 * 16 * 16
    images = torch.rand(B_N, 1, L_patches, patch_dim, dtype=torch.bfloat16)
    spatial_shapes = torch.tensor([[4, 4]], dtype=torch.long)

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "images": images,
        "spatial_shapes": spatial_shapes,
        "sample_id": ["sample_1", "sample_2"],
    }

    # Run forward step
    logits, targets = model(batch)

    assert logits.shape == (B, T, args.vocab_size)
    assert not torch.isnan(logits).any(), "NaN in forward logits"
    assert not torch.isinf(logits).any(), "Inf in forward logits"

    # Run backward step
    loss = logits.sum()
    loss.backward()

    # Verify gradients flow and are finite for MoE / mHC parameters
    mhc_blocks = model.rwkv.blocks[:4]
    for block in mhc_blocks:
        # Check phi gradients
        assert block.phi_pre_att.grad is not None
        assert block.phi_pre_ffn.grad is not None
        assert not torch.isnan(block.phi_pre_att.grad).any()
        assert not torch.isnan(block.phi_pre_ffn.grad).any()

        # Check experts gradients
        for expert in block.experts:
            assert expert.key.weight.grad is not None
            assert not torch.isnan(expert.key.weight.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
