"""Tests for VLA (Vision-Language-Action) model components."""

import pytest
import torch

from Vela7.src.models.vla import FlowMatchingHead, SinusoidalTimestepEncoder, info_nce_loss
from Vela7.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER


class TestInfoNCELoss:
    """Test InfoNCE contrastive loss for embeddings."""

    def test_info_nce_returns_scalar(self):
        """Verify InfoNCE returns a scalar loss value."""
        pred = torch.randn(2, 10, 256)
        target = torch.randn(2, 10, 256)
        negatives = torch.randn(100, 256)

        loss = info_nce_loss(pred, target, negatives)

        assert loss.dim() == 0, "InfoNCE loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"

    def test_info_nce_gradient_flow(self):
        """Verify gradients flow through InfoNCE loss."""
        pred = torch.randn(2, 10, 256, requires_grad=True)
        target = torch.randn(2, 10, 256)
        negatives = torch.randn(100, 256)

        loss = info_nce_loss(pred, target, negatives)
        loss.backward()

        assert pred.grad is not None, "Gradient should exist on pred"
        assert not torch.isnan(pred.grad).any(), "Gradient should not contain NaN"

    def test_info_nce_similar_embeddings_higher_similarity(self):
        """InfoNCE should give higher similarity to similar embeddings."""
        pred = torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float32)
        target = torch.tensor([[[[0.9, 0.1, 0.1]]]], dtype=torch.float32)
        negatives = torch.tensor(
            [[-1.0, 0.0, 0.0], [-0.5, 0.5, 0.5]],
            dtype=torch.float32
        )

        loss = info_nce_loss(pred, target, negatives)

        # Similar embeddings should produce lower loss
        assert loss.item() < 1.0, "Similar embeddings should minimize loss"


class TestFlowMatchingHead:
    """Test Flow Matching head for action prediction."""

    def test_timestep_encoding(self):
        """Verify timestep encoder produces valid embeddings."""
        encoder = SinusoidalTimestepEncoder(embedding_dim=128)
        timesteps = torch.tensor([0.0, 0.5, 1.0])

        emb = encoder(timesteps)

        assert emb.shape == (3, 128), "Embedding shape mismatch"
        assert torch.isfinite(emb).all(), "Embeddings should be finite"

    def test_flow_head_forward_shape(self):
        """Verify FlowMatchingHead produces correct output shape."""
        head = FlowMatchingHead(
            hidden_size=256,
            action_dim=14,
            action_horizon=16,
        )

        noise = torch.randn(2, 16, 14)
        V_blocks = torch.randn(4, 2, 64, 256)
        partial_block = torch.randn(2, 64, 256)

        output = head(noise, V_blocks, partial_block)

        assert output.shape == (2, 16, 14), f"Expected (2, 16, 14), got {output.shape}"

    def test_flow_head_gradient_flow(self):
        """Verify gradients flow through FlowMatchingHead."""
        head = FlowMatchingHead(
            hidden_size=128,
            action_dim=14,
            action_horizon=16,
        )

        noise = torch.randn(2, 16, 14, requires_grad=True)
        V_blocks = torch.randn(4, 2, 32, 128)
        partial_block = torch.randn(2, 32, 128)

        output = head(noise, V_blocks, partial_block)
        loss = output.sum()
        loss.backward()

        assert noise.grad is not None, "Gradient should exist on noise"
        assert not torch.isnan(noise.grad).any(), "Gradient should not be NaN"


class TestVLATokens:
    """Test VLA special token integration."""

    def test_think_tokens_in_vocab(self):
        """Verify think tokens are in tokenizer vocabulary."""
        tokenizer = TRIE_TOKENIZER("VELA-v7/tokenizer/rwkv_vocab_v20230424.txt")

        think_ids = tokenizer.encode("<script>")
        assert len(think_ids) > 0, "Think start token should be in vocab"

        think_end_ids = tokenizer.encode("</script>")
        assert len(think_end_ids) > 0, "Think end token should be in vocab"

        act_ids = tokenizer.encode("<act>")
        assert len(act_ids) > 0, "Act token should be in vocab"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
