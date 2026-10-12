import torch

from pixelwm.models.vq import VectorQuantizer, VQTokenizer


def _tokenizer() -> VQTokenizer:
    torch.manual_seed(0)
    return VQTokenizer(num_codes=16, code_dim=8, depth=8)


def test_tokenizer_shapes_and_id_range():
    tokenizer = _tokenizer()
    x = torch.randn(3, 3, 64, 64) * 0.2
    ids = tokenizer.encode_ids(x)
    assert ids.shape == (3, 64) and ids.dtype == torch.long
    assert ids.min() >= 0 and ids.max() < 16
    assert tokenizer.decode_ids(ids).shape == (3, 3, 64, 64)
    recon, ids2, commitment = tokenizer(x, update=False)
    assert recon.shape == x.shape and ids2.shape == (3, 64) and commitment.ndim == 0


def test_decode_in_chunks_matches_a_single_pass():
    tokenizer = _tokenizer()
    ids = torch.randint(0, 16, (5, 64))
    small = tokenizer.decode_ids(ids, chunk=2)
    large = tokenizer.decode_ids(ids, chunk=100)
    assert torch.allclose(small, large, atol=1e-5)


def test_codebook_only_moves_when_updating():
    tokenizer = _tokenizer()
    x = torch.randn(4, 3, 64, 64) * 0.2
    tokenizer(x, update=True)  # initialises the codebook from the encoder outputs
    before = tokenizer.quantizer.codebook.clone()
    tokenizer(x, update=False)
    assert torch.equal(before, tokenizer.quantizer.codebook)
    tokenizer(x, update=True)
    assert not torch.equal(before, tokenizer.quantizer.codebook)


def test_straight_through_gradient_reaches_the_encoder_output():
    quantizer = VectorQuantizer(num_codes=8, dim=4)
    z = torch.randn(10, 4, requires_grad=True)
    quantised, indices, _ = quantizer(z, update=False)
    quantised.sum().backward()
    assert indices.shape == (10,)
    assert z.grad is not None and torch.allclose(z.grad, torch.ones_like(z))


def test_dead_codes_are_restarted():
    quantizer = VectorQuantizer(num_codes=8, dim=4, restart_every=1, dead_threshold=100.0)
    quantizer(torch.randn(64, 4), update=True)
    # every code is below the (huge) threshold, so every code was re-initialised
    assert torch.all(quantizer.ema_count == 1.0)
