import torch
from torch.nn import functional as F


def cached_layer(layer, inputs, previous=None):
    """Post-norm Transformer layer in evaluation mode, with optional past K/V."""
    b, t, d = inputs.shape
    heads = layer.self_attn.num_heads
    q, k, v = F.linear(inputs, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias).chunk(3, -1)
    q, k, v = [x.view(b, t, heads, d // heads).transpose(1, 2) for x in (q, k, v)]
    if previous is not None:
        k, v = torch.cat((previous[0], k), -2), torch.cat((previous[1], v), -2)
    attended = F.scaled_dot_product_attention(q, k, v, is_causal=previous is None, dropout_p=0.0)
    attended = attended.transpose(1, 2).reshape(b, t, d)
    x = layer.norm1(inputs + layer.self_attn.out_proj(attended))
    return layer.norm2(x + layer.linear2(layer.activation(layer.linear1(x)))), (k, v)


def prefill(layers, tokens):
    caches = []
    for layer in layers:
        tokens, cache = cached_layer(layer, tokens)
        caches.append(cache)
    return tokens, caches


def decode(layers, token, caches):
    for i, layer in enumerate(layers):
        token, caches[i] = cached_layer(layer, token, caches[i])
    return token
