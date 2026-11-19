"""Attention selection for π₀.₅'s eager joint training forward."""


def set_attention_backend(model, backend):
    if backend != "eager":
        raise ValueError("π₀.₅ joint training attention supports only eager attention here")
