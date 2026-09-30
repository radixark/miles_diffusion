"""Set a Diffusers attention backend on both frozen and control blocks."""


def set_attention_backend(model, backend):
    model.set_attention_backend(backend)
