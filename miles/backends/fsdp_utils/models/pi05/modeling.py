"""π₀.₅ model-side training behavior."""


def enable_gradient_checkpointing(model):
    model.gradient_checkpointing_enable()
