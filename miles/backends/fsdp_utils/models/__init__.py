"""Self-built (non-diffusers) modeling for FSDP training.

Onboarding a model family:

- **Diffusers checkpoint** (has ``model_index.json``):
  ``DiffusersModelBackend`` loads it; add
  ``models/diffusers/<family>/parallel_plan.py`` for its FSDP precision plan.

- **Native modeling** (official repo code, non-diffusers checkpoint): add a
  package ``models/<family>/`` with at least:

  - ``loading.py`` — checkpoint resolution/materialization and ``load_component``
  - ``modeling.py`` — ``enable_gradient_checkpointing``,
    optional ``flash_attention_entrypoints`` /
    ``required_flash_kernel_label`` for deterministic flash patching
  - ``parallel_plan.py`` — ``FSDP_PARALLEL_PLAN``,
    ``sequence_parallel_plan``
  - ``attention.py`` — ``set_attention_backend``

  Point ``configs/<family>.py`` at::

      model_backend_path = "...model_backend.MilesModelBackend"
      model_package = "...models.<family>"

  Model-specific forward semantics live on the family config
  (``compute_noise_pred`` override), not in the trainer.

  See ``models/ltx/`` for the reference package.
"""
