class TrainProfiler:
    """No-op profiler shim. Production never enables profiling; the actor
    keeps the call sites so a future profiler can drop in here."""

    def __init__(self, args):
        del args

    def on_init_end(self):
        pass

    def step(self, rollout_id: int):
        del rollout_id
