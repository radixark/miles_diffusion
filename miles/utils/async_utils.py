import asyncio
import threading

__all__ = ["get_async_loop", "ray_get_cancellable", "run"]


# Create a background event loop thread
class AsyncLoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro):
        # Schedule a coroutine onto the loop and block until it's done
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


# Create one global instance
async_loop = None


def get_async_loop():
    global async_loop
    if async_loop is None:
        async_loop = AsyncLoopThread()
    return async_loop


def run(coro):
    """Run a coroutine in the background event loop."""
    return get_async_loop().run(coro)


async def ray_get_cancellable(submit):
    """Await the ObjectRef from submit() off the event loop, and ray.cancel it if this task is cancelled.

    submit (e.g. lambda: actor.method.remote(...)) runs in a worker thread, since .remote() serializes its
    arguments in the calling thread. Cancelling the asyncio task alone leaves the Ray call running.
    """
    import ray

    def cancel_when_submitted(fut):
        if not fut.cancelled() and fut.exception() is None:
            ray.cancel(fut.result())

    submitted = asyncio.ensure_future(asyncio.to_thread(submit))
    try:
        ref = await asyncio.shield(submitted)
    except asyncio.CancelledError:
        # the .remote() call may still land after we were cancelled; cancel it once it has a ref
        submitted.add_done_callback(cancel_when_submitted)
        raise
    try:
        return await asyncio.to_thread(ray.get, ref)
    except asyncio.CancelledError:
        ray.cancel(ref)
        raise
