import argparse
import asyncio
import json
import logging
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse


logger = logging.getLogger(__name__)


def run_router(args):
    """
    Run the Miles router with the specified configuration.
    """
    # Initialize the router with tokenizer and lazy worker initialization
    miles_router = MilesRouter(args, verbose=False)

    # Start the server
    uvicorn.run(miles_router.app, host=args.sglang_router_ip, port=args.sglang_router_port, log_level="info")


class MilesRouter:
    def __init__(self, args, verbose=False):
        """Initialize the miles-router with SGLang router address"""
        self.args = args
        self.verbose = verbose

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self._start_background_health_check()
            yield

        self.app = FastAPI(lifespan=lifespan)

        # URL -> Active Request Count (load state)
        self.worker_request_counts: dict[str, int] = {}
        # URL -> Consecutive Failures
        self.worker_failure_counts: dict[str, int] = {}
        # Quarantined workers excluded from routing pool
        self.dead_workers: set[str] = set()

        max_connections = args.miles_router_max_connections
        if max_connections is None:
            max_connections = (
                args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
            )

        timeout = args.miles_router_timeout

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(timeout),
        )

        self._setup_routes()

    def _setup_routes(self):
        """Setup all the HTTP routes"""
        # sglang-router api
        self.app.post("/add_worker")(self.add_worker)
        self.app.post("/remove_worker")(self.remove_worker)
        self.app.get("/pick_worker_for_request")(self.pick_worker_for_request)
        self.app.post("/worker_finish_request")(self.worker_finish_request)
        self.app.get("/list_workers")(self.list_workers)
        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    async def _start_background_health_check(self):
        asyncio.create_task(self._health_check_loop())

    async def _check_worker_health(self, url):
        """Encapsulated health check logic for better maintainability."""
        try:
            response = await self.client.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                return url, True
            logger.debug(f"[miles-router] Worker {url} is unhealthy (Status: {response.status_code})")
        except Exception as e:
            logger.debug(f"[miles-router] Worker {url} health check failed: {e}")
        return url, False

    async def _health_check_loop(self):
        """Background loop to monitor worker health and adjust routing pool."""
        interval = self.args.rollout_health_check_interval
        threshold = self.args.miles_router_health_check_failure_threshold

        while True:
            try:
                await asyncio.sleep(interval)

                urls = [u for u in self.worker_request_counts if u not in self.dead_workers]
                if not urls:
                    continue

                results = await asyncio.gather(*(self._check_worker_health(url) for url in urls))

                for url, is_healthy in results:
                    if not is_healthy:
                        failures = self.worker_failure_counts.get(url, 0) + 1
                        self.worker_failure_counts[url] = failures

                        if failures >= threshold:
                            logger.warning(
                                f"[miles-router] Worker {url} failed {threshold} consecutive health checks. Marking as DEAD."
                            )
                            self.dead_workers.add(url)
                            # TODO (chenyang): Connect back 'dead' workers requires a mechanism to sync
                            # model versions to avoid off-policy issues from stale weights, since these
                            # dead workers' parameters may not be refitted.
                    else:
                        self.worker_failure_counts[url] = 0

                logger.debug(
                    f"[miles-router] Health check complete. {len(self.worker_request_counts) - len(self.dead_workers)} workers healthy."
                )

            except asyncio.CancelledError:
                logger.warning("[miles-router] Background health check loop is being cancelled.")
                raise
            except Exception as e:
                logger.error(f"[miles-router] Unexpected error in health check loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router"""
        # Forward all other paths to SGLang router
        worker_url = self._use_url()
        url = f"{worker_url}/{path}"

        # Get request body and headers
        body = await request.body()
        headers = dict(request.headers)

        try:
            upstream = self.client.build_request(request.method, url, content=body, headers=headers)
            response = await self.client.send(upstream, stream=True)
        except Exception:
            self._finish_url(worker_url)
            raise

        # The relay is byte-transparent, so the worker's headers stay true here;
        # drop only date/server, which uvicorn re-stamps.
        out_headers = dict(response.headers)
        for hop in ("date", "server"):
            out_headers.pop(hop, None)

        async def relay():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                self._finish_url(worker_url)

        return StreamingResponse(relay(), status_code=response.status_code, headers=out_headers)

    async def pick_worker_for_request(self, request: Request):
        """Least-loaded engine URL; the pick counts in-flight until /worker_finish_request acks it.

        A client that dies without acking leaks one count -- acceptable until crashes exist.
        """
        try:
            url = self._use_url()
        except RuntimeError as e:
            return JSONResponse(status_code=503, content={"error": str(e)})
        return {"url": url}

    async def worker_finish_request(self, request: Request):
        worker_url = request.query_params.get("url")
        if not worker_url or worker_url not in self.worker_request_counts:
            return JSONResponse(status_code=400, content={"error": f"unknown worker url {worker_url!r}"})
        self._finish_url(worker_url)
        return {"status": "success"}

    async def add_worker(self, request: Request):
        """Add a new worker to the router.
        Supports providing the URL via query string or JSON body.
        Examples:
        - POST /add_worker?url=http://127.0.0.1:10090
        - POST /add_worker  with body {"url": "http://127.0.0.1:10090"}
        """
        # 1) Prefer query param
        worker_url = request.query_params.get("url") or request.query_params.get("worker_url")

        # 2) Fallback to JSON body
        if not worker_url:
            body = await request.body()
            payload = json.loads(body) if body else {}
            worker_url = payload.get("url") or payload.get("worker_url")

        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        # Add if new, keep a simple request count per worker
        if worker_url not in self.worker_request_counts:
            self.worker_request_counts[worker_url] = 0
            self.worker_failure_counts[worker_url] = 0
            if self.verbose:
                print(f"[miles-router] Added new worker: {worker_url}")

        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def remove_worker(self, request: Request):
        """Drop a worker from the routing pool, same URL forms as add_worker.

        Engines call this on shutdown. Without the route it fell through to the
        catch-all and was proxied to a worker as if it were a generate request.
        """
        worker_url = request.query_params.get("url") or request.query_params.get("worker_url")
        if not worker_url:
            body = await request.body()
            payload = json.loads(body) if body else {}
            worker_url = payload.get("url") or payload.get("worker_url")

        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        self.worker_request_counts.pop(worker_url, None)
        self.worker_failure_counts.pop(worker_url, None)
        self.dead_workers.discard(worker_url)
        if self.verbose:
            print(f"[miles-router] Removed worker: {worker_url}")
        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def list_workers(self, request: Request):
        """List all registered workers"""
        return {"urls": list(self.worker_request_counts.keys())}

    def _use_url(self):
        """Select the worker URL with the fewest active requests."""
        candidates = (w for w in self.worker_request_counts if w not in self.dead_workers)
        try:
            url = min(candidates, key=self.worker_request_counts.get)
        except ValueError:
            raise RuntimeError("No healthy workers available in the pool") from None

        self.worker_request_counts[url] += 1
        return url

    def _finish_url(self, url):
        """Mark the request to the given URL as finished"""
        assert url in self.worker_request_counts, f"URL {url} not recognized"
        self.worker_request_counts[url] -= 1
        assert self.worker_request_counts[url] >= 0, f"URL {url} count went negative"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--sglang-host", type=str, required=True)
    parser.add_argument("--sglang-port", type=int, required=True)
    parser.add_argument("--tokenizer-name", type=str, help="Name of the tokenizer to use for tokenization")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Run the router
    run_router(args)
