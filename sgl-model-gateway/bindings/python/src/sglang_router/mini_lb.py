"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import asyncio
import ipaddress
import logging
import random
import time
import urllib
from http import HTTPStatus
from itertools import chain
from typing import Optional

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, Response, StreamingResponse
from sglang_router.router_args import RouterArgs

try:
    from sglang.srt.tracing.trace import (
        process_tracing_init,
        trace_get_remote_propagate_context,
        trace_req_finish,
        trace_req_start,
        trace_set_thread_info,
        trace_slice_end,
        trace_slice_start,
    )

    trace_package_imported = True
except ImportError:
    trace_package_imported = False

logger = logging.getLogger(__name__)

AIOHTTP_STREAM_READ_CHUNK_SIZE = (
    1024 * 64
)  # 64KB, to prevent aiohttp's "Chunk too big" error


def maybe_wrap_ipv6_address(address: str) -> str:
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ValueError:
        return address


class MiniLoadBalancer:
    def __init__(
        self,
        router_args: RouterArgs,
    ):
        self._validate_router_args(router_args)

        self.host = router_args.host
        self.port = router_args.port
        self.timeout = router_args.request_timeout_secs
        self.prefill_urls = [url[0] for url in router_args.prefill_urls]
        self.prefill_bootstrap_ports = [url[1] for url in router_args.prefill_urls]
        self.decode_urls = router_args.decode_urls
        self.otlp_traces_endpoint = router_args.otlp_traces_endpoint
        self.enable_trace = router_args.enable_trace
        if self.enable_trace and not trace_package_imported:
            logger.warning(
                "Tracing is not supported in this environment. Please install sglang."
            )
            self.enable_trace = False

        # LAPS dynamic allocation state
        self.enable_laps_alloc = router_args.enable_laps_alloc
        if self.enable_laps_alloc:
            n = len(self.prefill_urls)
            if n < 2:
                logger.warning(
                    "[LAPS] Dynamic allocation requires >=2 prefill instances, disabling"
                )
                self.enable_laps_alloc = False
            else:
                mid = n // 2
                self._laps_short_group = list(range(0, mid))
                self._laps_long_group = list(range(mid, n))
                self._laps_short_pending = 0
                self._laps_long_pending = 0
                self._laps_last_rebalance = 0.0
                self._laps_threshold = router_args.laps_alloc_threshold
                self._laps_rebalance_interval = router_args.laps_rebalance_interval_s
                self._laps_rebalance_ratio = router_args.laps_rebalance_ratio
                logger.info(
                    f"[LAPS] Dynamic allocation enabled: "
                    f"short_group={self._laps_short_group}, "
                    f"long_group={self._laps_long_group}, "
                    f"threshold={self._laps_threshold}"
                )

    def _validate_router_args(self, router_args: RouterArgs):
        logger.warning(
            "\x1b[33mMiniLB is only for debugging purposes, it only supports random policy!\033[0m"
        )

        # NOTE: too many arguments unsupported, just validate some important ones
        if router_args.policy != "random":
            logger.warning("[MiniLB] Overriding policy to random")
            router_args.policy = "random"

        if not router_args.pd_disaggregation:
            raise ValueError("MiniLB only supports PD disaggregation mode")

        if len(router_args.prefill_urls) == 0 or len(router_args.decode_urls) == 0:
            raise ValueError(
                "MiniLB requires at least one prefill and one decode server"
            )

    def start(self):
        global lb
        lb = self
        if self.enable_trace:
            process_tracing_init(self.otlp_traces_endpoint, "sglang")
            trace_set_thread_info("Mini lb")
        uvicorn.run(app, host=self.host, port=self.port)

    def select_pair(self):
        assert len(self.prefill_urls) > 0, "No prefill servers available"
        assert len(self.decode_urls) > 0, "No decode servers available"
        pidx = random.randint(0, len(self.prefill_urls) - 1)
        didx = random.randint(0, len(self.decode_urls) - 1)
        return (
            self.prefill_urls[pidx],
            self.prefill_bootstrap_ports[pidx],
            self.decode_urls[didx],
        )

    def _laps_classify(self, request_data: dict) -> str:
        """Classify request as 'short' or 'long' based on estimated token count."""
        tokens = _estimate_prompt_tokens(request_data)
        if tokens == 0:
            # Unknown length — route to lower-pressure group
            if self._laps_short_pending <= self._laps_long_pending:
                return "short"
            return "long"
        return "short" if tokens <= self._laps_threshold else "long"

    def _laps_maybe_rebalance(self):
        """Time-gated inline rebalance: move at most 1 instance between groups."""
        now = time.monotonic()
        if now - self._laps_last_rebalance < self._laps_rebalance_interval:
            return

        self._laps_last_rebalance = now
        sp = max(self._laps_short_pending, 1)
        lp = max(self._laps_long_pending, 1)
        ratio = self._laps_rebalance_ratio

        if sp > lp * ratio and len(self._laps_long_group) > 1:
            moved = self._laps_long_group.pop()
            self._laps_short_group.append(moved)
            logger.info(
                f"[LAPS] Rebalance: moved prefill[{moved}] long→short "
                f"(short_pending={self._laps_short_pending}, long_pending={self._laps_long_pending})"
            )
        elif lp > sp * ratio and len(self._laps_short_group) > 1:
            moved = self._laps_short_group.pop()
            self._laps_long_group.append(moved)
            logger.info(
                f"[LAPS] Rebalance: moved prefill[{moved}] short→long "
                f"(short_pending={self._laps_short_pending}, long_pending={self._laps_long_pending})"
            )

    def select_pair_laps(self, request_data: dict):
        """Select prefill/decode pair using LAPS dynamic allocation.

        Returns (prefill_url, bootstrap_port, decode_url, category).
        """
        self._laps_maybe_rebalance()
        category = self._laps_classify(request_data)

        if category == "short":
            group = self._laps_short_group if self._laps_short_group else self._laps_long_group
        else:
            group = self._laps_long_group if self._laps_long_group else self._laps_short_group

        pidx = group[random.randint(0, len(group) - 1)]
        didx = random.randint(0, len(self.decode_urls) - 1)

        if category == "short":
            self._laps_short_pending += 1
        else:
            self._laps_long_pending += 1

        return (
            self.prefill_urls[pidx],
            self.prefill_bootstrap_ports[pidx],
            self.decode_urls[didx],
            category,
        )

    def laps_request_done(self, category: str):
        """Decrement the pending counter for the given category."""
        if category == "short":
            self._laps_short_pending = max(0, self._laps_short_pending - 1)
        else:
            self._laps_long_pending = max(0, self._laps_long_pending - 1)

    async def generate(
        self, modified_request, prefill_server, decode_server, endpoint
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.timeout
            )  # Add timeout for request reliability
        ) as session:
            headers = {}
            bootstrap_room_list = []
            if self.enable_trace:
                bootstrap_room_list = (
                    modified_request["bootstrap_room"]
                    if isinstance(modified_request["bootstrap_room"], list)
                    else [modified_request["bootstrap_room"]]
                )
                trace_context = trace_get_remote_propagate_context(bootstrap_room_list)
                headers = {"trace_context": trace_context}

            tasks = [
                session.post(
                    f"{prefill_server}/{endpoint}",
                    json=modified_request,
                    headers=headers,
                ),
                session.post(
                    f"{decode_server}/{endpoint}",
                    json=modified_request,
                    headers=headers,
                ),
            ]

            for bootstrap_room in bootstrap_room_list:
                trace_slice_end("mini_lb_launch", bootstrap_room, auto_next_anon=True)

            # Wait for both responses to complete. Prefill should end first.
            prefill_response, decode_response = await asyncio.gather(*tasks)

            if "return_logprob" in modified_request:

                prefill_json = await prefill_response.json()
                ret_json = await decode_response.json()

                # merge `meta_info.input_token_logprobs` from prefill to decode
                if "meta_info" in ret_json:
                    if "input_token_logprobs" in ret_json["meta_info"]:
                        ret_json["meta_info"]["input_token_logprobs"] = (
                            prefill_json["meta_info"]["input_token_logprobs"]
                            + ret_json["meta_info"]["input_token_logprobs"]
                        )
            else:
                ret_json = await decode_response.json()

            for bootstrap_room in bootstrap_room_list:
                trace_slice_end(
                    "wait_PD_finish",
                    bootstrap_room,
                    thread_finish_flag=True,
                )
                trace_req_finish(bootstrap_room)

            return ORJSONResponse(
                content=ret_json,
                status_code=decode_response.status,
            )

    async def generate_stream(
        self, modified_request, prefill_server, decode_server, endpoint="generate"
    ):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.timeout
                )  # Add timeout for request reliability
            ) as session:
                # Create the tasks for both prefill and decode requests
                headers = {}
                bootstrap_room_list = []
                if self.enable_trace:
                    bootstrap_room_list = (
                        modified_request["bootstrap_room"]
                        if isinstance(modified_request["bootstrap_room"], list)
                        else [modified_request["bootstrap_room"]]
                    )
                    trace_context = trace_get_remote_propagate_context(
                        bootstrap_room_list
                    )
                    headers = {"trace_context": trace_context}

                tasks = [
                    session.post(
                        f"{prefill_server}/{endpoint}",
                        json=modified_request,
                        headers=headers,
                    ),
                    session.post(
                        f"{decode_server}/{endpoint}",
                        json=modified_request,
                        headers=headers,
                    ),
                ]

                for bootstrap_room in bootstrap_room_list:
                    trace_slice_end(
                        "mini_lb_launch", bootstrap_room, auto_next_anon=True
                    )
                # Wait for both responses to complete. Since this is streaming, they return immediately.
                prefill_response, decode_response = await asyncio.gather(*tasks)

                if modified_request.get("return_logprob", False):
                    prefill_chunks = []
                    async for chunk in prefill_response.content:
                        prefill_chunks.append(chunk)

                    first_prefill_chunk = (
                        prefill_chunks[0].decode("utf-8")[5:].strip("\n")
                    )
                    first_prefill_chunk_json = orjson.loads(first_prefill_chunk)

                    async for chunk in decode_response.content:
                        # Note: This is inefficient
                        # merge prefill input_token_logprobs, output_token_logprobs to decode
                        decoded_chunk = chunk.decode("utf-8")
                        if (
                            decoded_chunk
                            and decoded_chunk.startswith("data:")
                            and "[DONE]" not in decoded_chunk
                        ):
                            ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                            ret_json["meta_info"]["input_token_logprobs"] = (
                                first_prefill_chunk_json["meta_info"][
                                    "input_token_logprobs"
                                ]
                                + ret_json["meta_info"]["input_token_logprobs"]
                            )

                            yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                        else:
                            yield chunk
                else:
                    async for chunk in decode_response.content.iter_chunked(
                        AIOHTTP_STREAM_READ_CHUNK_SIZE
                    ):
                        yield chunk

            for bootstrap_room in bootstrap_room_list:
                trace_slice_end(
                    "wait_PD_finish",
                    bootstrap_room,
                    thread_finish_flag=True,
                )
                trace_req_finish(bootstrap_room)

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )


app = FastAPI()
lb: Optional[MiniLoadBalancer] = None


@app.get("/health")
async def health_check():
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate():
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(lb.prefill_urls, lb.decode_urls):
            tasks.append(session.get(f"{server}/health_generate"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.post("/flush_cache")
async def flush_cache():
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(lb.prefill_urls, lb.decode_urls):
            tasks.append(session.post(f"{server}/flush_cache"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.get("/laps_status")
async def laps_status():
    if not lb or not lb.enable_laps_alloc:
        return ORJSONResponse(content={"enabled": False})
    return ORJSONResponse(
        content={
            "enabled": True,
            "threshold": lb._laps_threshold,
            "rebalance_interval_s": lb._laps_rebalance_interval,
            "rebalance_ratio": lb._laps_rebalance_ratio,
            "short_group": [lb.prefill_urls[i] for i in lb._laps_short_group],
            "long_group": [lb.prefill_urls[i] for i in lb._laps_long_group],
            "short_group_indices": lb._laps_short_group,
            "long_group_indices": lb._laps_long_group,
            "short_pending": lb._laps_short_pending,
            "long_pending": lb._laps_long_pending,
        }
    )


@app.get("/get_server_info")
async def get_server_info():
    prefill_infos = []
    decode_infos = []
    all_internal_states = []

    async with aiohttp.ClientSession() as session:
        for server in lb.prefill_urls:
            server_info = await session.get(f"{server}/get_server_info")
            prefill_infos.append(await server_info.json())
        for server in lb.decode_urls:
            server_info = await session.get(f"{server}/get_server_info")
            info_json = await server_info.json()
            decode_infos.append(info_json)
            # Extract internal_states from decode servers
            if "internal_states" in info_json:
                all_internal_states.extend(info_json["internal_states"])

    # Return format expected by bench_one_batch_server.py
    if all_internal_states:
        return {
            "internal_states": all_internal_states,
            "prefill": prefill_infos,
            "decode": decode_infos,
        }
    else:
        # Fallback with dummy data if no internal states found
        return {
            "internal_states": [
                {
                    "last_gen_throughput": 0.0,
                    "avg_spec_accept_length": None,
                }
            ],
            "prefill": prefill_infos,
            "decode": decode_infos,
        }


async def _get_model_info_impl():
    if not lb or not lb.prefill_urls:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="There is no server registered",
        )

    target_server_url = lb.prefill_urls[0]
    endpoint_url = f"{target_server_url}/model_info"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_GATEWAY,
                        detail=(
                            f"Failed to get model info from {target_server_url}"
                            f"Status: {response.status}, Response: {error_text}"
                        ),
                    )

                model_info_json = await response.json()
                return ORJSONResponse(content=model_info_json)

        except aiohttp.ClientError as e:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail=f"Failed to get model info from backend",
            )


@app.get("/model_info")
async def model_info():
    return await _get_model_info_impl()


@app.get("/get_model_info")
async def get_model_info():
    return await _get_model_info_impl()


@app.post("/generate")
async def handle_generate_request(request_data: dict):
    laps_category = None
    if lb.enable_laps_alloc:
        prefill_server, bootstrap_port, decode_server, laps_category = (
            lb.select_pair_laps(request_data)
        )
    else:
        prefill_server, bootstrap_port, decode_server = lb.select_pair()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
    modified_request = request_data.copy()

    batch_size = _get_request_batch_size(modified_request)
    if batch_size is not None:
        modified_request.update(
            {
                "bootstrap_host": [hostname] * batch_size,
                "bootstrap_port": [bootstrap_port] * batch_size,
                "bootstrap_room": [
                    _generate_bootstrap_room() for _ in range(batch_size)
                ],
            }
        )
    else:
        modified_request.update(
            {
                "bootstrap_host": hostname,
                "bootstrap_port": bootstrap_port,
                "bootstrap_room": _generate_bootstrap_room(),
            }
        )

    try:
        if request_data.get("stream", False):
            return await lb.generate_stream(
                modified_request, prefill_server, decode_server, "generate"
            )
        else:
            return await lb.generate(
                modified_request, prefill_server, decode_server, "generate"
            )
    finally:
        if laps_category is not None:
            lb.laps_request_done(laps_category)


async def _forward_to_backend(request_data: dict, endpoint_name: str):
    laps_category = None
    if lb.enable_laps_alloc:
        prefill_server, bootstrap_port, decode_server, laps_category = (
            lb.select_pair_laps(request_data)
        )
    else:
        prefill_server, bootstrap_port, decode_server = lb.select_pair()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
    modified_request = request_data.copy()
    modified_request.update(
        {
            "bootstrap_host": hostname,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": _generate_bootstrap_room(),
        }
    )

    try:
        if request_data.get("stream", False):
            return await lb.generate_stream(
                modified_request,
                prefill_server,
                decode_server,
                endpoint=endpoint_name,
            )
        else:
            return await lb.generate(
                modified_request,
                prefill_server,
                decode_server,
                endpoint=endpoint_name,
            )
    finally:
        if laps_category is not None:
            lb.laps_request_done(laps_category)


@app.post("/v1/chat/completions")
async def handle_chat_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/chat/completions")


@app.post("/v1/completions")
async def handle_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/completions")


def _generate_bootstrap_room():
    bootstrap_room = random.randint(0, 2**63 - 1)
    if lb.enable_trace:
        trace_req_start(bootstrap_room, bootstrap_room, role="router")
        trace_slice_start("mini_lb_launch", bootstrap_room)
    return bootstrap_room


# We may utilize `GenerateReqInput`'s logic later
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    return None


def _estimate_prompt_tokens(request_data: dict) -> int:
    """Estimate the number of prompt tokens from request data.

    Uses exact count from input_ids, or chars//4 heuristic for text.
    For batch requests, returns the max length across items.
    Returns 0 if nothing recognized (caller should use random fallback).
    """
    # input_ids: exact token count
    input_ids = request_data.get("input_ids")
    if input_ids is not None:
        if isinstance(input_ids, list) and len(input_ids) > 0:
            if isinstance(input_ids[0], int):
                return len(input_ids)
            else:
                # list-of-lists (batch)
                return max(len(ids) for ids in input_ids)
        return 0

    # text: chars // 4 heuristic
    text = request_data.get("text")
    if text is not None:
        if isinstance(text, str):
            return len(text) // 4
        elif isinstance(text, list):
            return max((len(t) // 4 for t in text), default=0)
        return 0

    # messages (chat completions): sum content lengths // 4
    messages = request_data.get("messages")
    if messages is not None:
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # multimodal content parts
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total_chars += len(part.get("text", ""))
        return total_chars // 4

    return 0


@app.get("/v1/models")
async def get_models():
    prefill_server = lb.prefill_urls[0]  # Get the first prefill server
    async with aiohttp.ClientSession() as session:
        try:
            response = await session.get(f"{prefill_server}/v1/models")
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Prefill server error: Status {response.status}",
                )
            return ORJSONResponse(content=await response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
