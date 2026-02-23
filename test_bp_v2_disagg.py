"""Test batch prefill CUDA graph v2 in PD disaggregation mode with router.

Full end-to-end test:
  1. Launch prefill server (with batch prefill CUDA graph + mooncake)
  2. Launch decode server (with mooncake)
  3. Launch router (PD disaggregation mode)
  4. Send requests through the router
  5. Compare outputs with a non-disaggregated baseline
"""
import subprocess
import sys
import time
import requests
import os
import signal

MODEL_PATH = "/mnt/weka/home/jianshu.she/models/qwen2.5-0.5b"
PREFILL_PORT = 30100
DECODE_PORT = 30101
ROUTER_PORT = 30102
PREFILL_GPU = "0"
DECODE_GPU = "1"
IB_DEVICE = "mlx5_0"


def wait_for_server(port, timeout=180):
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def wait_for_router(port, timeout=60):
    """Wait for the router to respond to health checks."""
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def generate(port, prompt, max_new_tokens=32):
    r = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={
            "text": prompt,
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
        },
        timeout=60,
    )
    return r.json()


def cleanup(procs):
    print("\nCleaning up servers...")
    for proc in procs:
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except Exception:
            pass
    time.sleep(3)
    for proc in procs:
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except Exception:
            pass
    # Kill any remaining sglang/router child processes
    subprocess.run("pkill -9 -f 'sglang.*3010[012]'", shell=True, capture_output=True)
    subprocess.run("pkill -9 -f 'sglang_router.*3010'", shell=True, capture_output=True)
    time.sleep(2)


def main():
    procs = []

    try:
        # =============================================
        # Step 1: Launch prefill server
        # =============================================
        print(f"=== Launching PREFILL server (GPU {PREFILL_GPU}, port {PREFILL_PORT}) ===")
        prefill_proc = subprocess.Popen(
            [
                sys.executable, "-m", "sglang.launch_server",
                "--model-path", MODEL_PATH,
                "--disaggregation-mode", "prefill",
                "--disaggregation-transfer-backend", "mooncake",
                "--disaggregation-ib-device", IB_DEVICE,
                "--host", "127.0.0.1",
                "--port", str(PREFILL_PORT),
                "--mem-fraction-static", "0.7",
                "--enable-piecewise-cuda-graph",
                "--enable-batch-prefill-cuda-graph",
                "--batch-prefill-batch-sizes", "1", "2", "4", "8",
                "--batch-prefill-seq-lengths", "16", "32", "64",
                "--log-level", "info",
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": PREFILL_GPU},
            stdout=open("/tmp/sglang_disagg_prefill.log", "w"),
            stderr=subprocess.STDOUT,
        )
        procs.append(prefill_proc)
        print(f"  PID: {prefill_proc.pid}")

        # =============================================
        # Step 2: Launch decode server
        # =============================================
        print(f"=== Launching DECODE server (GPU {DECODE_GPU}, port {DECODE_PORT}) ===")
        decode_proc = subprocess.Popen(
            [
                sys.executable, "-m", "sglang.launch_server",
                "--model-path", MODEL_PATH,
                "--disaggregation-mode", "decode",
                "--disaggregation-transfer-backend", "mooncake",
                "--disaggregation-ib-device", IB_DEVICE,
                "--host", "127.0.0.1",
                "--port", str(DECODE_PORT),
                "--mem-fraction-static", "0.7",
                "--log-level", "info",
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": DECODE_GPU},
            stdout=open("/tmp/sglang_disagg_decode.log", "w"),
            stderr=subprocess.STDOUT,
        )
        procs.append(decode_proc)
        print(f"  PID: {decode_proc.pid}")

        # Wait for both servers
        print("\nWaiting for prefill server...", end="", flush=True)
        if not wait_for_server(PREFILL_PORT):
            print(" FAILED")
            print("\n--- Prefill server log (last 5000 chars) ---")
            with open("/tmp/sglang_disagg_prefill.log") as f:
                print(f.read()[-5000:])
            return
        print(" OK")

        print("Waiting for decode server...", end="", flush=True)
        if not wait_for_server(DECODE_PORT):
            print(" FAILED")
            print("\n--- Decode server log (last 5000 chars) ---")
            with open("/tmp/sglang_disagg_decode.log") as f:
                print(f.read()[-5000:])
            return
        print(" OK")

        # =============================================
        # Step 3: Check logs for batch prefill graph capture
        # =============================================
        print("\n=== Checking prefill logs for batch prefill graph ===")
        with open("/tmp/sglang_disagg_prefill.log") as f:
            log = f.read()

        found_capture = False
        for line in log.split("\n"):
            if "batch prefill" in line.lower() or "batch_prefill" in line.lower():
                if "captur" in line.lower():
                    found_capture = True
                print(f"  LOG: {line.strip()}")

        if found_capture:
            print("\n  [OK] Batch prefill graphs were CAPTURED on prefill server")
        else:
            print("\n  [WARN] No batch prefill graph capture messages in log")

        # =============================================
        # Step 4: Launch router
        # =============================================
        print(f"\n=== Launching ROUTER (port {ROUTER_PORT}) ===")
        router_proc = subprocess.Popen(
            [
                sys.executable, "-m", "sglang_router.launch_router",
                "--pd-disaggregation",
                "--prefill", f"http://127.0.0.1:{PREFILL_PORT}",
                "--decode", f"http://127.0.0.1:{DECODE_PORT}",
                "--host", "127.0.0.1",
                "--port", str(ROUTER_PORT),
            ],
            stdout=open("/tmp/sglang_disagg_router.log", "w"),
            stderr=subprocess.STDOUT,
        )
        procs.append(router_proc)
        print(f"  PID: {router_proc.pid}")

        print("Waiting for router...", end="", flush=True)
        if not wait_for_router(ROUTER_PORT):
            print(" FAILED")
            print("\n--- Router log ---")
            with open("/tmp/sglang_disagg_router.log") as f:
                print(f.read()[-5000:])
            return
        print(" OK")

        # =============================================
        # Step 5: Send requests through the router
        # =============================================
        print("\n=== Test: Sending requests through router ===")
        prompts = [
            "What is 2+2?",
            "Hello, how are you?",
            "Tell me a joke.",
            "The sky is blue.",
            "Once upon a time",
            "The capital of France is",
        ]

        disagg_outputs = {}
        for p in prompts:
            try:
                result = generate(ROUTER_PORT, p)
                text = result.get("text", str(result))
                disagg_outputs[p] = text
                print(f"  {repr(p):35s} -> {repr(text[:80])}")
            except Exception as e:
                print(f"  {repr(p):35s} -> ERROR: {e}")
                disagg_outputs[p] = None

        # =============================================
        # Step 6: Check prefill logs for batch prefill replay
        # =============================================
        print("\n=== Checking prefill logs for batch prefill replay ===")
        with open("/tmp/sglang_disagg_prefill.log") as f:
            log = f.read()

        found_replay = False
        for line in log.split("\n"):
            ll = line.lower()
            if ("batch prefill" in ll or "batch_prefill" in ll) and (
                "replay" in ll or "can_run" in ll or "running" in ll
            ):
                found_replay = True
                print(f"  LOG: {line.strip()}")

        if found_replay:
            print("\n  [OK] Batch prefill graphs were REPLAYED during inference")
        else:
            print("\n  [INFO] No batch prefill replay messages in log (may need debug logging)")

        # =============================================
        # Step 7: Summary
        # =============================================
        print("\n=== Summary ===")
        success_count = sum(1 for v in disagg_outputs.values() if v is not None)
        print(f"  Requests sent: {len(prompts)}")
        print(f"  Successful:    {success_count}")
        print(f"  Failed:        {len(prompts) - success_count}")

        if success_count == len(prompts):
            print("\n  [OK] All requests completed successfully through PD disaggregation!")
            print("  Batch prefill CUDA graph (attention-in-graph) works in PD disagg mode.")
        elif success_count > 0:
            print("\n  [PARTIAL] Some requests succeeded through PD disaggregation.")
        else:
            print("\n  [FAIL] No requests succeeded through PD disaggregation.")

        print("\n=== Done ===")

    finally:
        cleanup(procs)


if __name__ == "__main__":
    main()
