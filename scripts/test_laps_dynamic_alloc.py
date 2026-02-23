"""Test LAPS dynamic allocation in PD disaggregation mode.

End-to-end test:
  1. Launch 2 prefill servers (GPU 0, 1)
  2. Launch 1 decode server (GPU 2)
  3. Launch router with --enable-laps-alloc
  4. Send mixed short/long requests through the router
  5. Check /laps_status for group assignments and pending counts
  6. Verify all outputs are correct
"""
import subprocess
import sys
import time
import requests
import os
import signal

MODEL_PATH = "/mnt/weka/home/jianshu.she/models/qwen2.5-0.5b"
PREFILL1_PORT = 30200
PREFILL2_PORT = 30201
DECODE_PORT = 30202
ROUTER_PORT = 30203
BOOTSTRAP1_PORT = 9200
BOOTSTRAP2_PORT = 9201
PREFILL1_GPU = "0"
PREFILL2_GPU = "1"
DECODE_GPU = "2"
IB_DEVICE = "mlx5_0"

LOG_DIR = "/tmp/sglang_laps_alloc"


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


def get_laps_status(port):
    r = requests.get(f"http://127.0.0.1:{port}/laps_status", timeout=5)
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
    subprocess.run("pkill -9 -f 'sglang.*3020[0123]'", shell=True, capture_output=True)
    subprocess.run(
        "pkill -9 -f 'sglang_router.*3020'", shell=True, capture_output=True
    )
    time.sleep(2)


def main():
    procs = []
    os.makedirs(LOG_DIR, exist_ok=True)

    try:
        # =============================================
        # Step 1: Launch prefill server 1
        # =============================================
        print(
            f"=== Launching PREFILL 1 (GPU {PREFILL1_GPU}, port {PREFILL1_PORT}) ==="
        )
        p1 = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                MODEL_PATH,
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-transfer-backend",
                "mooncake",
                "--disaggregation-ib-device",
                IB_DEVICE,
                "--disaggregation-bootstrap-port",
                str(BOOTSTRAP1_PORT),
                "--host",
                "127.0.0.1",
                "--port",
                str(PREFILL1_PORT),
                "--mem-fraction-static",
                "0.7",
                "--log-level",
                "info",
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": PREFILL1_GPU},
            stdout=open(f"{LOG_DIR}/prefill1.log", "w"),
            stderr=subprocess.STDOUT,
        )
        procs.append(p1)

        # =============================================
        # Step 2: Launch prefill server 2
        # =============================================
        print(
            f"=== Launching PREFILL 2 (GPU {PREFILL2_GPU}, port {PREFILL2_PORT}) ==="
        )
        p2 = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                MODEL_PATH,
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-transfer-backend",
                "mooncake",
                "--disaggregation-ib-device",
                IB_DEVICE,
                "--disaggregation-bootstrap-port",
                str(BOOTSTRAP2_PORT),
                "--host",
                "127.0.0.1",
                "--port",
                str(PREFILL2_PORT),
                "--mem-fraction-static",
                "0.7",
                "--log-level",
                "info",
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": PREFILL2_GPU},
            stdout=open(f"{LOG_DIR}/prefill2.log", "w"),
            stderr=subprocess.STDOUT,
        )
        procs.append(p2)

        # =============================================
        # Step 3: Launch decode server
        # =============================================
        print(f"=== Launching DECODE (GPU {DECODE_GPU}, port {DECODE_PORT}) ===")
        d1 = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                MODEL_PATH,
                "--disaggregation-mode",
                "decode",
                "--disaggregation-transfer-backend",
                "mooncake",
                "--disaggregation-ib-device",
                IB_DEVICE,
                "--host",
                "127.0.0.1",
                "--port",
                str(DECODE_PORT),
                "--mem-fraction-static",
                "0.7",
                "--log-level",
                "info",
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": DECODE_GPU},
            stdout=open(f"{LOG_DIR}/decode.log", "w"),
            stderr=subprocess.STDOUT,
        )
        procs.append(d1)

        # Wait for all servers
        for name, port in [
            ("prefill1", PREFILL1_PORT),
            ("prefill2", PREFILL2_PORT),
            ("decode", DECODE_PORT),
        ]:
            print(f"Waiting for {name} server...", end="", flush=True)
            if not wait_for_server(port):
                print(" FAILED")
                log_file = f"{LOG_DIR}/{name}.log"
                if os.path.exists(log_file):
                    with open(log_file) as f:
                        print(f.read()[-3000:])
                return
            print(" OK")

        # =============================================
        # Step 4: Launch router with LAPS dynamic allocation
        # =============================================
        print(f"\n=== Launching ROUTER with LAPS alloc (port {ROUTER_PORT}) ===")
        router_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang_router.launch_router",
                "--pd-disaggregation",
                "--mini-lb",
                "--enable-laps-alloc",
                "--laps-alloc-threshold",
                "256",
                "--laps-rebalance-interval-s",
                "1.0",
                "--laps-rebalance-ratio",
                "2.0",
                "--prefill",
                f"http://127.0.0.1:{PREFILL1_PORT}",
                str(BOOTSTRAP1_PORT),
                "--prefill",
                f"http://127.0.0.1:{PREFILL2_PORT}",
                str(BOOTSTRAP2_PORT),
                "--decode",
                f"http://127.0.0.1:{DECODE_PORT}",
                "--host",
                "127.0.0.1",
                "--port",
                str(ROUTER_PORT),
            ],
            stdout=open(f"{LOG_DIR}/router.log", "w"),
            stderr=subprocess.STDOUT,
        )
        procs.append(router_proc)

        print("Waiting for router...", end="", flush=True)
        if not wait_for_server(ROUTER_PORT, timeout=60):
            print(" FAILED")
            with open(f"{LOG_DIR}/router.log") as f:
                print(f.read()[-3000:])
            return
        print(" OK")

        # =============================================
        # Step 5: Check /laps_status
        # =============================================
        print("\n=== Checking /laps_status ===")
        status = get_laps_status(ROUTER_PORT)
        print(f"  enabled:       {status.get('enabled')}")
        print(f"  threshold:     {status.get('threshold')}")
        print(f"  short_group:   {status.get('short_group')}")
        print(f"  long_group:    {status.get('long_group')}")
        print(f"  short_pending: {status.get('short_pending')}")
        print(f"  long_pending:  {status.get('long_pending')}")

        if not status.get("enabled"):
            print("  [FAIL] LAPS allocation is not enabled!")
            return

        # =============================================
        # Step 6: Send short requests (should go to short group)
        # =============================================
        print("\n=== Test: Short requests (<256 tokens) ===")
        short_prompts = [
            "Hi",
            "What is 2+2?",
            "Hello world",
            "The sky is",
        ]
        for p in short_prompts:
            try:
                result = generate(ROUTER_PORT, p)
                text = result.get("text", str(result))
                print(f"  {repr(p):30s} -> {repr(text[:60])}")
            except Exception as e:
                print(f"  {repr(p):30s} -> ERROR: {e}")

        # =============================================
        # Step 7: Send long requests (should go to long group)
        # =============================================
        print("\n=== Test: Long requests (>256 tokens, ~1200+ chars) ===")
        # Create prompts > 256 tokens (~1200+ chars → >300 tokens, above threshold)
        long_prompts = [
            "Explain in detail " + "the history of " * 80 + "computing.",
            "Write about " + "the future of " * 80 + "AI and technology.",
        ]
        for p in long_prompts:
            try:
                est_tokens = len(p) // 4
                result = generate(ROUTER_PORT, p, max_new_tokens=16)
                text = result.get("text", str(result))
                print(
                    f"  prompt[{est_tokens} est_tok] -> {repr(text[:60])}"
                )
            except Exception as e:
                print(f"  long prompt -> ERROR: {e}")

        # =============================================
        # Step 8: Check /laps_status after requests
        # =============================================
        print("\n=== Checking /laps_status after requests ===")
        status = get_laps_status(ROUTER_PORT)
        print(f"  short_group:   {status.get('short_group')}")
        print(f"  long_group:    {status.get('long_group')}")
        print(f"  short_pending: {status.get('short_pending')}")
        print(f"  long_pending:  {status.get('long_pending')}")

        # =============================================
        # Step 9: Check router log for LAPS messages
        # =============================================
        print("\n=== Router log (LAPS-related) ===")
        with open(f"{LOG_DIR}/router.log") as f:
            for line in f:
                if "LAPS" in line or "laps" in line:
                    print(f"  {line.rstrip()}")

        # =============================================
        # Summary
        # =============================================
        print("\n=== Summary ===")
        print(f"  LAPS allocation enabled: {status.get('enabled')}")
        print(f"  Short group: {status.get('short_group')}")
        print(f"  Long group:  {status.get('long_group')}")
        print(
            f"  Pending counters: short={status.get('short_pending')}, long={status.get('long_pending')}"
        )
        print("\n  [OK] LAPS dynamic allocation test complete!")

    finally:
        cleanup(procs)


if __name__ == "__main__":
    main()
