"""Test script to verify batch prefill CUDA graph is being used."""
import requests
import json
import concurrent.futures
import time

SERVER_URL = "http://localhost:30010"

def send_short_request(prompt, max_tokens=5):
    """Send a short prefill request."""
    resp = requests.post(
        f"{SERVER_URL}/generate",
        json={
            "text": prompt,
            "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0},
        },
    )
    return resp.json()

def test_concurrent_short_requests():
    """Send multiple short requests concurrently to trigger batch prefill."""
    prompts = [
        "What is 2+2?",
        "Hello world!",
        "The capital of France is",
        "1, 2, 3, 4,",
    ]

    print("=" * 60)
    print("Test 1: 4 concurrent short requests (should trigger batch prefill)")
    print("=" * 60)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(send_short_request, p) for p in prompts]
        results = [f.result() for f in futures]

    for prompt, result in zip(prompts, results):
        text = result.get("text", "ERROR")
        print(f"  Prompt: {prompt!r:40s} -> {text!r}")

    print()

def test_sequential_requests():
    """Send requests one by one (should use single-sequence path)."""
    prompts = [
        "What is the meaning of life?",
        "Tell me a joke.",
    ]

    print("=" * 60)
    print("Test 2: Sequential requests (single-sequence path)")
    print("=" * 60)

    for prompt in prompts:
        result = send_short_request(prompt)
        text = result.get("text", "ERROR")
        print(f"  Prompt: {prompt!r:40s} -> {text!r}")

    print()

def test_burst_requests():
    """Send a burst of 8 short requests at once."""
    prompts = [
        "A", "B", "C", "D",
        "E", "F", "G", "H",
    ]

    print("=" * 60)
    print("Test 3: Burst of 8 short requests")
    print("=" * 60)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(send_short_request, p, 3) for p in prompts]
        results = [f.result() for f in futures]

    for prompt, result in zip(prompts, results):
        text = result.get("text", "ERROR")
        print(f"  Prompt: {prompt!r:5s} -> {text!r}")

    print()

if __name__ == "__main__":
    # Warm up
    print("Warming up...")
    send_short_request("warm up", 1)
    time.sleep(1)

    test_concurrent_short_requests()
    time.sleep(2)
    test_burst_requests()
    time.sleep(2)
    test_sequential_requests()

    print("Done! Check server logs for '[BatchPrefill]' messages.")
