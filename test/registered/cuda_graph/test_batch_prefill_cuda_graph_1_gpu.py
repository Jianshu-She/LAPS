"""
Test suite for batch prefill CUDA graph functionality.

Tests verify that:
1. Batch prefill CUDA graph is correctly captured for multiple batch_size x seq_len combinations
2. Multiple short prefill requests are batched together correctly
3. Padding works correctly when sequences have different lengths
4. Output correctness matches non-CUDA-graph execution
"""

import unittest

import torch

from sglang import Engine
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
)

# CI Registration - Small 1-GPU tests (24GB GPU sufficient)
register_cuda_ci(est_time=300, suite="stage-b-test-large-1-gpu")


class TestBatchPrefillCudaGraphCorrectness(CustomTestCase):
    """Test batch prefill CUDA graph correctness compared to non-CUDA-graph execution."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-piecewise-cuda-graph",
                "--enable-batch-prefill-cuda-graph",
                "--batch-prefill-max-seq-len",
                "256",
                "--batch-prefill-batch-sizes",
                "1",
                "2",
                "4",
                "8",
                "--batch-prefill-seq-lengths",
                "16",
                "32",
                "64",
                "128",
                "256",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        """Test MMLU accuracy with batch prefill CUDA graph."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)


class TestBatchPrefillCudaGraphEngine(CustomTestCase):
    """Test batch prefill CUDA graph using the Engine API directly."""

    def test_batch_prefill_same_length(self):
        """Test batch prefill with 4 requests of same length (16 tokens each)."""
        model_path = DEFAULT_MODEL_NAME_FOR_TEST

        # Engine with batch prefill CUDA graph
        engine_with_graph = Engine(
            model_path=model_path,
            enable_piecewise_cuda_graph=True,
            enable_batch_prefill_cuda_graph=True,
            batch_prefill_max_seq_len=256,
            batch_prefill_batch_sizes=[1, 2, 4, 8],
            batch_prefill_seq_lengths=[16, 32, 64, 128, 256],
        )

        # Engine without batch prefill CUDA graph
        engine_without_graph = Engine(
            model_path=model_path,
            enable_piecewise_cuda_graph=True,
            enable_batch_prefill_cuda_graph=False,
        )

        # Short prompts (around 16 tokens each)
        prompts = [
            "Hello, how are you today?",
            "What is the capital of France?",
            "Tell me a short joke.",
            "Explain machine learning briefly.",
        ]

        # Generate with both engines
        outputs_with_graph = engine_with_graph.generate(
            prompts, sampling_params={"max_new_tokens": 16, "temperature": 0}
        )
        outputs_without_graph = engine_without_graph.generate(
            prompts, sampling_params={"max_new_tokens": 16, "temperature": 0}
        )

        engine_with_graph.shutdown()
        engine_without_graph.shutdown()

        # Verify outputs match
        for i, (out_graph, out_no_graph) in enumerate(
            zip(outputs_with_graph, outputs_without_graph)
        ):
            self.assertEqual(
                out_graph["text"],
                out_no_graph["text"],
                f"Output mismatch for prompt {i}: {prompts[i]}",
            )

    def test_batch_prefill_different_lengths(self):
        """Test batch prefill with requests of different lengths (with padding)."""
        model_path = DEFAULT_MODEL_NAME_FOR_TEST

        # Engine with batch prefill CUDA graph
        engine_with_graph = Engine(
            model_path=model_path,
            enable_piecewise_cuda_graph=True,
            enable_batch_prefill_cuda_graph=True,
            batch_prefill_max_seq_len=256,
            batch_prefill_batch_sizes=[1, 2, 4, 8],
            batch_prefill_seq_lengths=[16, 32, 64, 128, 256],
        )

        # Engine without batch prefill CUDA graph
        engine_without_graph = Engine(
            model_path=model_path,
            enable_piecewise_cuda_graph=True,
            enable_batch_prefill_cuda_graph=False,
        )

        # Prompts with different lengths
        prompts = [
            "Hi!",  # Very short
            "What is the meaning of life?",  # Medium
            "Can you explain the theory of relativity in simple terms?",  # Longer
        ]

        # Generate with both engines
        outputs_with_graph = engine_with_graph.generate(
            prompts, sampling_params={"max_new_tokens": 16, "temperature": 0}
        )
        outputs_without_graph = engine_without_graph.generate(
            prompts, sampling_params={"max_new_tokens": 16, "temperature": 0}
        )

        engine_with_graph.shutdown()
        engine_without_graph.shutdown()

        # Verify outputs match
        for i, (out_graph, out_no_graph) in enumerate(
            zip(outputs_with_graph, outputs_without_graph)
        ):
            self.assertEqual(
                out_graph["text"],
                out_no_graph["text"],
                f"Output mismatch for prompt {i}: {prompts[i]}",
            )


class TestBatchPrefillCudaGraphEdgeCases(CustomTestCase):
    """Test edge cases for batch prefill CUDA graph."""

    def test_single_request(self):
        """Test that single request still works with batch prefill enabled."""
        model_path = DEFAULT_MODEL_NAME_FOR_TEST

        engine = Engine(
            model_path=model_path,
            enable_piecewise_cuda_graph=True,
            enable_batch_prefill_cuda_graph=True,
            batch_prefill_max_seq_len=256,
        )

        prompt = "Hello, world!"
        output = engine.generate(
            [prompt], sampling_params={"max_new_tokens": 16, "temperature": 0}
        )

        engine.shutdown()

        self.assertIsNotNone(output[0]["text"])
        self.assertGreater(len(output[0]["text"]), 0)

    def test_sequence_exceeds_max_len(self):
        """Test that sequences exceeding max_seq_len fall back to non-graph execution."""
        model_path = DEFAULT_MODEL_NAME_FOR_TEST

        engine = Engine(
            model_path=model_path,
            enable_piecewise_cuda_graph=True,
            enable_batch_prefill_cuda_graph=True,
            batch_prefill_max_seq_len=32,  # Very small max
        )

        # Long prompt that exceeds max_seq_len
        long_prompt = "This is a very long prompt. " * 20  # ~160 tokens

        output = engine.generate(
            [long_prompt], sampling_params={"max_new_tokens": 16, "temperature": 0}
        )

        engine.shutdown()

        # Should still work, just not use batch prefill graph
        self.assertIsNotNone(output[0]["text"])
        self.assertGreater(len(output[0]["text"]), 0)


if __name__ == "__main__":
    unittest.main()
