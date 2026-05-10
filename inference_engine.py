"""
Scalable Inference Engine
Supports batched inference, async generation queues, and ONNX export.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor


class BatchedInferenceEngine:
    """
    High-throughput batched inference engine.
    Queues individual requests and processes them in optimal batches.

    Usage:
        engine = BatchedInferenceEngine(pipeline, max_batch=32)
        engine.start()
        future = engine.submit(class_label=0)
        image = future.result()
        engine.stop()
    """

    def __init__(
        self,
        pipeline,
        max_batch_size: int = 32,
        max_wait_ms: float = 50.0,
        num_workers: int = 1,
    ):
        self.pipeline = pipeline
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.num_workers = num_workers
        self._request_queue: Queue = Queue()
        self._workers: List[Thread] = []
        self._running = False
        self.stats = {"total_requests": 0, "total_batches": 0, "avg_batch_size": 0.0}

    def start(self):
        """Start background worker threads."""
        self._running = True
        for _ in range(self.num_workers):
            t = Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)
        print(f"✓ Inference engine started ({self.num_workers} workers)")

    def stop(self):
        self._running = False
        for _ in self._workers:
            self._request_queue.put(None)
        for t in self._workers:
            t.join()
        print("✓ Inference engine stopped")

    def submit(
        self,
        class_label: Optional[int] = None,
        seed: Optional[int] = None,
        num_steps: int = 50,
    ) -> "InferenceFuture":
        """Submit a single generation request. Returns a future."""
        future = InferenceFuture()
        self._request_queue.put({
            "future": future,
            "class_label": class_label,
            "seed": seed,
            "num_steps": num_steps,
        })
        self.stats["total_requests"] += 1
        return future

    def _worker_loop(self):
        """Collect requests into batches and process them."""
        while self._running:
            batch = []
            deadline = time.time() + self.max_wait_ms / 1000.0

            # Collect up to max_batch_size requests within the deadline
            while len(batch) < self.max_batch_size and time.time() < deadline:
                try:
                    req = self._request_queue.get(timeout=max(0, deadline - time.time()))
                    if req is None:
                        return
                    batch.append(req)
                except Exception:
                    break

            if not batch:
                continue

            self._process_batch(batch)
            self.stats["total_batches"] += 1
            n = len(batch)
            self.stats["avg_batch_size"] = (
                self.stats["avg_batch_size"] * (self.stats["total_batches"] - 1) + n
            ) / self.stats["total_batches"]

    def _process_batch(self, batch: List[dict]):
        """Generate images for a collected batch."""
        n = len(batch)
        labels = None
        if any(r["class_label"] is not None for r in batch):
            labels = torch.tensor(
                [r["class_label"] or 0 for r in batch],
                dtype=torch.long,
                device=self.pipeline.device,
            )

        try:
            images = self.pipeline.generate(
                batch_size=n,
                class_labels=labels,
                num_inference_steps=batch[0]["num_steps"],
                show_progress=False,
            )
            for i, req in enumerate(batch):
                req["future"].set_result(images[i])
        except Exception as e:
            for req in batch:
                req["future"].set_exception(e)


class InferenceFuture:
    """Simple future-like object for async inference results."""

    def __init__(self):
        self._result = None
        self._exception = None
        self._event = asyncio.Event() if asyncio.get_event_loop().is_running() else None
        self._done = False

    def set_result(self, result):
        self._result = result
        self._done = True

    def set_exception(self, exc):
        self._exception = exc
        self._done = True

    def result(self, timeout: float = 60.0) -> torch.Tensor:
        """Block until result is available."""
        deadline = time.time() + timeout
        while not self._done:
            if time.time() > deadline:
                raise TimeoutError("Inference timed out")
            time.sleep(0.001)
        if self._exception:
            raise self._exception
        return self._result


# ─────────────────────────────────────────────
#  ONNX Export
# ─────────────────────────────────────────────

class ONNXExporter:
    """
    Export UNet to ONNX for accelerated inference with ONNX Runtime.
    ONNX models are ~2-3× faster on CPU and compatible with TensorRT.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

    def export(
        self,
        output_path: str,
        image_size: int = 32,
        batch_size: int = 1,
        in_channels: int = 3,
        opset_version: int = 17,
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            output_path:    Path to save .onnx file
            image_size:     Spatial size of input image
            batch_size:     Static batch size (use 'dynamic' for variable)
            in_channels:    Image channels
            opset_version:  ONNX opset version

        Returns:
            Path to exported file
        """
        import torch.onnx

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Dummy inputs
        dummy_x = torch.randn(batch_size, in_channels, image_size, image_size).to(self.device)
        dummy_t = torch.randint(0, 1000, (batch_size,)).to(self.device)

        # Dynamic axes for variable batch / sequence
        dynamic_axes = {
            "x": {0: "batch"},
            "t": {0: "batch"},
            "output": {0: "batch"},
        }

        torch.onnx.export(
            self.model,
            (dummy_x, dummy_t),
            str(output_path),
            input_names=["x", "t"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )
        print(f"✓ Exported ONNX model → {output_path}")

        # Verify
        self._verify(str(output_path))
        return str(output_path)

    def _verify(self, path: str):
        """Verify ONNX model is valid."""
        try:
            import onnx
            model = onnx.load(path)
            onnx.checker.check_model(model)
            print(f"  ✓ ONNX validation passed")
        except ImportError:
            print("  Warning: onnx not installed, skipping verification")

    def benchmark_onnx(self, path: str, n_runs: int = 100, **input_kwargs) -> Dict[str, float]:
        """Benchmark ONNX model vs PyTorch."""
        try:
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime not installed. pip install onnxruntime")
            return {}

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        image_size = input_kwargs.get("image_size", 32)
        dummy_x = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
        dummy_t = np.array([500], dtype=np.int64)

        # Warmup
        for _ in range(10):
            sess.run(None, {"x": dummy_x, "t": dummy_t})

        # Benchmark ONNX
        t0 = time.time()
        for _ in range(n_runs):
            sess.run(None, {"x": dummy_x, "t": dummy_t})
        onnx_ms = (time.time() - t0) / n_runs * 1000

        # Benchmark PyTorch
        x_pt = torch.tensor(dummy_x)
        t_pt = torch.tensor(dummy_t)
        with torch.no_grad():
            for _ in range(10):  # warmup
                self.model(x_pt, t_pt)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                self.model(x_pt, t_pt)
        pt_ms = (time.time() - t0) / n_runs * 1000

        results = {"pytorch_ms": pt_ms, "onnx_ms": onnx_ms, "speedup": pt_ms / onnx_ms}
        print(f"\n  PyTorch: {pt_ms:.2f}ms | ONNX: {onnx_ms:.2f}ms | Speedup: {pt_ms/onnx_ms:.2f}×")
        return results
