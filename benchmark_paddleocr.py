"""Benchmark PaddleOCR latency on 1024x1024 images (CPU & GPU)."""
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

NUM_WARMUP = 3
NUM_RUNS = 10
IMG_SIZE = (1024, 1024)


def make_test_image() -> np.ndarray:
    """Create a 1024x1024 image with some text for OCR to process."""
    img = Image.new("RGB", IMG_SIZE, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except IOError:
        font = ImageFont.load_default()
    text_lines = [
        "Hello World",
        "PaddleOCR Benchmark",
        "1024 x 1024 Test Image",
        "ABCDEFGHIJKLMNOP",
        "The quick brown fox",
        "jumps over the lazy dog",
    ]
    y = 100
    for line in text_lines:
        draw.text((80, y), line, fill=(0, 0, 0), font=font)
        y += 120
    return np.array(img)


def check_gpu_available() -> bool:
    """Check whether PaddlePaddle can see a GPU."""
    try:
        import paddle
        return paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        return False


def benchmark_device(device: str, use_gpu: bool, img: np.ndarray):
    """Run init + warmup + timed benchmark on the given device."""
    print(f"\n{'='*50}")
    print(f"  Device: {device}")
    print(f"{'='*50}")

    print(f"Initializing PaddleOCR ({device}) ...")
    t0 = time.perf_counter()
    ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=use_gpu, show_log=False)
    init_time = time.perf_counter() - t0
    print(f"Init time: {init_time:.3f}s")

    # Warmup
    print(f"Warming up ({NUM_WARMUP} runs) ...")
    for _ in range(NUM_WARMUP):
        ocr.ocr(img, cls=False)

    # Benchmark
    latencies = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        result = ocr.ocr(img, cls=False)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

    if result and result[0]:
        texts = [r[1][0] for r in result[0]]
        print(f"Recognized texts: {texts}")

    latencies = np.array(latencies)
    print(f"\n--- [{device}] Latency over {NUM_RUNS} runs (seconds) ---")
    print(f"  Mean:   {latencies.mean():.4f}")
    print(f"  Median: {np.median(latencies):.4f}")
    print(f"  Std:    {latencies.std():.4f}")
    print(f"  Min:    {latencies.min():.4f}")
    print(f"  Max:    {latencies.max():.4f}")
    print(f"  All:    {[f'{x:.4f}' for x in latencies]}")
    return latencies


def main():
    print(f"Image size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"Warmup runs: {NUM_WARMUP}, Benchmark runs: {NUM_RUNS}")

    img = make_test_image()
    print(f"Test image shape: {img.shape}, dtype: {img.dtype}")

    gpu_available = check_gpu_available()
    if gpu_available:
        import paddle
        gpu_name = "GPU"
        try:
            gpu_name = f"GPU ({paddle.device.cuda.get_device_name(0)})"
        except Exception:
            pass
        print(f"\nGPU detected: {gpu_name}")
    else:
        print("\nNo GPU detected — will only benchmark CPU.")

    # --- CPU benchmark ---
    cpu_lat = benchmark_device("CPU", use_gpu=False, img=img)

    # --- GPU benchmark ---
    gpu_lat = None
    if gpu_available:
        gpu_lat = benchmark_device(gpu_name, use_gpu=True, img=img)

    # --- Summary ---
    print(f"\n{'='*50}")
    print("  Summary")
    print(f"{'='*50}")
    print(f"  CPU  mean latency: {cpu_lat.mean():.4f}s  (median {np.median(cpu_lat):.4f}s)")
    if gpu_lat is not None:
        print(f"  GPU  mean latency: {gpu_lat.mean():.4f}s  (median {np.median(gpu_lat):.4f}s)")
        speedup = cpu_lat.mean() / gpu_lat.mean()
        print(f"  GPU speedup:       {speedup:.2f}x")


if __name__ == "__main__":
    main()
