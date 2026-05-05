# Task Dependencies

The quick start installs the common miles-diffusion runtime. Some tasks require
extra reward or evaluation dependencies. This page records those task-scoped
dependency sets only.

## OCR Dependencies

Use this section for any task that enables the OCR reward, for example through
`--rm-type ocr` or `--diffusion-reward ocr:...`.

The dependency boundary is the OCR reward implementation:

```text
miles.rollout.rm_hub.ocr
```

The OCR reward depends on PaddleOCR, PaddlePaddle, OpenCV runtime libraries, and
string-distance packages.

Start from the base environment:

```bash
conda activate miles-diffusion
cd /path/to/miles
```

Install the system libraries required by PaddleOCR and OpenCV:

```bash
sudo apt-get update
sudo apt-get install -y libglib2.0-0 libgl1
```

In a root container, use `apt-get` directly if `sudo` is unavailable:

```bash
apt-get update
apt-get install -y libglib2.0-0 libgl1
```

Install the Python dependencies from the pinned `flow_grpo` setup file:

```bash
cd /path/to/miles/flow_grpo
grep -v '^apt-get install ' setup.sh | bash
```

The relevant pins include:

```text
diffusers==0.37.0
peft==0.18.1
bitsandbytes==0.48.0
opencv-python==4.11.0.86
opencv-python-headless==4.10.0.84
opencv-contrib-python==4.11.0.86
paddlepaddle-gpu==2.6.2
paddleocr==2.9.1
python-Levenshtein==0.27.3
levenshtein==0.27.3
rapidfuzz==3.14.3
```

Verify the OCR reward stack:

```bash
cd /path/to/miles
python -c "from paddleocr import PaddleOCR; from Levenshtein import distance; import miles.rollout.rm_hub.ocr; print('OCR deps OK')"
```
