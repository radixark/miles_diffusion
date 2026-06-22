ARG SGLANG_IMAGE_TAG=v0.5.12-cu129
FROM lmsysorg/sglang:${SGLANG_IMAGE_TAG} AS sglang

# ======================================== Setup =============================================

WORKDIR /root/

# ======================================== Apt dependencies =============================================

RUN apt update
RUN apt install -y nvtop rsync dnsutils ethtool
# Runtime shared libraries for OpenCV/PaddleOCR.
RUN apt install -y libglib2.0-0 libgl1

# nccl-tests for diagnostics
RUN git clone https://github.com/NVIDIA/nccl-tests.git /tmp/nccl-tests && \
    cd /tmp/nccl-tests && \
    git checkout ae98985f5599617be94042f4aa3637d10014ce89 && \
    make -j$(nproc) CUDA_HOME=/usr/local/cuda && \
    cp /tmp/nccl-tests/build/*_perf /usr/local/bin/ && \
    rm -rf /tmp/nccl-tests

# ====================================== Python dependencies ============================================

# ===== Swap stock upstream sglang for the Rockdu sglang-diffusion fork =====
# miles-diffusion needs multimodal_gen + RL RPCs (release/resume_memory_occupation,
# update_weights_from_tensor) that now ONLY exist on Rockdu/sglang@sglang-diffusion-rollout-test.
# sglang is installed editable at /sgl-workspace/sglang, so check out the pinned
# fork commit in place and re-register editable (--no-deps keeps the base image's
# torch 2.11 + prebuilt sgl-kernel intact). Pinned SHA = reproducible rollouts.
ARG SGLANG_DIFFUSION_REPO=https://github.com/Rockdu/sglang.git
ARG SGLANG_DIFFUSION_BRANCH=sglang-diffusion-rollout-test
ARG SGLANG_DIFFUSION_COMMIT=0372158dd66bc7cb0740c733bd60047db790ec7d
RUN cd /sgl-workspace/sglang && \
    (git remote add rockdu "${SGLANG_DIFFUSION_REPO}" || git remote set-url rockdu "${SGLANG_DIFFUSION_REPO}") && \
    (git fetch --depth=1 rockdu "${SGLANG_DIFFUSION_COMMIT}" || git fetch rockdu "${SGLANG_DIFFUSION_BRANCH}") && \
    git checkout --detach "${SGLANG_DIFFUSION_COMMIT}" && \
    pip install -e python --no-deps
# torch_memory_saver backs release/resume_memory_occupation; pin the validated version.
RUN pip install torch_memory_saver==0.0.9

COPY requirements.txt /tmp/requirements.txt
# Pre-seed a pip-managed PyJWT so requirements upgrades don't trip on the
# debian-installed copy that has no RECORD file (uninstall-no-record-file).
RUN pip install --ignore-installed PyJWT
RUN pip install -r /tmp/requirements.txt

# ====================================== Install main package ============================================

# Install miles_diffusion from a pinned ref (like radixark/miles does for miles),
# not from the local build context. Defaults to main for official releases;
# override with --build-arg MILES_DIFFUSION_COMMIT=<branch|sha>. CI does not rely
# on this baked copy (it re-installs the checked-out PR source), so building from
# a clean ref keeps the image reproducible instead of baking the local tree.
ARG MILES_DIFFUSION_COMMIT=main
RUN git clone https://github.com/radixark/miles_diffusion.git /root/miles_diffusion && \
    cd /root/miles_diffusion && \
    git checkout ${MILES_DIFFUSION_COMMIT} && \
    pip install -e . --no-deps

# =========================================== Pre-download ===============================================

# Pre-download PaddleOCR (en) det+rec+cls models so the OCR-reward actors don't
# race-download them from bcebos at runtime. 
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)"
