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

# sglang-diffusion is merged upstream: track sgl-project/sglang main; SGLANG_DIFFUSION_COMMIT=<sha> pins, "none" follows the branch tip.
ARG SGLANG_DIFFUSION_REPO=https://github.com/sgl-project/sglang.git
ARG SGLANG_DIFFUSION_BRANCH=main
ARG SGLANG_DIFFUSION_COMMIT=none
# SGLANG_BUILD_RUST_EXTS=none skips the gRPC Rust extension: unused here and the base image has no Rust toolchain.
RUN cd /sgl-workspace/sglang && \
    if [ "${SGLANG_DIFFUSION_COMMIT}" = "none" ]; then \
        git fetch --depth=1 "${SGLANG_DIFFUSION_REPO}" "${SGLANG_DIFFUSION_BRANCH}"; \
    else \
        git fetch --depth=1 "${SGLANG_DIFFUSION_REPO}" "${SGLANG_DIFFUSION_COMMIT}"; \
    fi && \
    git checkout --detach FETCH_HEAD && \
    SGLANG_BUILD_RUST_EXTS=none pip install -e python --no-deps
# sgl main's FA3 wrappers need sglang-kernel 0.4.5 (only_qv kwarg); the base image ships 0.4.2.
RUN pip install --no-deps "sglang-kernel==0.4.5+cu129" --extra-index-url https://docs.sglang.io/whl/cu129
# Bake the kernels lockfile like upstream so the FA3 hub loader never resolves a stale snapshot.
RUN cd /sgl-workspace/sglang && \
    kernels lock python && \
    mkdir -p /root/.cache/sglang && \
    mv python/kernels.lock /root/.cache/sglang/
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
