#!/usr/bin/env bash
set -euo pipefail
STAGING=/staging/t/txu98
DEST="$STAGING/python_packages/reg2rg_torch201_ds0126_v1"
SIF="$STAGING/reg2rg-pytorch2.0.1-cu117.sif"
mkdir -p "$DEST"
export DS_BUILD_OPS=0
apptainer exec --bind "$STAGING:$STAGING" "$SIF" python -m pip install \
 --disable-pip-version-check --no-cache-dir --no-deps --target "$DEST" \
 'pydantic==1.10.13' 'hjson==3.1.0' 'ninja==1.11.1.1' \
 'py-cpuinfo==9.0.0' 'pynvml==11.5.0' 'accelerate==0.25.0'
export PYTHONPATH="$DEST"
apptainer exec --bind "$STAGING:$STAGING" "$SIF" python -m pip install \
 --disable-pip-version-check --no-cache-dir --no-deps --no-build-isolation \
 --target "$DEST" 'deepspeed==0.12.6'
apptainer exec --bind "$STAGING:$STAGING" "$SIF" python -c \
 'import torch,transformers,deepspeed,accelerate,peft,pydantic; print("VERIFIED",torch.__version__,transformers.__version__,deepspeed.__version__,accelerate.__version__,peft.__version__,pydantic.__version__); assert deepspeed.__version__=="0.12.6"'
# Pack the verified overlay so GPU jobs copy one file, not many staging files.
tar -czf "$STAGING/python_packages/reg2rg_torch201_ds0126_v1.tar.gz.partial" -C "$DEST" .
mv "$STAGING/python_packages/reg2rg_torch201_ds0126_v1.tar.gz.partial" "$STAGING/python_packages/reg2rg_torch201_ds0126_v1.tar.gz"
sha256sum "$STAGING/python_packages/reg2rg_torch201_ds0126_v1.tar.gz"
