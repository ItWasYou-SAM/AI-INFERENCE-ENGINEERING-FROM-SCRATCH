#!/usr/bin/env bash
# Lab 6.8 — print the PTX and SASS of one kernel from kernels.cu.
#
#   ./inspect.sh saxpy            PTX and SASS for saxpy
#   ./inspect.sh saxpy4 sass      SASS only
#   ./inspect.sh all counts       an opcode count for every kernel
#
# Needs nvcc and cuobjdump (both ship with the CUDA toolkit) on PATH.
# ARCH defaults to sm_86 (RTX 30xx); use sm_75 for a T4, sm_89 for RTX 40xx, sm_90 for H100.
set -euo pipefail
cd "$(dirname "$0")"
ARCH=${ARCH:-sm_86}
KERNEL=${1:-saxpy}
WHAT=${2:-both}
NVCC_FLAGS=${NVCC_FLAGS:--O2}

mkdir -p build
nvcc $NVCC_FLAGS -arch=$ARCH -ptx kernels.cu -o build/kernels.ptx
nvcc $NVCC_FLAGS -arch=$ARCH -cubin kernels.cu -o build/kernels.cubin

ptx_of() {   # the .entry block for one kernel
    awk -v k="$1" '$0 ~ "\\.entry "k"\\(" {on=1} on {print} on && /^}/ {exit}' build/kernels.ptx
}
sass_of() {  # instructions only, without the encoding comments
    cuobjdump -sass -fun "$1" build/kernels.cubin | grep -E '^\s+/\*[0-9a-f]{4}\*/' | grep -v 'NOP;' | sed -E 's#/\* 0x[0-9a-f]+ \*/##; s/ +;/;/; s/\s+$//'
}

if [ "$KERNEL" = all ]; then
    for k in $(grep -oE '__global__ void [a-z0-9_]+' kernels.cu | awk '{print $3}'); do
        printf '%-22s' "$k"
        sass_of "$k" | sed -E 's#^\s+/\*[0-9a-f]+\*/\s+##; s/^@!?U?P[0-9T]+ //' | awk '{print $1}' | sed 's/;//' \
            | sort | uniq -c | sort -rn | awk '{printf " %s×%s", $2, $1} END {print ""}'
    done
    exit
fi
if [ "$WHAT" != sass ]; then echo "=== PTX: $KERNEL"; ptx_of "$KERNEL"; echo; fi
if [ "$WHAT" != ptx ];  then echo "=== SASS ($ARCH): $KERNEL"; sass_of "$KERNEL"; fi
