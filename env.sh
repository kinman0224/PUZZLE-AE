#!/bin/bash

conda activate ae
source /public/software/profile.d/apps_cuda-11.8.sh
source /public/software/profile.d/compiler_gcc-7.3.1.sh

export RAY_DEDUP_LOGS=0
export MAX_JOBS=64