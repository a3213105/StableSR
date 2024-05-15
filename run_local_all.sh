#!/bin/bash
ENV_PATH=/root/miniconda3/envs/stablesr

export TSAN_OPTIONS='ignore_noninstrumented_modules=1'
export MKL_VERBOSE=0
export KMP_BLOCKTIME=10
# export KMP_AFFINITY=granularity=fine,compact,1,0
#export LD_PRELOAD=$ENV_PATH/lib/libjemalloc.so
#export LD_PRELOAD=$ENV_PATH/lib/libtcmalloc.so
export LD_PRELOAD=$ENV_PATH/lib/libiomp5.so:${LD_PRELOAD}
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export no_proxy="localhost, 127.0.0.1, ::1, 127.0.0.1:7860, 127.0.0.1:7861, 127.0.0.1:7862"

#for cc in 48 24 16 12 8 6 4
for cc in 48
do
    export OMP_NUM_THREADS=${cc} 
    bash ./run_local2.sh
done
