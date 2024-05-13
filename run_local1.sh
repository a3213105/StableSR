#!/bin/bash
ENV_PATH=/root/miniconda3/envs/stablesr
CORES=`lscpu | grep "per socket" | awk {'print $4'}`
export OMP_NUM_THREADS=${CORES}
export TSAN_OPTIONS='ignore_noninstrumented_modules=1'
export MKL_VERBOSE=0
export KMP_BLOCKTIME=10
# export KMP_AFFINITY=granularity=fine,compact,1,0
#export LD_PRELOAD=$ENV_PATH/lib/libjemalloc.so
#export LD_PRELOAD=$ENV_PATH/lib/libtcmalloc.so
export LD_PRELOAD=$ENV_PATH/lib/libiomp5.so:${LD_PRELOAD}
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export no_proxy="localhost, 127.0.0.1, ::1, 127.0.0.1:7860, 127.0.0.1:7861, 127.0.0.1:7862"

function check_log() {
    CORES=$1
    STEPS=$2
    CMD=$3
    rm -rf /tmp/perf.log 2>/dev/null
    for((c=0;c<${CORES};c+=${STEPS}))
    do
        cat /tmp/${c}.log | grep "##### total" >> /tmp/perf.log
    done
    echo "PERF ${CMD} ${STEPS}"
    cat /tmp/perf.log
    cat /tmp/perf.log | awk 'BEGIN {sum=0;count=0} {sum+=$4;count+=1} END {print int(sum/count*100)/100}' 2>/dev/null
}

function do_instance() {
    CORES=$1
    STEPS=$2
    CMD=$3
    for((c=0;c<${CORES};c+=${STEPS}))
    do
        c0=$((${c}+${CORES}))
        c1=$((${c0}+${STEPS}-1))
        c2=$((${c0}+${CORES}+${CORES}))
        c3=$((${c1}+${CORES}+${CORES}))
        # echo "${c0}-${c1},${c2}-${c3}"
        # echo "        OMP_NUM_THREADS=${STEPS} HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 \
        # numactl --physcpubind=${c0}-${c1},${c2}-${c3} python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas.py \
        # --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt stablesr_000117.ckpt \
        # --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img ./input --outdir ./output/ --ddpm_steps $step --dec_w 0.5 \
        # --colorfix_type adain --upscale 2 --loop 10 ${CMD} > /tmp/${c}.log 2>&1 &"
        OMP_NUM_THREADS=${STEPS} HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 \
        numactl --physcpubind=${c0}-${c1},${c2}-${c3} python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas.py \
        --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt stablesr_000117.ckpt \
        --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img ./input --outdir ./output/ --ddpm_steps $step --dec_w 0.5 \
        --colorfix_type adain --upscale 2 --loop 3 ${CMD} > /tmp/${c}.log 2>&1 &
    done
    wait
}

step=10
# cmds=("--bf16" "--bf16 --ipex1" "--bf16 --ipex1 --ipex2")
# cmds=("--bf16 --ipex1" "--bf16 --ipex1 --ipex2")
cmds=("--bf16 --ipex1")
echo "${cmds[@]}"
for cmd in "${cmds[@]}"
do
    do_instance ${CORES} ${OMP_NUM_THREADS} "${cmd}"
    check_log ${CORES} ${OMP_NUM_THREADS} "${cmd}"
done
