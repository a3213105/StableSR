#!/bin/bash

function check_log() {
    CORES=$1
    CSTEP=$2
    STEP=$3
    CMD=$4
    rm -rf /tmp/perf.log 2>/dev/null
    for((c=0;c<${CORES};c+=${CSTEP}))
    do
        cat /tmp/${c}.log | grep "##### total" >> /tmp/perf.log
    done
    echo "PERF ${CMD} ${CSTEP} ${STEP}"
    cat /tmp/perf.log
    cat /tmp/perf.log | awk 'BEGIN {sum=0;count=0} {sum+=$4;count+=1} END {print int(sum/count*100)/100}' 2>/dev/null
}

function do_instance() {
    CORES=$1
    CSTEP=$2
    STEP=$3
    CMD=$4
    for((c=0;c<${CORES};c+=${CSTEP}))
    do
        #c0=$((${c}+${CORES}))
	c0=${c}
        c1=$((${c0}+${CSTEP}-1))
        c2=$((${c0}+${CORES}+${CORES}))
        c3=$((${c1}+${CORES}+${CORES}))
        echo "${c0}-${c1},${c2}-${c3}"
        OMP_NUM_THREADS=${CSTEP} HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 \
        numactl --physcpubind=${c0}-${c1},${c2}-${c3} python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas.py \
        --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt stablesr_000117.ckpt \
        --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img ./input --outdir ./output/ --ddpm_steps ${STEP} \
        --dec_w 0.5 --colorfix_type adain --upscale 2 --loop 10 ${CMD} > /tmp/${c}.log 2>&1 &
    done
    wait
}
CSTEP=${1}
cmds=("--bf16" "--bf16 --ipex1")
CORES=`lscpu | grep "per socket" | awk {'print $4'}`
echo "${cmds[@]}"
for step in 10
do
    for cmd in "${cmds[@]}"
    do
        # for n in 1 2 3 4 5 6
        for n in 6 8 12 16 24
        do
            new_cmd="--n_samples ${n} ${cmd}"
            do_instance ${CORES} ${CSTEP} ${step} "${new_cmd}"
            check_log ${CORES} ${CSTEP} ${step} "${new_cmd}"
	done
    done
done
