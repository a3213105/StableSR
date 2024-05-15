
#ENV_PATH=/root/miniconda3/envs/sd_pt21
ENV_PATH=/root/miniconda3/envs/stablesr
#ENV_PATH=/root/miniconda3/envs/ipex21_dev

export TSAN_OPTIONS='ignore_noninstrumented_modules=1'
export OMP_NUM_THREADS=48
export MKL_VERBOSE=0
export KMP_BLOCKTIME=10
#export KMP_AFFINITY=granularity=fine,compact,1,0
#export LD_PRELOAD=$ENV_PATH/lib/libjemalloc.so
#export LD_PRELOAD=$ENV_PATH/lib/libtcmalloc.so
export LD_PRELOAD=$ENV_PATH/lib/libiomp5.so:${LD_PRELOAD}
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
#export PYTORCH_TENSOREXPR=0
export no_proxy="localhost, 127.0.0.1, ::1, 127.0.0.1:7860, 127.0.0.1:7861, 127.0.0.1:7862"

#systemctl stop firewalld.service
#numactl --localalloc --physcpubind=60-119 python3 app.py

step=10

#HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 numactl -C 48-95 python  scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt stablesr_000117.ckpt --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img ./input --outdir ./output/ --ddpm_steps $step --dec_w 0.5 --colorfix_type adain --loop 3 

#echo "run torch bf16"


#HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 numactl -C 48-95 python  scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt stablesr_000117.ckpt --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img ./input --outdir ./output/ --ddpm_steps $step --dec_w 0.5 --colorfix_type adain --bf16 --loop 3

# echo "run fp32 + ipex"


# HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 numactl -C 48-95 python  scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt stablesr_000117.ckpt --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img ./input --outdir ./output/ --ddpm_steps $step --dec_w 0.5 --colorfix_type adain  --loop 3 --ipex


echo "run bf16 + ipex"

HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 \
numactl --physcpubind=48-95,144-191 python  scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas.py \
--config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt stablesr_000117.ckpt \
--vqgan_ckpt vqgan_cfw_00011.ckpt --init-img ./input --outdir ./output/ --ddpm_steps $step \
--dec_w 0.5 --colorfix_type adain --upscale 2 --loop 3 --bf16 #--ipex1

# OMP_NUM_THREADS=48 HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 \
# numactl --physcpubind=48-95,144-191 python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas.py \
# --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt stablesr_000117.ckpt \
# --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img ./input --outdir ./output/ --ddpm_steps 10 \
# --dec_w 0.5 --colorfix_type adain --upscale 2 --loop 10 --bf16 > /tmp/0.log 2>&1 &
# # HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 numactl -C 48-79 python  scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt stablesr_000117.ckpt --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img ./input --outdir ./output/ --ddpm_steps $step --dec_w 0.5 --colorfix_type adain --bf16 --loop 3 
# wait
# cat /tmp/0.log | grep "##### total" | awk 'BEGIN {sum=0;count=0} {sum+=$4;count+=1} END {print int(sum/count*100)/100}' 2>/dev/null
