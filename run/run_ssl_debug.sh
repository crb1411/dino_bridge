# 
# conda activate /mnt/unicom/conda_env/pt26
WORK_DIR=/mnt/seek/ssl/dinov3
cd $WORK_DIR

# 显式指定 rendezvous 到本机 127.0.0.1:29505
export ASCEND_RT_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=40028


# export PYTHONPATH=.

# 创建日志目录（如果需要）
# output_dir 
OUTPUTDIR=/mnt/data/train_ssl/imagenet_1k/dino_neg
LOG_DIR="${OUTPUTDIR}/logs_train"
mkdir -p $LOG_DIR

# 获取当前时间戳用于日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 设置主机名环境变量
export HOSTNAME=$(hostname)

# 在后台运行训练任务，并将输出重定向到日志文件
NNODES=1
NODE_RANK=0
GPUS=1
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}_node${NNODES}_${NODE_RANK}.log"

nohup torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m dinov3.new_train.train.train_img \
    --config-file /mnt/data/train_ssl/run/dinov3_vitlarge_pretrain.yaml \
    --checkpoint_dir /mnt/data/train_ssl/imagenet_1k/dino_neg/logs_train/log_20260219_1504/ckpt/170999 \
    --output-dir ${LOG_DIR} > $LOG_FILE 2>&1 &
    # --checkpoint_dir /mnt/data/train/crb/train_out/train_imagenet_1k_126/logs_out/log_20260126_2216/ckpt/31999 \
    # /mnt/work/ckpt/1270499
# 显示后台任务信息
echo "训练任务已在后台启动，PID: $!"
echo "日志文件: $LOG_FILE"
echo "可以使用以下命令查看日志: tail -f $LOG_FILE"
echo "终止所有torchrun进程: pkill -f train_mul"