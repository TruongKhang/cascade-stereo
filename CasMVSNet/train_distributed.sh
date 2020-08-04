#!/usr/bin/env bash
MVS_TRAINING="/dev/dtu_dataset/train/"

LOG_DIR=$2
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python -m torch.distributed.launch --nproc_per_node=$1 train.py --logdir $LOG_DIR --dataset=dtu_yao --batch_size=4 --trainpath=$MVS_TRAINING \
                --trainlist lists/dtu/train.txt --testlist lists/dtu/val.txt --numdepth=192 ${@:3} | tee -a $LOG_DIR/log.txt
