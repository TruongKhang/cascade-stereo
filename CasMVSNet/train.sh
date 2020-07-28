#!/usr/bin/env bash
MVS_TRAINING="/home/khang/project/dtu_dataset/train/"

LOG_DIR=$1
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python train.py --logdir $LOG_DIR --dataset=dtu_yao --batch_size=2 --trainpath=$MVS_TRAINING --resume \
                --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --numdepth=192 ${@:3} | tee -a $LOG_DIR/log.txt
