#!/usr/bin/env bash
TESTPATH="/dev/dtu_dataset/train"
TESTLIST="lists/dtu/train.txt"
CKPT_FILE=$1
python test.py --dataset=general_eval --batch_size=1 --testpath=$TESTPATH  --testlist=$TESTLIST --loadckpt $CKPT_FILE ${@:2}
