#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

dataset=fewcomm 
dataset_mode=IO
N=5
K=1
python -u main.py --multi_margin --use_proto_as_neg --model MTNet --dataset $dataset --dataset_mode $dataset_mode --mode $mode --trainN $N --N $N --K $K --Q 1 --trainable_margin_init 6.5

K=5
python -u main.py --multi_margin --use_proto_as_neg --model MTNet --dataset $dataset --dataset_mode $dataset_mode --mode $mode --trainN $N --N $N --K $K --Q 1 --trainable_margin_init 7.3

N=10
K=1
python -u main.py --multi_margin --use_proto_as_neg --model MTNet --dataset $dataset --dataset_mode $dataset_mode --mode $mode --trainN $N --N $N --K $K --Q 1 --trainable_margin_init 6.1

N=10
K=5
python -u main.py --multi_margin --use_proto_as_neg --model MTNet --dataset $dataset --dataset_mode $dataset_mode --mode $mode --trainN $N --N $N --K $K --Q 1 --trainable_margin_init 6.4

endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 
