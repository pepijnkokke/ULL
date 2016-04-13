#!/bin/bash

ALPHA=${ALPHA:-0.5}
P_HASH=${P_HASH:-0.5}

if ([[ $1 = 'phono' ]] || [[ $1 == 'text' ]]) && [[ -n $2 ]]
then
    DATASET=$1
    N=$2
else
    echo 'Please specify the dataset (phono or text) and number of iterations'
    exit -1
fi

./segment --out_path "${DATASET}_train_${ALPHA}_${P_HASH}.txt" --train_path data/br-$DATASET-train-raw.txt -n $N -a $ALPHA -p $P_HASH
