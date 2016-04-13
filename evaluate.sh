#!/bin/bash

ALPHA=${ALPHA:-0.5}
P_HASH=${P_HASH:-0.5}

if ([[ $1 = 'phono' ]] || [[ $1 == 'text' ]]) && ([[ $2 == 'train' ]] || [[ $2 == 'test' ]])
then
    DATASET=$1
    TYPE=$2
else
    echo 'Please specify the dataset (phono or text) and wether to use training or test data'
    exit -1
fi

python2 evaluate.py "data/br-${DATASET}-${TYPE}-raw.txt" "data/br-${DATASET}-${TYPE}.txt" "${DATASET}_${TYPE}_${ALPHA}_${P_HASH}.txt"
