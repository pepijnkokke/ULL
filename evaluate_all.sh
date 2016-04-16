#!/bin/bash

ALPHAS=${ALPHAS:-"1 10 100 1000 3000 5000 10000"}
P_HASHES=${P_HASHES:-"0.2 0.5 0.8"}
NS=${NS:-"1 3 5 8 10 25 50 75 100 250 500 750 1000 2500 5000 7500 10000"}
OUT_DIR=${OUT_DIR:-"model"}

if ([[ $1 = 'phono' ]] || [[ $1 == 'text' ]])
then
    DATASET=$1
else
    echo 'Please specify the dataset (phono or text)'
    exit -1
fi

for P_HASH in $P_HASHES; do
    for ALPHA in $ALPHAS; do
        for N in $NS; do
            echo "${OUT_DIR}/${N}_${ALPHA}_${P_HASH}.txt"
            python evaluate.py \
                   "data/br-${DATASET}-train-raw.txt" \
                   "data/br-${DATASET}-train.txt" \
                   "${OUT_DIR}/${N}_${ALPHA}_${P_HASH}.txt"
        done
    done
done
