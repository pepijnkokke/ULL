#!/bin/bash

ALPHAS=${ALPHAS:-"1 10 100 1000 3000 5000 10000"}
P_HASHES=${P_HASHES:-"0.2 0.5 0.8"}
NS=${NS:-"1 3 5 8 10 25 50 75"}
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
        ./segment --out_dir "$OUT_DIR" --train_path "data/br-$DATASET-train-raw.txt" -a $ALPHA -p $P_HASH -n $NS
    done &
done
