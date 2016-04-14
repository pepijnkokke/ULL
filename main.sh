#!/bin/bash

ALPHAS=${ALPHAS:-"1 10 100 1000 3000 5000 10000"}
P_HASHES=${P_HASHES:-"0.2 0.5 0.8"}
NS=${NS:-"100 250 500 750 1000 2500 5000 7500 10000"}
OUT_DIR=${DIR:-"model"}

if ([[ $1 = 'phono' ]] || [[ $1 == 'text' ]])
then
    DATASET=$1
else
    echo 'Please specify the dataset (phono or text)'
    exit -1
fi

for P_HASH in $P_HASHES; do
    for ALPHA in $ALPHAS; do
        ./segment
          --out_dir $DIR
          --train_path data/br-$DATASET-train-raw.txt
          -a $ALPHA
          -p $P_HASH
          -n $NS
    done &
done
