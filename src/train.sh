#!/bin/bash

sl=50
bs=32
ed=256
hd=256
lr=0.0001
dr=0.2
ep=30

t=$(date "+%Y-%m-%d_%H-%M-%S")
f="sl$sl-bs$bs-ed$ed-hd$hd-lr$lr-dr$dr-ep$ep"

if [ ! -d "../output/model/$f" ]; then
    mkdir "../output/model/$f"
fi

python3 -u -m train.train ../corpus/corpus.txt --output ../output/model/$f/model.bin --output-c ../output/model/$f/corpus.bin --seq-length $sl --batch-size $bs --embedding-dim $ed --hidden-dim $hd --lr $lr --dropout $dr --epochs $ep > ../output/log/$t.log
