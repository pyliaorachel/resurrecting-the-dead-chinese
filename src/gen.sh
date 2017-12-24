#!/bin/bash

sl=50
bs=32
ed=256
hd=256
lr=0.0001
dr=0.2
ep=30

f="sl$sl-bs$bs-ed$ed-hd$hd-lr$lr-dr$dr-ep$ep"

python3 -m generate_text.gen ../output/model/$f/corpus.bin ../output/model/$f/model.bin --n-sent 10
