#!/bin/bash
# decompress evaluation outputs and training logs
for f in results/*.csv.gz runs/*/episodes.csv.gz; do gunzip -kf "$f"; done
