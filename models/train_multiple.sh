#!/bin/bash

LOSS=("ivhd" "mds")
EXAMPLES=("full" "4000ex")
N_GRAPHS=(1 100)
TYPE=("one_graph" "batched")
METRIC=("binary" "cosine" "euclidean")
NN=(2 5 5 5 10 15 100)
RN=(1 1 2 0 0 0 0)

for (( l = 0; l < ${#LOSS[@]}; l += 1 )); do
  for (( e = 0; e < ${#EXAMPLES[@]}; e += 1 )); do
    for (( m = 0; m < ${#METRIC[@]}; m += 1 )); do
      for (( n = 0; n < ${#NN[@]}; n += 1 )); do
        if [[ ${METRIC[m]} == "binary" && ${RN[n]} -eq 0 ]]; then
          continue
        elif [[ (${METRIC[m]} == "cosine" || ${METRIC[m]} == "euclidean") && ${RN[n]} -ne 0 ]]; then
          continue
        fi

        python3 train.py \
          --data "../data/mnist_784/${N_GRAPHS[e]}g_${EXAMPLES[e]}_${METRIC[m]}_${NN[n]}nn_${RN[n]}rn.pkl.lz4" \
          --output_path "${LOSS[l]}/${TYPE[e]}/${METRIC[m]}/${NN[n]}nn_${RN[n]}rn" \
          --batch_size 8 \
          --epochs 50 \
          --hidden_dim 32 \
          --num_layers 5 \
          --lr 0.0003 \
          --loss ${LOSS[l]}
      done
    done
  done
done
