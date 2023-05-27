#!/bin/bash
device=$1
input_model_file=$2
dataset=$3
model=$4
grad_accum_step=$5
epochs=$6




if  [[ $model = 'gin' || $model = 'gcn' || $model = 'gat' ]]
then
	for runseed in `seq 1 10`; do
  CUDA_VISIBLE_DEVICES=$device  python finetune_property_prediction_graph_only.py --model_name_or_path $input_model_file --dataset $dataset --runseed $runseed  --batch_size=256   --drop_ratio=0.5  --output_model_dir=ckpts --gnn_type $model --epochs $epochs
  done
elif [ $model = 'graphormer' ]
then
	for runseed in `seq 1 10`; do
  CUDA_VISIBLE_DEVICES=$device python finetune_property_prediction_graph_only.py  --dataset $dataset --runseed $runseed  --batch_size 40 --grad_accum_step $grad_accum_step --output_model_dir=ckpts/tem/ --lr 1e-5 --epochs $epochs  --backbone graphormer --arch graphormer_base --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --encoder-layers 12 --encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --encoder-attention-heads 32 --pre-layernorm --num_classes 1 --restore_file_graphormer $input_model_file --transform_in_collator --num_workers 0 --not_load_pretrained_model_output_layer
  done
elif [ $model = 'kvplm' ]
then
  for runseed in `seq 1 10`; do
    CUDA_VISIBLE_DEVICES=$device python finetune_property_prediction_graph_only.py --dataset $dataset --runseed $runseed  --batch_size 40 --grad_accum_step $grad_accum_step --output_model_dir=ckpts/tem/ --lr 1e-5 --epochs $epochs --backbone kvplm --num_workers 0 --init_checkpoint /mnt/data/zhaohaiteng/KV-PLM/save_model/ckpt_KV.pt
  done

elif [ $model = 'grapht5' ]
then
	for runseed in `seq 1 10`; do
  CUDA_VISIBLE_DEVICES=$device python finetune_property_prediction_graph_only.py  --dataset $dataset --runseed $runseed  --batch_size 40 --grad_accum_step $grad_accum_step --output_model_dir=ckpts/tem/ --lr 1e-5 --lr_scale 10 --epochs $epochs  --backbone grapht5 --arch graphormer_base --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --encoder-layers 12 --encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --encoder-attention-heads 32 --pre-layernorm  --transform_in_collator --num_workers 0  --model_name_or_path $input_model_file --unimodel --graphonly_readout cls
  done
fi