#!/bin/bash
device=$1
model=$2
model_ckpt_name=$3

few_shot_number=$4

prompt_augmentation=$5

ckpt_dir="ckpts/"
out_file_prefix="cache/testing_"
outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
echo $outfile

export dataset_list=( bace hiv muv tox21 toxcast bbbp cyp450 esol lipo freesolv)


if [ $model = 'grapht0' ]
then
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python  finetune_property_prediction_graph_transformer_multitask.py --model_name_or_path $ckpt_dir$model_ckpt_name --tokenizer_name t5-small --dataset "$dataset" --runseed 5 --eval_train --batch_size 16 --grad_accum_step 1 --output_model_dir=ckpts/tem/ --lr 1e-5 --epochs 50 --prompt_policy traversal --graph_transformer_graph_backbone graphormer --arch graphormer_base --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --encoder-layers 12 --encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --encoder-attention-heads 32 --pre-layernorm --num_classes 1 --attention_fasion coattention --transform_in_collator --task_policy traversal --zero_shot --only_test --unimodel --maskt2g --output_result_to_file $outfile --not_retest_tasks_in_result_file

  done

elif [ $model = 'grapht0_base' ]
then
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python  finetune_property_prediction_graph_transformer_multitask.py --model_name_or_path $ckpt_dir$model_ckpt_name --tokenizer_name t5-base --dataset "$dataset" --runseed 5 --eval_train --batch_size 4 --grad_accum_step 1 --output_model_dir=ckpts/tem/ --lr 1e-5 --epochs 50 --prompt_policy traversal --graph_transformer_graph_backbone graphormer --arch graphormer_base --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --encoder-layers 12 --encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --encoder-attention-heads 32 --pre-layernorm --num_classes 1 --attention_fasion coattention --transform_in_collator --task_policy traversal --zero_shot --only_test --unimodel --maskt2g --output_result_to_file $outfile --not_retest_tasks_in_result_file

  done

elif [ $model = 'grapht0_large' ]
then
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python  finetune_property_prediction_graph_transformer_multitask.py --model_name_or_path $ckpt_dir$model_ckpt_name --tokenizer_name t5-large --dataset "$dataset" --runseed 5 --eval_train --batch_size 1 --grad_accum_step 1 --output_model_dir=ckpts/tem/ --lr 1e-5 --epochs 50 --prompt_policy traversal --graph_transformer_graph_backbone graphormer --arch graphormer_base --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --encoder-layers 12 --encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --encoder-attention-heads 32 --pre-layernorm --num_classes 1 --attention_fasion coattention --transform_in_collator --task_policy traversal --zero_shot --only_test --unimodel --maskt2g --output_result_to_file $outfile --not_retest_tasks_in_result_file

  done


elif [ $model = 'grapht0_nomask' ]
then
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python  finetune_property_prediction_graph_transformer_multitask.py --model_name_or_path $ckpt_dir$model_ckpt_name --tokenizer_name t5-small --dataset "$dataset" --runseed 5 --eval_train --batch_size 16 --grad_accum_step 1 --output_model_dir=ckpts/tem/ --lr 1e-5 --epochs 50 --prompt_policy traversal --graph_transformer_graph_backbone graphormer --arch graphormer_base --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --encoder-layers 12 --encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --encoder-attention-heads 32 --pre-layernorm --num_classes 1 --attention_fasion coattention --transform_in_collator --task_policy traversal --zero_shot --only_test --unimodel --output_result_to_file $outfile --not_retest_tasks_in_result_file

  done

elif [ $model = 'grapht0_gnn' ]
then
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python  finetune_property_prediction_graph_transformer_multitask.py --model_name_or_path $ckpt_dir$model_ckpt_name --tokenizer_name t5-small --dataset "$dataset" --runseed 5 --eval_train --batch_size 16 --grad_accum_step 1 --output_model_dir=ckpts/tem/ --lr 1e-5 --epochs 50 --prompt_policy traversal --graph_transformer_graph_backbone gin --graph_transformer_text_backbone t5  --transform_in_collator --task_policy traversal --zero_shot --only_test  --output_result_to_file $outfile --not_retest_tasks_in_result_file
  done


elif [ $model = 'grapht0_fewshot' ]
then
  out_file_prefix=$out_file_prefix"few_shot_"$few_shot_number"_"
  outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
  echo $outfile
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python finetune_property_prediction_graph_transformer_multitask.py --dataset "$dataset" --model_name_or_path $ckpt_dir$model_ckpt_name --tokenizer_name t5-small  --runseed 5  --batch_size 128 --grad_accum_step 8 --lr 1e-2 --epochs 60 --prompt_policy traversal --graph_transformer_graph_backbone graphormer --arch graphormer_base --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --encoder-layers 12 --encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --encoder-attention-heads 32 --pre-layernorm --num_classes 1 --attention_fasion coattention --transform_in_collator --task_policy traversal  --few_shot $few_shot_number --few_shot_fashion linear --few_shot_prompt_fashion max --no_eval_train --disable_tqdm --test_interval 20 --output_result_to_file $outfile --unimodel --maskt2g --not_retest_tasks_in_result_file

  done

elif [ $model = 'grapht0_fewshot_reg' ]
then
  out_file_prefix=$out_file_prefix"few_shot_"$few_shot_number"_"
  outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
  echo $outfile
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python finetune_property_prediction_graph_transformer_multitask.py --dataset "$dataset" --model_name_or_path $ckpt_dir$model_ckpt_name --tokenizer_name t5-small  --runseed 5  --batch_size 128 --grad_accum_step 8 --lr 1e-2 --epochs 60 --prompt_policy traversal --graph_transformer_graph_backbone graphormer --arch graphormer_base --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --encoder-layers 12 --encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --encoder-attention-heads 32 --pre-layernorm --num_classes 1 --attention_fasion coattention --transform_in_collator --task_policy traversal  --few_shot $few_shot_number --few_shot_fashion linear --few_shot_prompt_fashion traversal --no_eval_train --disable_tqdm --test_interval 20 --output_result_to_file $outfile --unimodel --maskt2g --not_retest_tasks_in_result_file

  done

elif [ $model = 'grapht0_aug' ]
then
  out_file_prefix=$out_file_prefix"augment_"$prompt_augmentation"_"
  outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
  echo $outfile
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python  finetune_property_prediction_graph_transformer_multitask.py --model_name_or_path $ckpt_dir$model_ckpt_name --tokenizer_name t5-small --dataset "$dataset" --runseed 5 --eval_train --batch_size 16 --grad_accum_step 1 --output_model_dir=ckpts/tem/ --lr 1e-5 --epochs 50 --prompt_policy traversal --graph_transformer_graph_backbone graphormer --arch graphormer_base --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --encoder-layers 12 --encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --encoder-attention-heads 32 --pre-layernorm --num_classes 1 --attention_fasion coattention --transform_in_collator --task_policy traversal --zero_shot --only_test --unimodel --maskt2g --output_result_to_file $outfile --prompt_file all_augmented_downstream_task_prompt_multitask.json --prompt_augmentation $prompt_augmentation --not_retest_tasks_in_result_file
  done

elif [ $model = 'grapht0_abla' ]
then
  out_file_prefix=$out_file_prefix"augment_"$prompt_augmentation"_"
  outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
  echo $outfile
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python  finetune_property_prediction_graph_transformer_multitask.py --model_name_or_path $ckpt_dir$model_ckpt_name --tokenizer_name t5-small --dataset "$dataset" --runseed 5 --eval_train --batch_size 16 --grad_accum_step 1 --output_model_dir=ckpts/tem/ --lr 1e-5 --epochs 50 --prompt_policy traversal --graph_transformer_graph_backbone graphormer --arch graphormer_base --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --encoder-layers 12 --encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --encoder-attention-heads 32 --pre-layernorm --num_classes 1 --attention_fasion coattention --transform_in_collator --task_policy traversal --zero_shot --only_test --unimodel --maskt2g --output_result_to_file $outfile --prompt_augmentation $prompt_augmentation --not_retest_tasks_in_result_file --prompt_augmentation_file_prefix ablated
  done


elif [ $model = 'kvplm' ]
then
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python finetune_property_prediction_graph_transformer_multitask.py --dataset "$dataset" --runseed 5 --eval_train --batch_size 40 --grad_accum_step 2  --lr 1e-5 --epochs 50  --graph_transformer_graph_backbone kvplm --model_name_or_path $ckpt_dir$model_ckpt_name  --zero_shot --task_policy traversal --output_result_to_file $outfile --only_test --not_retest_tasks_in_result_file

  done

elif [ $model = 'kvplm_fewshot' ]
then
  out_file_prefix=$out_file_prefix"few_shot_"$few_shot_number"_"
  outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
  echo $outfile
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python finetune_property_prediction_graph_transformer_multitask.py --dataset "$dataset" --runseed 5 --eval_train --batch_size 128 --grad_accum_step 32 --lr 1e-2 --epochs 60  --graph_transformer_graph_backbone kvplm --init_checkpoint $ckpt_dir$model_ckpt_name --task_policy traversal --few_shot $few_shot_number --few_shot_fashion linear --few_shot_prompt_fashion max --test_interval 20 --no_eval_train --output_result_to_file $outfile --disable_tqdm --not_retest_tasks_in_result_file

  done

elif [ $model = 'kvplm_fewshot_reg' ]
then
  out_file_prefix=$out_file_prefix"few_shot_"$few_shot_number"_"
  outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
  echo $outfile
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python finetune_property_prediction_graph_transformer_multitask.py --dataset "$dataset" --runseed 5 --eval_train --batch_size 128 --grad_accum_step 32 --lr 1e-2 --epochs 60  --graph_transformer_graph_backbone kvplm --init_checkpoint $ckpt_dir$model_ckpt_name --task_policy traversal --few_shot $few_shot_number --few_shot_fashion linear --few_shot_prompt_fashion traversal --test_interval 20 --no_eval_train --output_result_to_file $outfile --disable_tqdm --not_retest_tasks_in_result_file

  done


elif [ $model = 'kvplm_aug' ]
then
  out_file_prefix=$out_file_prefix"augment_"$prompt_augmentation"_"
  outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
  echo $outfile
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python finetune_property_prediction_graph_transformer_multitask.py --dataset "$dataset" --runseed 5 --eval_train --batch_size 40 --grad_accum_step 2  --lr 1e-5 --epochs 50  --graph_transformer_graph_backbone kvplm --model_name_or_path $ckpt_dir$model_ckpt_name  --zero_shot --task_policy traversal --output_result_to_file $outfile --only_test --prompt_file all_augmented_downstream_task_prompt_multitask.json  --prompt_augmentation $prompt_augmentation --not_retest_tasks_in_result_file

  done

elif [ $model = 'momu' ]
then
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python finetune_property_prediction_graph_transformer_multitask.py --dataset "$dataset" --runseed 5 --eval_train --batch_size 40 --grad_accum_step 2  --lr 1e-5 --epochs 50  --graph_transformer_graph_backbone momu  --model_name_or_path $ckpt_dir$model_ckpt_name --zero_shot --task_policy traversal --output_result_to_file $outfile --only_test --not_retest_tasks_in_result_file

  done

elif [ $model = 'momu_fewshot' ]
then
  out_file_prefix=$out_file_prefix"few_shot_"$few_shot_number"_"
  outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
  echo $outfile
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python finetune_property_prediction_graph_transformer_multitask.py --dataset "$dataset" --runseed 5 --eval_train --batch_size 128 --grad_accum_step 32 --lr 1e-2 --epochs 60   --graph_transformer_graph_backbone momu  --init_checkpoint $ckpt_dir$model_ckpt_name --task_policy traversal --few_shot $few_shot_number --few_shot_fashion linear --few_shot_prompt_fashion max --test_interval 20 --no_eval_train --output_result_to_file $outfile --disable_tqdm --not_retest_tasks_in_result_file --num_workers 0

  done

elif [ $model = 'momu_aug' ]
then
  out_file_prefix=$out_file_prefix"augment_"$prompt_augmentation"_"
  outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
  echo $outfile
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python finetune_property_prediction_graph_transformer_multitask.py --dataset "$dataset" --runseed 5 --eval_train --batch_size 40 --grad_accum_step 2  --lr 1e-5 --epochs 50  --graph_transformer_graph_backbone momu  --model_name_or_path $ckpt_dir$model_ckpt_name --zero_shot --task_policy traversal --output_result_to_file $outfile --only_test --prompt_file all_augmented_downstream_task_prompt_multitask.json --prompt_augmentation $prompt_augmentation --not_retest_tasks_in_result_file

  done


elif [ $model = 'galactica' ]
then
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python finetune_property_prediction_graph_transformer_multitask.py --dataset "$dataset" --runseed 5 --eval_train --batch_size 1 --grad_accum_step 1  --lr 1e-5 --epochs 50  --graph_transformer_graph_backbone galactica --model_name_or_path $model_ckpt_name  --zero_shot --task_policy traversal --output_result_to_file $outfile --only_test --not_retest_tasks_in_result_file
  done


elif [ $model = 'gpt3' ]
then
for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device OPENAI_API_KEY=$openai_key python finetune_property_prediction_graph_transformer_multitask.py --dataset "$dataset" --runseed 5 --eval_train --batch_size 1 --grad_accum_step 1  --lr 1e-5 --epochs 50  --graph_transformer_graph_backbone gpt3 --model_name_or_path $model_ckpt_name  --zero_shot --task_policy traversal --output_result_to_file $outfile --only_test
done
fi