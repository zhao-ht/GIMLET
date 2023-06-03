#!/bin/bash
device=$1
model=$2
model_ckpt_name=$3

few_shot_number=$4

prompt_augmentation=$5

ckpt_dir="ckpts/"
if [ ! -d cache  ];then
  mkdir cache
fi
out_file_prefix="cache/testing_"
outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
echo "Output results to: "$outfile

export dataset_list=(bace hiv muv tox21 toxcast bbbp cyp450 pcba esol lipo freesolv)


if [ $model = 'gimlet' ]
then
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python  downstream_test.py --zero_shot  --transformer_backbone gimlet --model_name_or_path $ckpt_dir$model_ckpt_name --tokenizer_name t5-small --dataset "$dataset" --runseed 5 --batch_size 40 --grad_accum_step 1   --transform_in_collator --only_test --output_result_to_file $outfile --not_retest_tasks_in_result_file

  done

elif [ $model = 'gimlet_fewshot' ]
then
  out_file_prefix=$out_file_prefix"few_shot_"$few_shot_number"_"
  outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
  echo $outfile
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python  downstream_test.py --few_shot $few_shot_number --few_shot_fashion linear    --transformer_backbone gimlet --model_name_or_path $ckpt_dir$model_ckpt_name --tokenizer_name t5-small --dataset "$dataset" --runseed 5 --batch_size 128 --grad_accum_step 8 --lr 1e-2 --epochs 60 --test_interval 20  --transform_in_collator --no_eval_train --output_result_to_file $outfile --not_retest_tasks_in_result_file

  done

elif [ $model = 'gimlet_aug' ]
then
  out_file_prefix=$out_file_prefix"augment_"$prompt_augmentation"_"
  outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
  echo $outfile
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python  downstream_test.py --zero_shot  --transformer_backbone gimlet --model_name_or_path $ckpt_dir$model_ckpt_name --tokenizer_name t5-small --dataset "$dataset" --prompt_file augmented_selected_prompt_downstream_task.json --runseed 5 --batch_size 40 --grad_accum_step 1   --transform_in_collator --only_test --output_result_to_file $outfile --not_retest_tasks_in_result_file --prompt_augmentation $prompt_augmentation

  done

elif [ $model = 'kvplm' ]
then
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python downstream_test.py  --zero_shot  --transformer_backbone kvplm  --model_name_or_path $ckpt_dir$model_ckpt_name --dataset "$dataset" --prompt_file prompt_downstream_task.json --runseed 5 --batch_size 40 --grad_accum_step 1  --only_test --output_result_to_file $outfile  --not_retest_tasks_in_result_file

  done

elif [ $model = 'kvplm_fewshot' ]
then
  out_file_prefix=$out_file_prefix"few_shot_"$few_shot_number"_"
  outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
  echo $outfile
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python downstream_test.py  --few_shot $few_shot_number --few_shot_fashion linear  --transformer_backbone kvplm  --model_name_or_path $ckpt_dir$model_ckpt_name --dataset "$dataset" --prompt_file prompt_downstream_task.json --runseed 5 --batch_size 128 --grad_accum_step 8 --lr 1e-2 --epochs 60 --test_interval 20 --no_eval_train   --output_result_to_file $outfile  --not_retest_tasks_in_result_file

  done

elif [ $model = 'kvplm_aug' ]
then
  out_file_prefix=$out_file_prefix"augment_"$prompt_augmentation"_"
  outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
  echo $outfile
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python downstream_test.py  --zero_shot  --transformer_backbone kvplm  --model_name_or_path $ckpt_dir$model_ckpt_name --dataset "$dataset" --prompt_file augmented_prompt_downstream_task.json --runseed 5 --batch_size 40 --grad_accum_step 1  --only_test --output_result_to_file $outfile  --not_retest_tasks_in_result_file --prompt_augmentation $prompt_augmentation

  done

elif [ $model = 'momu' ]
then
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python downstream_test.py --zero_shot --transformer_backbone momu  --model_name_or_path $ckpt_dir$model_ckpt_name  --dataset "$dataset" --prompt_file prompt_downstream_task.json --runseed 5 --batch_size 40 --grad_accum_step 1  --only_test --output_result_to_file $outfile  --not_retest_tasks_in_result_file

  done

elif [ $model = 'momu_fewshot' ]
then
  out_file_prefix=$out_file_prefix"few_shot_"$few_shot_number"_"
  outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
  echo $outfile
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python downstream_test.py --few_shot $few_shot_number --few_shot_fashion linear --transformer_backbone momu  --model_name_or_path $ckpt_dir$model_ckpt_name  --dataset "$dataset" --prompt_file prompt_downstream_task.json --runseed 5  --batch_size 128 --grad_accum_step 8 --lr 1e-2 --epochs 60 --test_interval 20 --no_eval_train --output_result_to_file $outfile  --not_retest_tasks_in_result_file

  done

elif [ $model = 'momu_aug' ]
then
  out_file_prefix=$out_file_prefix"augment_"$prompt_augmentation"_"
  outfile=$out_file_prefix${model_ckpt_name//'/'/'_'}".csv"
  echo $outfile
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python downstream_test.py --zero_shot --transformer_backbone momu  --model_name_or_path $ckpt_dir$model_ckpt_name  --dataset "$dataset" --prompt_file augmented_prompt_downstream_task.json --runseed 5 --batch_size 40 --grad_accum_step 1  --only_test --output_result_to_file $outfile  --not_retest_tasks_in_result_file --prompt_augmentation $prompt_augmentation

  done

elif [ $model = 'galactica' ]
then
  for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device python downstream_test.py --zero_shot  --transformer_backbone galactica --model_name_or_path $model_ckpt_name --dataset "$dataset" --prompt_file prompt_downstream_task.json --runseed 5 --batch_size 1 --grad_accum_step 1   --only_test  --output_result_to_file $outfile --not_retest_tasks_in_result_file

  done

elif [ $model = 'gpt3' ]
then
for dataset in "${dataset_list[@]}"; do
    CUDA_VISIBLE_DEVICES=$device OPENAI_API_KEY=$openai_key python downstream_test.py --dataset "$dataset" --runseed 5 --eval_train --batch_size 1 --grad_accum_step 1  --lr 1e-5 --epochs 50  --transformer_backbone gpt3 --model_name_or_path $model_ckpt_name  --zero_shot  --output_result_to_file $outfile --only_test

done
fi