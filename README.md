# GIMLET


This is the code for paper GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning

GIMLET is a unified transformer model for both graph and text data and is pretrained on large scale molecule tasks with instructions, towards instruction-based molecule zero-shot learning.

<div align=center>
<img src="https://github.com/zhao-ht/GIMLET/blob/master/fig/gimlet.png" width="600px">
</div>

The pretraining and downstream tasks are as follows:

<div align=center>
<img src="https://github.com/zhao-ht/GIMLET/blob/master/fig/mol_tasks.png" width="300px">
</div>



## Installation
To run GIMLET, please clone the repository to your local machine and install the required dependencies using the script provided.
### Note
Please note that the environment initialization script is for CUDA 11.1 and python 3.7. If you are using a different version of CUDA or python, please adjust the package version in script as necessary.
### Environment
```
conda create -n task_multi python=3.7

source activate task_multi

pip3 install torch==1.9+cu111 torchaudio  -f https://download.pytorch.org/whl/cu111/torch_stable.html

wget https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
pip install torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl

pip install torch_geometric==1.7.2

git clone https://github.com/huggingface/transformers

cd transformers
pip install --editable ./
python setup.py build_ext --inplace
cd ..

pip install datasets
pip install evaluate


pip install ogb
pip install spacy
pip install tqdm
pip install sklearn
pip install SentencePiece



pip install lmdb
pip install tensorboardX==2.4.1
pip install rdkit-pypi==2021.9.3
pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html


pip install tqdm
pip install wandb
pip install networkx
pip install setuptools==59.5.0
pip install multiprocess
pip install Cpython
pip install Cython
```


### Checkpoint Download

```
mkdir ckpts
cd ckpts
mkdir gimlet
cd gimlet
filename='gimlet.bin'
fileid='1kMK90Jv-9LGO-vKA34X-63ai8boIZcpu'
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
cd ..
cd ..
```


### Dataset Download


#### Downstream task Data

```
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
mv dataset property_data
```
Besides MoleculeNet, we also includes CYP450 which can be downloaded by 

```
``` 

#### Pretraining Data

You can download the pretraining dataset if you want to reproduce the pretraining or train your own model:

```
```



## Demo

We provide two simple examples of graph classification tasks. With informative prompt, The pretrained Graphormer-T5 coattention transformer achieves 0.76 AUC **zero-shot** performance on predicting molecule binding to Beta-secretase, and 0.72 AUC **zero shot** performance on predicting inhibition of molecule to HIV, while the SOTA pretrained GNN get 0.79 and 0.76 respectively by **supervised training**. Finetuning our model with prompt instruction achieves better performance, for example 0.82 AUC on Bace.

Before running the demo script, please set up environment, download MoleculeNet dataset, and pretrained model checkpoints.





Then run the demo script for zero-shot testing on HIV and Bace dataset (set dataset=bace or hiv):

```
dataset=bace 
CUDA_VISIBLE_DEVICES=0 python finetune_property_prediction_graph_transformer.py --model_name_or_path ckpts/test-mlmv-bio_transductive_nosmile_split3_2/ --tokenizer_name laituan245/molt5-small --dataset $dataset --runseed 5 --eval_train --batch_size 80 --grad_accum_step 16 --output_model_dir=ckpts/tem/ --lr 1e-5 --epochs 50 --prompt_policy traversal --graph_transformer_graph_backbone graphormer --arch graphormer_base --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --encoder-layers 12 --encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --encoder-attention-heads 32 --pre-layernorm --num_classes 1 --attention_fasion coattention --transform_in_collator --zero_shot
```

And run the following script for prompt-based finetuning (by delleting the --zero_shot argument):

```
dataset=bace 
CUDA_VISIBLE_DEVICES=0 python finetune_property_prediction_graph_transformer.py --model_name_or_path ckpts/test-mlmv-bio_transductive_nosmile_split3_2/ --tokenizer_name laituan245/molt5-small --dataset $dataset --runseed 5 --eval_train --batch_size 80 --grad_accum_step 16 --output_model_dir=ckpts/tem/ --lr 1e-5 --epochs 50 --prompt_policy traversal --graph_transformer_graph_backbone graphormer --arch graphormer_base --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --encoder-layers 12 --encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --encoder-attention-heads 32 --pre-layernorm --num_classes 1 --attention_fasion coattention --transform_in_collator
```



## Run Zero-Shot Downstream Task

You can run all the downstream tasks by script finetune_property_prediction_graph_transformer_multitask.sh:


## Run the Pretraining 

You can reproduce the pretraining by yourself:

```
CUDA_VISIBLE_DEVICES=0 python pretraining_T0.py --model_name_or_path t5-small --tokenizer_name t5-small  --train_file pretrain_datasets/ChEMBL_STRING/mole_graph_text_T0_rich_negative05.csv  --use_graph_transformer --transform_in_collator --per_device_train_batch_size 64 --per_device_eval_batch_size 200 --do_train  --output_dir ckpts/pretrained_result   --line_by_line --save_steps 10000 --graph_transformer_graph_backbone graphormer --arch graphormer_base --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --encoder-layers 12 --encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --encoder-attention-heads 32 --pre-layernorm --num_classes 1 --attention_fasion coattention  --unimodel
```

You can run your own pretraining by specifying --train_file as your pretraining file, or imply your model into the pipeline.


## Citation

Please cite our paper if you find it helpful.
```

```









