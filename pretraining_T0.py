#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
# from charset_normalizer import md__mypyc
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
from datasets import load_dataset

import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.data.data_collator import _torch_collate_batch
from transformers.tokenization_utils_base import BatchEncoding

from model import GraphT5TransformerForConditionalGeneration,GraphormerModel,GraphTransformer_dict,GraphormerConfig,GinConfig,Graphormer_version_dict,KVPLMConfig,MoMuConfig,MolT5Config,get_model,GalacticaConfig

from dataloaders import GraphTransformer_tokenizer_dict,GraphTransformer_collator_dict,graphormer_transform_for_dataset,WrapDataset

from ogb.utils import smiles2graph

from collections.abc import Mapping, Sequence

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch
import re
import numpy as np
from sklearn.metrics import (roc_auc_score,f1_score,confusion_matrix,r2_score)

from tqdm import tqdm
import argparse
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
import faulthandler

# from finetune_property_prediction_graph_transformer_multitask import reg_thre_by_task
import matplotlib.pyplot as plt

faulthandler.enable()

check_min_version("4.24.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    # all_data_to_lower_case: Optional[bool] = field(default=False)

    use_graph_transformer: bool = field(default=False)
    graph_transformer_graph_backbone: str = field(default='')
    graph_transformer_text_backbone: str = field(default='t5')
    attention_fasion: str= field(default='sequential')




    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    rich_features: Optional[bool] = field(default=False)

    transform_in_collator: Optional[bool] = field(default=False)

    wrap_dataset: Optional[bool] = field(default=False)



    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")



def evaluate_performance(trainer):
    dataloader = trainer.get_eval_dataloader()
    model = trainer._wrap_model(trainer.model, training=False, dataloader=dataloader)
    model.eval()

    str_y='Yes'
    str_n='No'


    id_y=trainer.tokenizer(str_y)['input_ids'][0]
    id_n=trainer.tokenizer(str_n)['input_ids'][0]


    # labels_all = []
    # for step, inputs in tqdm(enumerate(dataloader)):
    #     labels_all+=inputs['labels'][:,0].cpu().numpy().tolist()
    #     # if len(labels_all)>=80020:
    #     #     break
    # rate=(np.asarray(labels_all) == id_y).mean()
    # print('rate ',rate)
    #
    # # rate=0.7510904956840987
    # sudo_pred=np.random.choice([1.0, 0.0], size=(len(labels_all),), p=[rate, 1-rate])
    # sudo_pred+=np.random.rand(len(labels_all))
    #
    # sudo_auc=roc_auc_score(((torch.tensor(labels_all)==id_y)*2-1).numpy(), sudo_pred)
    # print('sudo_auc ',sudo_auc)

    preds_all=[]
    scores_all=[]
    labels_all = []

    for step, inputs in tqdm(enumerate(dataloader)):
        # Prediction step
        loss, logits, labels = trainer.prediction_step(model, inputs,prediction_loss_only=False)
        labels_all+=labels[:,0].cpu().numpy().tolist()
        preds_all+=logits[0].argmax(2)[:,0].cpu().numpy().tolist()
        scores_all+=(logits[0][:,0,id_y]-logits[0][:,0,id_n]).cpu().numpy().tolist()
        # if len(labels_all)>=80000:
        #     break

    acc=int((torch.tensor(labels_all) == torch.tensor(preds_all)).sum()) / len(labels_all)
    auc=roc_auc_score(((torch.tensor(labels_all)==id_y)*2-1).numpy(), np.asarray(scores_all))
    f1=f1_score(((torch.tensor(labels_all)==id_y)*2-1).numpy(),((torch.tensor(preds_all)==id_y)*2-1).numpy())
    con_m=confusion_matrix(((torch.tensor(labels_all)==id_y)*2-1).numpy(),((torch.tensor(preds_all)==id_y)*2-1).numpy())
    print('f1 ',f1)
    print('confusion matrix ', con_m)
    print('pred_rate ',(np.asarray(preds_all) == id_y).mean())
    return {'acc':acc,'auc':auc}

def evaluate_performance_generative(trainer,tokenizer):
    dataloader = trainer.get_eval_dataloader()
    model = trainer._wrap_model(trainer.model, training=False, dataloader=dataloader)
    model.eval()

    model.eval()
    y_true, y_scores = [], []
    for step, inputs in tqdm(enumerate(dataloader)):
        loss, logits, labels = trainer.prediction_step(model, inputs,prediction_loss_only=False)
        logits=logits[0]
        pred = []
        for i in range(logits.shape[0]):
            pred.append(tokenizer.decode(logits[i, :, :].argmax(1)))
        pred_number = []
        for result in pred:
            number_list = re.findall(r"-?\d+\.?\d*e??\d*?", result)
            pred_number.append(number_list[0] if len(number_list) > 0 else float(np.nan))
        true = []
        for i in range(inputs['labels'].shape[0]):
            true.append(tokenizer.decode(inputs['labels'][i, inputs['labels'][i, :] > 0]))
        true_number = []
        for result in true:
            number_list = re.findall(r"-?\d+\.?\d*e??\d*?", result)
            true_number.append((number_list[0]) if len(number_list) > 0 else float(np.nan))


        y_true.append(true)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    return



def eval_result(trainer,task_type='cla'):
    loader = trainer.get_eval_dataloader()
    model = trainer.model
    device=model.device
    tokenizer=trainer.tokenizer

    str_y='Yes'
    str_n='No'

    id_y=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str_y))[0]
    id_n=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str_n))[0]
    # id_y=trainer.tokenizer(str_y)['input_ids'][0]
    # id_n=trainer.tokenizer(str_n)['input_ids'][0]
    id_invalid=-100

    if task_type=='cla':
        model.eval()
        y_true, y_scores = [], []

        for step, batch in enumerate(loader):
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            with torch.no_grad():
                labels = batch["labels"]
                if hasattr(model, 'decoder'):
                    del batch["labels"]
                    batch["max_length"] = 3  # <PAD> CLASS <EOS>
                    output = model.generate(
                        **batch, output_scores=True, return_dict_in_generate=True
                        # num_beams=beam_size,
                        # no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                    logits = output.scores[0].unsqueeze(1)  # logits of CLASS
                elif hasattr(model.base_model, 'decoder'): #galactica
                    del batch["labels"]
                    batch["max_new_tokens"] = 1  # <PAD> CLASS <EOS>
                    # batch["max_length"] = 3
                    output = model.generate(
                        **batch, output_scores=True, return_dict_in_generate=True
                        # num_beams=beam_size,
                        # no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                    # for i in range(output.sequences.shape[0]):
                    #     print(tokenizer.decode(output.sequences[i]))
                    logits = output.scores[0].unsqueeze(1)  # logits of CLASS
                elif hasattr(model,'graph_encoder'): #momu, contrastive model; the model code require label
                    logits = model(**batch)['logits']
                else: #kvplm, self-masking model
                    del batch["labels"]
                    logits = model(**batch)['logits']
            index = labels != id_invalid #mask both text not answer and invalid labels; shape: [batch,answer length]
            if not isinstance(logits,dict): # for generative model
                # try:
                assert logits[index].ndim==2 # selected answer shape:[n_valid_sample,n_vocabulary]
                # except:
                #     print(batch['labels'])
                #     print(logits[index].shape)
                pred=(logits[index][:, id_y] - logits[index][:, id_n]).view([-1,1])
                true = labels[index].view(pred.shape)
                true[true == id_y] = 1
                true[true == id_n] = 0
                true[true == id_invalid] = -100
            else: # for contrastive model, logits is dict
                pred = (logits['pos'].unsqueeze(1)[index] - logits['neg'].unsqueeze(1)[index]).view([-1, 1]) #shape of logits['pos] and logits['pos] are [batch]
                true = labels[index].view(pred.shape)
                assert torch.sum(true == id_invalid) == 0 # For contrastive model, invalid label is previously replaced by id_invalid(-100). Replace it here. Not necessary, because only valid label are selected


            y_true.append(true)
            y_scores.append(pred)

        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
        y_pred=(y_scores>0).astype(int)


        roc_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                is_valid = y_true[:, i]  >= 0
                roc_list.append(roc_auc_score(y_true[is_valid, i], y_scores[is_valid, i]))
            else:
                print('{} is invalid'.format(i))

        if len(roc_list) < y_true.shape[1]:
            print(len(roc_list))
            print('Some target is missing!')
            print('Missing ratio: %f' % (1 - float(len(roc_list)) / y_true.shape[1]))

        acc =  int((y_pred == y_true).sum()) / len(y_true)
        auc = roc_list
        f1 = f1_score(y_true,
                      y_pred)
        con_m = confusion_matrix(y_true,
                      y_pred)
        print('f1 ', f1)
        print('confusion matrix ', con_m)
        print('pred_rate ', y_pred.mean())
        return {'acc': acc, 'auc': auc[0]}


    else:
        model.eval()
        y_true, y_scores = [], []
        for step, batch in enumerate(loader):
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            with torch.no_grad():
                logits = model(**batch)['logits']
            # index = batch['labels'] != id_invalid #mask both text not answer and invalid labels; shape: [batch,answer length]
            if not isinstance(logits, dict):  # for generative model
                # try:
                # assert logits.ndim == 2  # selected answer shape:[n_valid_sample,n_vocabulary]
                # except:
                #     print(batch['labels'])
                #     print(logits[index].shape)
                pred=[]
                for i in range(logits.shape[0]):
                    ind_valid=batch['labels'][i, :]>0
                    pred.append(tokenizer.decode(logits[i,ind_valid,:].argmax(1)))
                pred_number=[]
                for result in pred:
                    number_list=re.findall(r"-?\d+\.?\d*e??\d*?",result)
                    try:
                        decoded_number=eval(number_list[0])
                    except:
                        decoded_number=float(np.nan)
                    # decoded_number=re.sub(r"[^-\d\.]",'',result)
                    # if '-' in decoded_number:
                    #     sign=-1
                    # else:
                    #     sign=1
                    # decoded_number=decoded_number.replace('-','')
                    # while decoded_number.count('.')>1:
                    #     ind=decoded_number.rfind('.')
                    #     decoded_number=decoded_number[:ind]+decoded_number[(ind+1):]
                    # while decoded_number[0]=='0' and decoded_number[1]!='.':
                    #     decoded_number=decoded_number[1:]
                    # if decoded_number[0]=='.':
                    #     decoded_number = '0'+decoded_number
                    # decoded_number=eval(decoded_number)*sign
                    pred_number.append(decoded_number)
                true=[]
                for i in range(batch['labels'].shape[0]):
                    true.append(tokenizer.decode(batch['labels'][i, batch['labels'][i, :]>0]))
                true_number=[]
                for result in true:
                    number_list=re.findall(r"-?\d+\.?\d*e??\d*?",result)
                    true_number.append(eval((number_list[0])) if len(number_list)>0 else float(np.nan))
                # true[true == id_y] = 1
                # true[true == id_n] = 0
                # true[true == id_invalid] = -100
            else:  # for contrastive model, logits is dict
                raise ValueError("Not implemented for dict output!")

            y_true+=true_number
            y_scores+=pred_number

        y_true = torch.tensor(y_true)
        y_scores = torch.tensor(y_scores)

        ind=(~y_scores.isnan())&(y_scores.abs()<reg_thre_by_task(args.dataset))
        # ind = (~y_scores.isnan())
        ratio=ind.float().mean()
        y_true=y_true[ind]
        y_scores=y_scores[ind]

        mrs=(y_true-y_scores).std()
        naive_msr=(y_true-y_true.mean()).std()

        corrcoef=np.corrcoef(y_true,y_scores)[0,1]
        r2=r2_score(y_true,y_scores)



        plt.figure()
        plt.scatter(y_true,y_scores)
        global fig
        plt.savefig('cache/{}fig{}.png'.format(args.dataset,fig))
        fig+=1

        print(naive_msr)

        return {'ratio':float(ratio),'RMSE':float(mrs),'corrcoef':float(corrcoef),'R-Square':float(r2)}, 0, y_true, y_scores




def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # args_tem,left=parser.parse_known_args()



        #


    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, extra_paras = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args,left = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    # assert model_args.graph_transformer_graph_backbone in ['graphormer', 'gin']
    # if model_args.graph_transformer_graph_backbone == 'graphormer':
    #     parsernew = HfArgumentParser(GraphormerConfig)
    #     # parsernew = argparse.ArgumentParser()
    #     parsernew=GraphormerModel.add_args(parsernew)
    #     graph_args=parsernew.parse_args(left)
    #     Graphormer_version_dict[graph_args.arch](graph_args)
    #     # print('graphormer_args',graphormer_args)
    # elif model_args.graph_transformer_graph_backbone == 'gin':
    #     parsernew = HfArgumentParser(GinConfig)
    #     graph_args = parsernew.parse_args(left)

    assert model_args.graph_transformer_graph_backbone in ['graphormer', 'gin', 'kvplm', 'molt5', 'momu','galactica']
    if model_args.graph_transformer_graph_backbone == 'graphormer':
        parsernew = HfArgumentParser(GraphormerConfig)
        # parsernew = argparse.ArgumentParser()
        parsernew = GraphormerModel.add_args(parsernew)
        graph_args = parsernew.parse_args(left)
        graph_args=Graphormer_version_dict[graph_args.arch](graph_args)
        # print('graphormer_args',graphormer_args)
    elif model_args.graph_transformer_graph_backbone == 'gin':
        parsernew = HfArgumentParser(GinConfig)
        graph_args = parsernew.parse_args(left)
    elif model_args.graph_transformer_graph_backbone == 'kvplm' or model_args.graph_transformer_text_backbone == 'kvplm':
        model_args.graph_transformer_graph_backbone = 'kvplm'
        model_args.graph_transformer_text_backbone = 'kvplm'
        parsernew = HfArgumentParser(KVPLMConfig)
        graph_args = parsernew.parse_args(left)
        model_args.tokenizer_name = 'allenai/scibert_scivocab_uncased'
    elif model_args.graph_transformer_graph_backbone == 'momu' or model_args.graph_transformer_text_backbone == 'momu':
        model_args.graph_transformer_graph_backbone = 'momu'
        model_args.graph_transformer_text_backbone = 'momu'
        parsernew = HfArgumentParser(MoMuConfig)
        graph_args = parsernew.parse_args(left)
        model_args.tokenizer_name = 'allenai/scibert_scivocab_uncased'
    elif model_args.graph_transformer_graph_backbone == 'molt5' or model_args.graph_transformer_text_backbone == 'molt5':
        model_args.graph_transformer_graph_backbone = 'molt5'
        model_args.graph_transformer_text_backbone = 'molt5'
        parsernew = HfArgumentParser(MolT5Config)
        graph_args = parsernew.parse_args(left)
        assert graph_args.init_checkpoint in ['laituan245/molt5-base', 'laituan245/molt5-small',
                                              'laituan245/molt5-large']
        model_args.tokenizer_name = graph_args.init_checkpoint
    elif model_args.graph_transformer_graph_backbone == 'galactica' or model_args.graph_transformer_text_backbone == 'galactica':
        model_args.graph_transformer_graph_backbone = 'galactica'
        model_args.graph_transformer_text_backbone = 'galactica'
        parsernew = HfArgumentParser(GalacticaConfig)
        graph_args = parsernew.parse_args(left)
        assert model_args.model_name_or_path in ['facebook/galactica-1.3b', 'facebook/galactica-125m']
        model_args.tokenizer_name = model_args.model_name_or_path
    if model_args.graph_transformer_graph_backbone in ['kvplm', 'momu']:
        if graph_args.init_checkpoint is None:
            graph_args.init_checkpoint = model_args.model_name_or_path
        if model_args.model_name_or_path is None:
            model_args.model_name_or_path = graph_args.init_checkpoint

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mlm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        # if "validation" not in raw_datasets.keys():
        #     raw_datasets["validation"] = load_dataset(
        #         extension,
        #         data_files=data_files,
        #         split=f"train[:{data_args.validation_split_percentage}%]",
        #         cache_dir=model_args.cache_dir,
        #         use_auth_token=True if model_args.use_auth_token else None,
        #     )
        #     raw_datasets["train"] = load_dataset(
        #         extension,
        #         data_files=data_files,
        #         split=f"train[{data_args.validation_split_percentage}%:]",
        #         cache_dir=model_args.cache_dir,
        #         use_auth_token=True if model_args.use_auth_token else None,
        #     )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        # "do_lower_case": model_args.all_data_to_lower_case,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    # elif model_args.model_name_or_path:
    #     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )


    # special_tokens_dict = {'mask_token':'<MASK>','additional_special_tokens': ['<BOA>', '<EOA>']}

    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # print('We have added', num_added_toks, 'tokens')
    # tokenizer.model_input_names.append('answer_mask')



    if model_args.model_name_or_path:
        if model_args.graph_transformer_graph_backbone !='':
            model=get_model(model_args, graph_args,tokenizer)

        else:
            config_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            if model_args.config_name:
                config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
            elif model_args.model_name_or_path:
                config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
            else:
                config = CONFIG_MAPPING[model_args.model_type]()
                logger.warning("You are instantiating a new config instance from scratch.")
                if model_args.config_overrides is not None:
                    logger.info(f"Overriding config: {model_args.config_overrides}")
                    config.update_from_string(model_args.config_overrides)
                    logger.info(f"New config: {config}")
            model = AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            model.resize_token_embeddings(len(tokenizer))


    else:
        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }
        if model_args.config_name:
            config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
        elif model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        else:
            config = CONFIG_MAPPING[model_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            if model_args.config_overrides is not None:
                logger.info(f"Overriding config: {model_args.config_overrides}")
                config.update_from_string(model_args.config_overrides)
                logger.info(f"New config: {config}")
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)





    # tokenizer = AutoTokenizer.from_pretrained("laituan245/molt5-small", model_max_length=512)
    # model = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-small')
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        if model_args.graph_transformer_graph_backbone!='':
            tokenize_function=lambda x: GraphTransformer_tokenizer_dict[model_args.graph_transformer_graph_backbone][model_args.graph_transformer_text_backbone](examples=x,tokenizer=tokenizer,text_column_name=text_column_name,padding=padding,max_seq_length=max_seq_length,rich_features=data_args.rich_features,transform_in_collator=(data_args.transform_in_collator or data_args.wrap_dataset))
            # def tokenize_function(examples):
            #     # Remove empty lines
            #     # examples[text_column_name] = [line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()]
            #     text=tokenizer(
            #         examples[text_column_name],
            #         padding=padding,
            #         truncation=True,
            #         max_length=max_seq_length,
            #         # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            #         # receives the `special_tokens_mask`.
            #         return_special_tokens_mask=True,
            #     )
            #     labels = tokenizer(
            #         examples['label'],
            #         padding=padding,
            #         truncation=True,
            #         max_length=max_seq_length,
            #         # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            #         # receives the `special_tokens_mask`.
            #         return_special_tokens_mask=True,
            #     )
            #
            #
            #     #在这里转换tensor是没有意义的，需要在collater里面转换，因为dataset会被缓存后再加载，缓存会把tensor变成list
            #     graph_data=smiles2graph(examples['graph'])
            #     # graph_data={'x':torch.tensor(graph_data['node_feat']).long(),
            #     #             'edge_index':torch.tensor(graph_data['edge_index']).long(),
            #     #             'edge_attr':torch.tensor(graph_data['edge_feat']).long()}
            #     return {'graph':graph_data,
            #             'input_ids':text.data['input_ids'],
            #             'attention_mask':text.data['attention_mask'],
            #             'special_tokens_mask':text.data['special_tokens_mask'],
            #             'labels':labels.data['input_ids']}
        else:
            def tokenize_function(examples):
                # Remove empty lines
                examples[text_column_name] = [
                    line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
                ]
                return tokenizer(
                    examples[text_column_name],
                    padding=padding,
                    truncation=True,
                    max_length=max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )
        # tokenize_function(raw_datasets['train'][0])
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(raw_datasets['train']), data_args.max_train_samples)
            raw_datasets['train'] = raw_datasets['train'].shuffle().select(range(max_train_samples))
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(raw_datasets['validation']), data_args.max_eval_samples)
            raw_datasets['validation'] = raw_datasets['validation'].shuffle().select(range(max_eval_samples))

        with training_args.main_process_first(desc="dataset map tokenization"):
            # for item in tqdm(raw_datasets['train']):
            #     tokenize_function(item)

            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )

        if data_args.wrap_dataset:
            def transform_func(examples):
                return graphormer_transform_for_dataset(examples=examples,rich_features=data_args.rich_features)
            if training_args.do_train:
                tokenized_datasets["train"]=WrapDataset(data=tokenized_datasets["train"],transform=transform_func)
            if training_args.do_eval:
                tokenized_datasets["validation"] = WrapDataset(data=tokenized_datasets["validation"], transform=transform_func)

    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with training_args.main_process_first(desc="grouping texts together"):
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        # if data_args.max_train_samples is not None:
        #     max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        #     train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        # if data_args.max_eval_samples is not None:
        #     max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        #     eval_dataset = eval_dataset.shuffle().select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        # metric = evaluate.load("accuracy")

        # def compute_metrics(eval_preds):
        #     preds, labels = eval_preds
        #     # preds have the same shape as the labels, after the argmax(-1) has been calculated
        #     # by preprocess_logits_for_metrics
        #     labels = labels.reshape(-1)
        #     preds = preds.reshape(-1)
        #     mask = labels != -100
        #     labels = labels[mask]
        #     preds = preds[mask]
        #     return metric.compute(predictions=preds, references=labels)

    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    if model_args.graph_transformer_graph_backbone!='':
        data_collator = GraphTransformer_collator_dict[model_args.graph_transformer_graph_backbone][model_args.graph_transformer_text_backbone](
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
            transform_in_collator=data_args.transform_in_collator,
            rich_features=data_args.rich_features
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )

    # Initialize our Trainer
    training_args.remove_unused_columns=False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None
    )


    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # metrics = trainer.evaluate()

        # metrics = evaluate_performance_generative(trainer,tokenizer)
        metrics = eval_result(trainer)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        # try:
        #     perplexity = math.exp(metrics["eval_loss"])
        # except OverflowError:
        #     perplexity = float("inf")
        # metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()