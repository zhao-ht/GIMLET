import torch.utils.data

from torch_geometric.data.dataloader import Collater
from transformers import AutoTokenizer, AutoModel, BertModel, BertForPreTraining, BertConfig

import torch
from typing import Optional

from transformers import (
    DataCollatorForLanguageModeling,
)

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch
import numpy as np
from .graphormer_transform import graphormer_data_transform_tensor
from .graphormer_collator import collator_graph_data,padding
from .basic_collate import basic_collate
import numbers


def gpt3_conditional_generation_tokenizer(examples,tokenizer,text_column_name,padding,max_seq_length,**kwargs):

    data_new = {}
    text = examples[text_column_name] if isinstance(examples[text_column_name], str) else examples[text_column_name][0]
    tokenized_input = tokenizer(
        'Please answer questions on this molecule. The SMILES of this molecule is:' + examples['graph'] + '\n\n##Question: ' + text + '\n\nAnswer:',
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )

    if isinstance(examples['label'], torch.Tensor) or isinstance(examples['label'], numbers.Number):
        label = tokenizer(str(round(float(examples['label']), 2)))
    else:
        label = tokenizer(str(examples['label']))

    data_new['input_ids']=tokenized_input['input_ids']


    data_new['attention_mask'] = tokenized_input['attention_mask']
    data_new['labels'] = label['input_ids']

    return data_new

def gpt3_add_prompt_conditional_generation_transform_single(data,data_label,input_ids,attention_mask,label_dict,transform_in_collator,rich_features=False,raw_prompts=None,raw_label=None,tokenizer=None,generaltive_label=False,**kwargs):
    tokenized_input=tokenizer('Please answer questions on this molecule. The SMILES of this molecule is:' + data['smiles']+ '\n\nQuestion: '+raw_prompts+'\n\nAnswer: ',max_length=512,truncation=True)
    
    if generaltive_label:
        if isinstance(data_label,torch.Tensor) or isinstance(data_label, numbers.Number):
            label=tokenizer(str(round(float(data_label), 2)))
        else:
            label = tokenizer(str(data_label))
    else:
        if float(data_label) in label_dict:
            label=label_dict[float(data_label)]
        else:
            label=label_dict['invalid']

    input_ids_out=tokenized_input['input_ids']
    attention_mask_out=tokenized_input['attention_mask']

    
    if generaltive_label:
        return {
            'input_ids': input_ids_out,
            'attention_mask': attention_mask_out,
            'labels': label['input_ids'],
            'decoder_attention_mask':label['attention_mask']
            }
    else:
        return {
                'input_ids': input_ids_out,
                'attention_mask': attention_mask_out,
                'labels': label,
                }
