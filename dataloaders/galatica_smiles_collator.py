
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


class CollatorForSmilesTextLanguageModeling(DataCollatorForLanguageModeling):

    def __init__(self,**kwargs):
        self.transform_in_collator= kwargs.pop('transform_in_collator')
        self.rich_features = kwargs.pop('rich_features')
        super().__init__(**kwargs)

    def __post_init__(self):

        if self.tf_experimental_compile:
            import tensorflow as tf

            self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)



    def torch_call(self, examples):

        text_batch=[]
        labels_batch=[]
        for example_data in examples:

            text_batch.append({'input_ids': example_data['input_ids'],
                    'attention_mask': example_data['attention_mask'],
                               })

            labels_batch.append({'labels':example_data['labels'] })

        text_batch = padding(text_batch, self.tokenizer.pad_token_id,self.tokenizer.pad_token_type_id,  pad_to_multiple_of=self.pad_to_multiple_of)
        labels_batch = padding(labels_batch, self.tokenizer.pad_token_id, self.tokenizer.pad_token_type_id,
                             pad_to_multiple_of=self.pad_to_multiple_of)

        batch={'input_ids': text_batch.data['input_ids'],
            'attention_mask': text_batch.data['attention_mask'],
               'labels':labels_batch.data['labels']}

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        return batch




def galactica_conditional_generation_tokenizer(examples,tokenizer,text_column_name,padding,max_seq_length,**kwargs):

    data_new = {}
    text = examples[text_column_name] if isinstance(examples[text_column_name], str) else examples[text_column_name][0]
    tokenized_input = tokenizer(
        # examples[text_column_name]+ ' ',
        '[START_I_SMILES]' + examples['graph'] + '[END_I_SMILES]\n\n##Question: ' + text + '\n\nAnswer:',
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



def galactica_add_prompt_conditional_generation_transform_single(data,data_label,input_ids,attention_mask,label_dict,transform_in_collator,rich_features=False,raw_prompts=None,raw_label=None,tokenizer=None,generaltive_label=False,**kwargs):
    tokenized_input=tokenizer('[START_I_SMILES]' + data['smiles']+'[END_I_SMILES]\n\n##Question: '+raw_prompts+'\n\nAnswer:',max_length=512,truncation=True)
    
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











