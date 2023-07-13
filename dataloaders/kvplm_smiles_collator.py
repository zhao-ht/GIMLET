
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

class CollatorForKVPLM(Collater):
    def __init__(self, **kwargs):
        self.transform_in_collator = kwargs.pop('transform_in_collator')
        self.include_y = kwargs.pop('include_y')
        self.rich_features = kwargs.pop('rich_features')
        # self.tokenizer = kwargs.pop('tokenizer')
        # self.smiles = kwargs.pop('smiles')
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        try:
            super().__init__([],[])
        except:
            print('torch_geometric version is too low. Collater only accept one parameter for initialization')
            super().__init__([])

    def collate(self, batch):
        items_new=[]
        for item in batch:
            index = item.id
            lab = item.y
            token = self.tokenizer.encode(item.smiles.strip('\n'))
            tok = np.zeros(64)
            att = torch.zeros(64).long()
            tok[:min(64, len(token))] = token[:min(64, len(token))]
            tok=torch.tensor(tok).long()
            att[:min(64, len(token))] = 1
            typ = torch.zeros(tok.shape).long()
            item_new=Data()
            item_new['y']=lab.unsqueeze(0)
            item_new['tokens']=tok.unsqueeze(0)
            item_new['token_type_ids']=typ.unsqueeze(0)
            item_new['attention_mask']=att.unsqueeze(0)
            items_new.append(item_new)

        graph_batch = basic_collate(items_new)
        return graph_batch


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

        # graph_batch=[]
        text_batch=[]
        labels_batch=[]
        for example_data in examples:

            text_batch.append({'input_ids': example_data['input_ids'],
                    'attention_mask': example_data['attention_mask'],
                               })

            labels_batch.append({'labels':example_data['labels'] })

        # galatica tokenizer has no pad_token_id
        if self.tokenizer.pad_token_id is None:
            if '<pad>' in self.tokenizer.vocab:
                self.tokenizer.pad_token_id = self.tokenizer.vocab['<pad>']
            else:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        text_batch = padding(text_batch, self.tokenizer.pad_token_id,self.tokenizer.pad_token_type_id,  pad_to_multiple_of=self.pad_to_multiple_of)
        labels_batch = padding(labels_batch, self.tokenizer.pad_token_id, self.tokenizer.pad_token_type_id,
                             pad_to_multiple_of=self.pad_to_multiple_of)

        batch={'input_ids': text_batch.data['input_ids'],
            'attention_mask': text_batch.data['attention_mask'],
               'labels':labels_batch.data['labels']}

        special_tokens_mask = batch.pop("special_tokens_mask", None)

        return batch


# for kvplm add prompt transform, assume data is smiles string, and we contact prompt strings before smiles.


def kvplm_conditional_generation_tokenizer(examples,tokenizer,text_column_name,padding,max_seq_length,**kwargs):

    data_new = {}
    text=examples[text_column_name] if isinstance(examples[text_column_name],str) else examples[text_column_name][0]
    tokenized_input = tokenizer(
        examples['graph'] + ' '+
        text+ ' ',
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )
    # tokenized_input = tokenizer(examples[]+ ' ', max_length=512,truncation=True)
    Mask_attention_mask=tokenizer('[MASK]')['attention_mask'][1]
    if isinstance(examples['label'], torch.Tensor) or isinstance(examples['label'], numbers.Number):
        label = tokenizer(str(round(float(examples['label']), 2)))
    else:
        label = tokenizer(str(examples['label']))
    if label['input_ids'][-1]==tokenizer.vocab['[SEP]']:
        label['input_ids']=label['input_ids'][:-1]
    if label['input_ids'][0]==tokenizer.vocab['[CLS]']:
        label['input_ids'] = label['input_ids'][1:]

    data_new['input_ids']=tokenized_input['input_ids']+\
                          (np.ones_like(label['input_ids'])*tokenizer.vocab['[MASK]']).tolist()

    data_new['attention_mask'] = tokenized_input['attention_mask']+\
                                 (np.ones_like(label['input_ids'])*Mask_attention_mask).tolist()
    data_new['labels'] = (np.ones_like(tokenized_input['input_ids']) * -100).tolist()+label['input_ids']

    return data_new



def kvplm_add_prompt_conditional_generation_transform_single(data,data_label,input_ids,attention_mask,label_dict,transform_in_collator,rich_features=False,raw_prompts=None,raw_label=None,tokenizer=None,generaltive_label=False,**kwargs):
    if not generaltive_label:
        data_new = {}
        tokenized_input=tokenizer(data['smiles']+' '+raw_prompts+' '+'[MASK]',max_length=512,truncation=True)
        data_new['input_ids']=tokenized_input['input_ids']
        data_new['attention_mask']=tokenized_input['attention_mask']
        data_new['labels'] = np.ones_like(data_new['input_ids']) * -100
        index_label=np.array(data_new['input_ids'])==tokenizer.vocab['[MASK]']
        if float(data.y) in label_dict:
            data_new['labels'][index_label] = label_dict[float(data.y)]
        else:
            data_new['labels'][index_label]=label_dict['invalid']
        data_new['labels'] = data_new['labels'].tolist()
    else:
        data_new = {}
        tokenized_input = tokenizer(data['smiles'] + ' ' + raw_prompts + ' ', max_length=512,truncation=True)
        Mask_attention_mask=tokenizer('[MASK]')['attention_mask'][0]
        if isinstance(data.y, torch.Tensor) or isinstance(data.y, numbers.Number):
            label = tokenizer(str(round(float(data.y), 2)))
        else:
            label = tokenizer(str(data.y))

        data_new['input_ids']=tokenized_input['input_ids']+\
                              (np.ones_like(label['input_ids'])*tokenizer.vocab['[MASK]']).tolist()

        data_new['attention_mask'] = tokenized_input['attention_mask']+\
                                     (np.ones_like(label['input_ids'])*Mask_attention_mask).tolist()
        data_new['labels'] = (np.ones_like(tokenized_input['input_ids']) * -100).tolist()+label['input_ids']

    return data_new









