
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
from .graph_text_transform import graphormer_data_transform_tensor
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
        # Handle dict or lists with proper padding and conversion to tensor.
        # elem = examples[0]
        #
        # if isinstance(examples[0], dict):
        #     batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        # else:
        #     batch = {
        #         "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        #     }

        # If special token mask has been preprocessed, pop it from the dict.

        #把所有list变成torch tensor
        # graph_batch=[]
        text_batch=[]
        labels_batch=[]
        for example_data in examples:
            # graph_data=example_data['graph']
            # graph_batch.append(graph_data)
            text_batch.append({'input_ids': example_data['input_ids'],
                    'attention_mask': example_data['attention_mask'],
                               })
            # if 'labels' in example_data and not isinstance(example_data['labels'],str):
            labels_batch.append({'labels':example_data['labels'] })
            # else:
            #     assert 'label' in example_data and not isinstance(example_data['label'],str)
            #     labels_batch.append({'labels': example_data['label']})

        # graph_batch = collator_graph_data(graph_batch,transform_in_collator=self.transform_in_collator,rich_features=self.rich_features)
        # text_batch = self.tokenizer.pad(text_batch, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        text_batch = padding(text_batch, self.tokenizer.pad_token_id,self.tokenizer.pad_token_type_id,  pad_to_multiple_of=self.pad_to_multiple_of)
        labels_batch = padding(labels_batch, self.tokenizer.pad_token_id, self.tokenizer.pad_token_type_id,
                             pad_to_multiple_of=self.pad_to_multiple_of)

        # graph_batch={'x':graph_batch['graph']['x'],'edge_index':graph_batch['graph']['edge_index'],'edge_attr':graph_batch['graph']['edge_attr'],'batch':graph_batch['graph']['batch'],'ptr':graph_batch['graph']['ptr']}
        batch={'input_ids': text_batch.data['input_ids'],
            'attention_mask': text_batch.data['attention_mask'],
               'labels':labels_batch.data['labels']}

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        # if self.mlm:
        #     batch["input_ids"], batch["labels"] = self.torch_mask_answer_tokens(
        #         text_batch, special_tokens_mask=special_tokens_mask
        #     )
        # # elif self.mam:
        #
        # else:
        #     labels = batch["input_ids"].clone()
        #     if self.tokenizer.pad_token_id is not None:
        #         labels[labels == self.tokenizer.pad_token_id] = -100
        #     batch["labels"] = labels
        return batch


    # def torch_mask_answer_tokens(self, text_batch, special_tokens_mask: Optional) :
    #     """
    #     Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    #     """
    #     import torch
    #     inputs = text_batch['input_ids'].clone()
    #     labels = text_batch['input_ids'].clone()
    #     # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    #     # probability_matrix = torch.full(labels.shape, self.mlm_probability)
    #     # if special_tokens_mask is None:
    #     #     special_tokens_mask = [
    #     #         self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    #     #     ]
    #     #     special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    #     # else:
    #     #     special_tokens_mask = special_tokens_mask.bool()
    #
    #     # probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    #     masked_indices = ((~text_batch['answer_mask']) & text_batch['attention_mask']).bool()
    #     labels[~masked_indices] = -100  # We only compute loss on masked tokens
    #
    #     # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    #     indices_replaced = masked_indices
    #     inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
    #
    #     # # 10% of the time, we replace masked input tokens with random word
    #     # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    #     # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
    #     # inputs[indices_random] = random_words[indices_random]
    #
    #     # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    #     return inputs, labels



# for kvplm add prompt transform, assume data is smiles string, and we contact prompt strings before smiles.


def galactica_conditional_generation_tokenizer(examples,tokenizer,text_column_name,padding,max_seq_length,**kwargs):

    data_new = {}
    tokenized_input = tokenizer(
        # examples[text_column_name]+ ' ',
        '[START_I_SMILES]' + examples['graph'] + '[END_I_SMILES]\n\n##Question: ' + examples[text_column_name] + '\n\nAnswer:',
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )
    # tokenized_input = tokenizer(examples[]+ ' ', max_length=512,truncation=True)
    # Mask_attention_mask=tokenizer('[MASK]')['attention_mask'][1]
    if isinstance(examples['label'], torch.Tensor) or isinstance(examples['label'], numbers.Number):
        label = tokenizer(str(round(float(examples['label']), 2)))
    else:
        label = tokenizer(str(examples['label']))
    # if label['input_ids'][-1]==tokenizer.vocab['[SEP]']:
    #     label['input_ids']=label['input_ids'][:-1]
    # if label['input_ids'][0]==tokenizer.vocab['[CLS]']:
    #     label['input_ids'] = label['input_ids'][1:]

    data_new['input_ids']=tokenized_input['input_ids']


    data_new['attention_mask'] = tokenized_input['attention_mask']
    data_new['labels'] = label['input_ids']

    return data_new



def galactica_add_prompt_conditional_generation_transform_single(data,data_label,input_ids,attention_mask,label_dict,transform_in_collator,rich_features=False,raw_prompts=None,raw_label=None,tokenizer=None,generaltive_label=False,**kwargs):
    tokenized_input=tokenizer('[START_I_SMILES]' + data['smiles']+'[END_I_SMILES]\n\n##Question: '+raw_prompts+' The answer should contain Yes or No\n\nAnswer:',max_length=512,truncation=True)
    
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











