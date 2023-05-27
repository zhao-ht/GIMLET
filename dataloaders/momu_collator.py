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
from ogb.utils import smiles2graph

def contrastive_conditional_generation_tokenizer(examples,tokenizer,text_column_name,padding,max_seq_length,rich_features,**kwargs):
    label_dict={'Yes':[1],'No':[0]}
    data_new = {}
    tokenized_input_pos=tokenizer(examples[text_column_name]+' '+'Yes',truncation=True,max_length=512)
    tokenized_input_neg=tokenizer(examples[text_column_name]+' '+'No',truncation=True,max_length=512)
    # if not transform_in_collator:
    #     examples['graph'] = smiles2graph(examples['graph'])
    data_new['graph']=examples['graph']
    data_new['input_ids_pos']=tokenized_input_pos['input_ids']
    data_new['attention_mask_pos']=tokenized_input_pos['attention_mask']
    data_new['input_ids_neg']=tokenized_input_neg['input_ids']
    data_new['attention_mask_neg']=tokenized_input_neg['attention_mask']


    # if float(examples['label']) in label_dict:
    #     data_new['labels'] = [float(data.y)]
    # else:
    data_new['labels']=label_dict[examples['label']]


    return data_new





def contrastive_add_prompt_conditional_generation_transform_single(data,data_label,input_ids,attention_mask,label_dict,transform_in_collator,rich_features=False,raw_prompts=None,raw_label=None,tokenizer=None,**kwargs):
    data_new = {}
    tokenized_input_pos=tokenizer(raw_prompts+' '+raw_label[1],truncation=True,max_length=512)
    tokenized_input_neg=tokenizer(raw_prompts+' '+raw_label[0],truncation=True,max_length=512)
    data_new['graph']=data
    if input_ids[-1]==tokenizer.sep_token_id:
        data_new['input_ids_pos']=input_ids[0:-1]+label_dict[1]+[input_ids[-1]]
        data_new['attention_mask_pos']=attention_mask+[1]
        data_new['input_ids_neg']=input_ids[0:-1]+label_dict[0]+[input_ids[-1]]
        data_new['attention_mask_neg']=attention_mask+[1]
    else:
        data_new['input_ids_pos']=input_ids[0:]+label_dict[1]
        data_new['attention_mask_pos']=attention_mask+[1]
        data_new['input_ids_neg']=input_ids[0:]+label_dict[0]
        data_new['attention_mask_neg']=attention_mask+[1]
    #
    #
    # data_new = {}
    # tokenized_input_pos=tokenizer(raw_prompts+' '+raw_label[1],truncation=True,max_length=512)
    # tokenized_input_neg=tokenizer(raw_prompts+' '+raw_label[0],truncation=True,max_length=512)
    # data_new['graph']=data
    # data_new['input_ids_pos']=tokenized_input_pos['input_ids']
    # data_new['attention_mask_pos']=tokenized_input_pos['attention_mask']
    # data_new['input_ids_neg']=tokenized_input_neg['input_ids']
    # data_new['attention_mask_neg']=tokenized_input_neg['attention_mask']

    if float(data.y) in label_dict:
        data_new['labels'] = [float(data.y)]
    else:
        data_new['labels']=label_dict['invalid']


    return data_new



# for kvplm add prompt transform, assume data is smiles string, and we contact prompt strings before smiles.
def contrastive_add_prompt_conditional_generation_transform_sample(data,data_label,input_ids,attention_mask,label_dict,prompt_ids,transform_in_collator,rich_features=False,raw_prompts=None,raw_label=None,tokenizer=None,**kwargs):
    data_new = {}
    prompt_ids = prompt_ids[np.random.randint(len(raw_prompts))]
    tokenized_input_pos=tokenizer(raw_prompts[prompt_ids]+' '+raw_label[1],truncation=True,max_length=512)
    tokenized_input_neg=tokenizer(raw_prompts[prompt_ids]+' '+raw_label[0],truncation=True,max_length=512)
    data_new['graph']=data
    data_new['input_ids_pos']=tokenized_input_pos['input_ids']
    data_new['attention_mask_pos']=tokenized_input_pos['attention_mask']
    data_new['input_ids_neg']=tokenized_input_neg['input_ids']
    data_new['attention_mask_neg']=tokenized_input_neg['attention_mask']

    if float(data.y) in label_dict:
        data_new['labels'] = [float(data.y)]
    else:
        data_new['labels'] = label_dict['invalid']

    return data_new



class CollatorForContrastiveGraphLanguageModeling(DataCollatorForLanguageModeling):

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

        #turn list to torch tensor

        #rich_features is not necessary for MoMu because MoMu Model only the basic features
        graph_batch=[]
        text_batch_pos=[]
        text_batch_neg=[]
        labels_batch=[]
        for example_data in examples:
            graph_data=example_data['graph']
            if isinstance(graph_data, str):
                graph_data = smiles2graph(graph_data)
                x = torch.tensor(graph_data['node_feat']).long()
                # print(x)
                edge_index = torch.tensor(graph_data['edge_index']).long()
                edge_attr = torch.tensor(graph_data['edge_feat']).long()
                num_nodes = int(graph_data['num_nodes'])
                if not self.rich_features:
                    x = x[:, 0:2]
                    edge_attr = edge_attr[:, 0:2]
                graph_data = Data(x, edge_index, edge_attr, num_nodes=num_nodes)
            graph_batch.append(graph_data)
            text_batch_pos.append({'input_ids': example_data['input_ids_pos'],
                    'attention_mask': example_data['attention_mask_pos'],
                               })
            text_batch_neg.append({'input_ids': example_data['input_ids_neg'],
                    'attention_mask': example_data['attention_mask_neg'],
                               })
            labels_batch.append({'labels':example_data['labels']})

        graph_batch = basic_collate(graph_batch)
        # text_batch = self.tokenizer.pad(text_batch, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        text_batch_pos = padding(text_batch_pos, self.tokenizer.pad_token_id,self.tokenizer.pad_token_type_id,  pad_to_multiple_of=self.pad_to_multiple_of)
        text_batch_neg = padding(text_batch_neg, self.tokenizer.pad_token_id, self.tokenizer.pad_token_type_id,
                                 pad_to_multiple_of=self.pad_to_multiple_of)
        labels_batch = padding(labels_batch, self.tokenizer.pad_token_id, self.tokenizer.pad_token_type_id,
                             pad_to_multiple_of=self.pad_to_multiple_of)

        # graph_batch={'x':graph_batch['graph']['x'],'edge_index':graph_batch['graph']['edge_index'],'edge_attr':graph_batch['graph']['edge_attr'],'batch':graph_batch['graph']['batch'],'ptr':graph_batch['graph']['ptr']}
        batch={'graph': graph_batch,
            'input_ids_pos': text_batch_pos.data['input_ids'],
            'attention_mask_pos': text_batch_pos.data['attention_mask'],
               'input_ids_neg': text_batch_neg.data['input_ids'],
               'attention_mask_neg': text_batch_neg.data['attention_mask'],
               'labels':labels_batch.data['labels']}

        # special_tokens_mask = batch.pop("special_tokens_mask", None)
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

