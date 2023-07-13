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
from ogb.utils import smiles2graph

def contrastive_conditional_generation_tokenizer(examples,tokenizer,text_column_name,padding,max_seq_length,rich_features,**kwargs):
    label_dict={'Yes':[1],'No':[0]}
    data_new = {}
    text=examples[text_column_name] if isinstance(examples[text_column_name],str) else examples[text_column_name][0]
    tokenized_input_pos=tokenizer(text+' '+'Yes',truncation=True,max_length=512)
    tokenized_input_neg=tokenizer(text+' '+'No',truncation=True,max_length=512)
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

        batch={'graph': graph_batch,
            'input_ids_pos': text_batch_pos.data['input_ids'],
            'attention_mask_pos': text_batch_pos.data['attention_mask'],
               'input_ids_neg': text_batch_neg.data['input_ids'],
               'attention_mask_neg': text_batch_neg.data['attention_mask'],
               'labels':labels_batch.data['labels']}

        return batch


