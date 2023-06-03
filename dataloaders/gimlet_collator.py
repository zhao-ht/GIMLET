# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from typing import Optional

from transformers import (
    DataCollatorForLanguageModeling,
)
from transformers.tokenization_utils_base import BatchEncoding

from collections.abc import Mapping, Sequence

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch
import numpy as np
from .graphormer_transform import graphormer_data_transform_tensor
from .graphormer_collator import collator_graph_data,padding
from .basic_collate import basic_collate
import time
class CollatorForGIMLETLanguageModeling(DataCollatorForLanguageModeling):

    def __init__(self,**kwargs):
        self.transform_in_collator= kwargs.pop('transform_in_collator')
        self.rich_features = kwargs.pop('rich_features')
        super().__init__(**kwargs)

    def __post_init__(self):

        if self.tf_experimental_compile:
            import tensorflow as tf

            self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)



    def torch_call(self, examples):

        #turn list to torch tensor
        graph_batch=[]
        text_batch=[]
        labels_batch=[]
        for example_data in examples:
            graph_data=example_data['graph']
            graph_batch.append(graph_data)
            text_batch.append({'input_ids': example_data['input_ids'],
                    'attention_mask': example_data['attention_mask'],
                               })
            if 'decoder_attention_mask' in example_data:
                labels_batch.append({'labels':example_data['labels'],'decoder_attention_mask':example_data['decoder_attention_mask']})
            else:
                labels_batch.append({'labels':example_data['labels']})

        graph_batch = collator_graph_data(graph_batch,transform_in_collator=self.transform_in_collator,rich_features=self.rich_features)

        text_batch = padding(text_batch, self.tokenizer.pad_token_id,self.tokenizer.pad_token_type_id,  pad_to_multiple_of=self.pad_to_multiple_of)
        labels_batch = padding(labels_batch, self.tokenizer.pad_token_id, self.tokenizer.pad_token_type_id,
                             pad_to_multiple_of=self.pad_to_multiple_of)

        batch={'graph': graph_batch,
            'input_ids': text_batch.data['input_ids'],
            'attention_mask': text_batch.data['attention_mask'],
               'labels':labels_batch.data['labels']}
        if 'decoder_attention_mask' in labels_batch.data:
            batch['decoder_attention_mask'] = labels_batch.data['decoder_attention_mask']

        return batch


