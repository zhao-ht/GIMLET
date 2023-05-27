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
from .graph_text_transform import graphormer_data_transform_tensor
from .graphormer_collator import collator_graph_data,padding
from .basic_collate import basic_collate
import time
class CollatorForGraphormerTextLanguageModeling(DataCollatorForLanguageModeling):

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

        # text_batch = self.tokenizer.pad(text_batch, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        text_batch = padding(text_batch, self.tokenizer.pad_token_id,self.tokenizer.pad_token_type_id,  pad_to_multiple_of=self.pad_to_multiple_of)
        labels_batch = padding(labels_batch, self.tokenizer.pad_token_id, self.tokenizer.pad_token_type_id,
                             pad_to_multiple_of=self.pad_to_multiple_of)

        # graph_batch={'x':graph_batch['graph']['x'],'edge_index':graph_batch['graph']['edge_index'],'edge_attr':graph_batch['graph']['edge_attr'],'batch':graph_batch['graph']['batch'],'ptr':graph_batch['graph']['ptr']}
        batch={'graph': graph_batch,
            'input_ids': text_batch.data['input_ids'],
            'attention_mask': text_batch.data['attention_mask'],
               'labels':labels_batch.data['labels']}
        if 'decoder_attention_mask' in labels_batch.data:
            batch['decoder_attention_mask'] = labels_batch.data['decoder_attention_mask']


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

