from typing import Optional

from transformers import (
    DataCollatorForLanguageModeling,
)
from transformers.tokenization_utils_base import BatchEncoding

from collections.abc import Mapping, Sequence

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch
from torch_geometric.data.dataloader import Collater
from .basic_collate import basic_collate
from ogb.utils import smiles2graph
from .graphormer_collator import collator_graph_data,padding

class CollatorForGinTextLanguageModeling(DataCollatorForLanguageModeling):


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
        graph_batch=[]
        text_batch=[]
        labels_batch=[]
        for example_data in examples:
            graph_data=example_data['graph']
            if self.transform_in_collator:
                if isinstance(graph_data, str):
                    graph_data = smiles2graph(graph_data)
            item_new = Data()
            if isinstance(graph_data, Data):
                for key in graph_data.keys:
                    item_new[key] = torch.tensor(graph_data[key]) if (not isinstance(graph_data[key], torch.Tensor)) else graph_data[key]
            else:
                for key in graph_data.keys():
                    item_new[key] = torch.tensor(graph_data[key]) if (not isinstance(graph_data[key], torch.Tensor)) else graph_data[key]
            graph_data=item_new
            if 'node_feat' in graph_data and 'edge_feat' in graph_data:
                edge_attr, edge_index, x = graph_data['edge_feat'].long(), graph_data['edge_index'].long(), graph_data['node_feat'].long()
            elif hasattr(graph_data, 'x') and hasattr(graph_data, 'edge_attr'):
                edge_attr, edge_index, x = graph_data.edge_attr.long(), graph_data.edge_index.long(), graph_data.x.long()
            else:
                raise ValueError('graph_data does not have expected keys or properties')

            graph_data = Data(x, edge_index, edge_attr)
            # graph_data={'x':torch.tensor(graph_data['node_feat']).long(),
            #             'edge_index':torch.tensor(graph_data['edge_index']).long(),
            #             'edge_attr':torch.tensor(graph_data['edge_feat']).long()}
            graph_batch.append({'graph': graph_data})
            # text_batch.append({'input_ids': torch.tensor(example_data['input_ids']).long(),
            #         'attention_mask': torch.tensor(example_data['attention_mask']).long(),
            #                    'answer_mask':torch.tensor(example_data['answer_mask']).long()})
            text_batch.append({'input_ids': example_data['input_ids'],
                    'attention_mask': example_data['attention_mask'],
                               })
            labels_batch.append({'labels':example_data['labels']})

        graph_batch = basic_collate(graph_batch)
        # text_batch = self.tokenizer.pad(text_batch, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        text_batch = padding(text_batch, self.tokenizer.pad_token_id,self.tokenizer.pad_token_type_id,  pad_to_multiple_of=self.pad_to_multiple_of)
        labels_batch = padding(labels_batch, self.tokenizer.pad_token_id, self.tokenizer.pad_token_type_id,
                             pad_to_multiple_of=self.pad_to_multiple_of)

        graph_batch=Data(**{'x':graph_batch['graph']['x'],'edge_index':graph_batch['graph']['edge_index'],'edge_attr':graph_batch['graph']['edge_attr'],'batch':graph_batch['graph']['batch'],'ptr':graph_batch['graph']['ptr']})
        batch={'graph': graph_batch,
            'input_ids': text_batch.data['input_ids'],
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



class CollatorForGNN(Collater):
    def __init__(self,**kwargs):
        self.transform_in_collator= kwargs.pop('transform_in_collator')
        super().__init__([],[])
