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

            graph_batch.append({'graph': graph_data})

            text_batch.append({'input_ids': example_data['input_ids'],
                    'attention_mask': example_data['attention_mask'],
                               })
            labels_batch.append({'labels':example_data['labels']})

        graph_batch = basic_collate(graph_batch)

        text_batch = padding(text_batch, self.tokenizer.pad_token_id,self.tokenizer.pad_token_type_id,  pad_to_multiple_of=self.pad_to_multiple_of)
        labels_batch = padding(labels_batch, self.tokenizer.pad_token_id, self.tokenizer.pad_token_type_id,
                             pad_to_multiple_of=self.pad_to_multiple_of)

        graph_batch=Data(**{'x':graph_batch['graph']['x'],'edge_index':graph_batch['graph']['edge_index'],'edge_attr':graph_batch['graph']['edge_attr'],'batch':graph_batch['graph']['batch'],'ptr':graph_batch['graph']['ptr']})
        batch={'graph': graph_batch,
            'input_ids': text_batch.data['input_ids'],
            'attention_mask': text_batch.data['attention_mask'],
               'labels':labels_batch.data['labels']}

        special_tokens_mask = batch.pop("special_tokens_mask", None)

        return batch



class CollatorForGNN(Collater):
    def __init__(self,**kwargs):
        self.transform_in_collator= kwargs.pop('transform_in_collator')
        super().__init__([],[])
