import torch
from torch_geometric.data import Dataset

from .utils.GraphAug import drop_nodes, permute_edges, subgraph, mask_nodes
from copy import deepcopy
import numpy as np
import os
import random
from transformers import BertTokenizer

class GINMatchDataset(Dataset):
    def __init__(self, root, args):
        super(GINMatchDataset, self).__init__(root)
        self.root = root
        self.graph_aug = args.graph_aug
        self.text_max_len = args.text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        self.data_type = args.data_type

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)

        data_aug = self.augment(data_graph, self.graph_aug)
        text_path = os.path.join(self.root, 'text', text_name)

        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line.strip('\n')
            text_list.append(line)
            if count > 500:
                break
        
        text = mask = None

        if self.data_type == 0: # paragraph-level
            text, mask = self.tokenizer_text(text_list[0][:256])
        
        if self.data_type == 1: #random sentence
            sts = text_list[0].split('.')
            remove_list = []
            for st in (sts):
                if len(st.split(' ')) < 5: 
                    remove_list.append(st)
            remove_list = sorted(remove_list, key=len, reverse=False)
            for r in remove_list:
                if len(sts) > 1:
                    sts.remove(r)
            text_index = random.randint(0, len(sts)-1)
            text, mask = self.tokenizer_text(sts[text_index])

        return data_aug, text.squeeze(0), mask.squeeze(0)#, index

    def augment(self, data, graph_aug):

        if graph_aug == 'noaug':
            data_aug = deepcopy(data)
        elif graph_aug == 'dnodes':
            data_aug = drop_nodes(deepcopy(data))
        elif graph_aug == 'pedges':
            data_aug = permute_edges(deepcopy(data))
        elif graph_aug == 'subgraph':
            data_aug = subgraph(deepcopy(data))
        elif graph_aug == 'mask_nodes':
            data_aug = mask_nodes(deepcopy(data))
        elif graph_aug == 'random2':  # choose one from two augmentations
            n = np.random.randint(2)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False
        elif graph_aug == 'random3':  # choose one from three augmentations
            n = np.random.randint(3)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False
        elif graph_aug == 'random4':  # choose one from four augmentations
            n = np.random.randint(4)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            elif n == 3:
                data_aug = mask_nodes(deepcopy(data))
            else:
                print('sample error')
                assert False
        else:
            data_aug = deepcopy(data)
            data_aug.x = torch.ones((data.edge_index.max()+1, 1))

        return data_aug

    def tokenizer_text(self, text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.text_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask