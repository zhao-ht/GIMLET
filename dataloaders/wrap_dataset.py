import torch



# class WrapDataset(torch.utils.data.Dataset):
#     def __init__(self, data, transform=None):
#         self.data=data
#         self.transform=transform
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         examples=self.data[idx]
#         if self.transform:
#             examples = self.transform(examples)
#         return examples


import torch.utils.data
import lmdb
import pickle
from io import BytesIO
from tqdm import tqdm
import os

class WrapDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, lmdb_path='cache/lmdb_cache'):
        self.data = data
        self.transform = transform
        self.lmdb_path = lmdb_path
        self.env = None

        # if self.lmdb_path is not None:
        #     self.env = lmdb.open(
        #         self.lmdb_path,
        #         map_size=1024 ** 4,  # 1TB maximum database size
        #         subdir=True,
        #         readonly=False,
        #         lock=True,
        #         meminit=False
        #     )
        #     if not os.path.exists(self.lmdb_path):
        #         os.makedirs(self.lmdb_path)
        #     # Create LMDB environment
        #     # Cache all transformed data into LMDB
        #         with self.env.begin(write=True) as txn:
        #             for i in tqdm(range(len(self.data))):
        #                 key = str(i).encode('ascii')
        #                 data_graph = self.transform(self.data[i])['graph'] if self.transform is not None else self.data[i]
        #                 buf = BytesIO()
        #                 pickle.dump(data_graph, buf)
        #                 txn.put(key, buf.getvalue())
        self.cached=[]
        for i in tqdm(range(len(self.data))):
            self.cached.append(self.transform(self.data[i])['graph'])



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # if self.env is not None:
        #     # Retrieve transformed data from LMDB
        #     with self.env.begin() as txn:
        #         key = str(idx).encode('ascii')
        #         value = txn.get(key)
        #         if value is None:
        #             raise IndexError('Index out of range')
        #         buf = BytesIO(value)
        #         data_graph = pickle.load(buf)
        #     data=self.data[idx]
        #     data['graph']=data_graph
        # else:
        #     # Transform data on the fly
        #     data = self.transform(self.data[idx]) if self.transform is not None else self.data[idx]

        data_graph = self.cached[idx]
        data=self.data[idx]
        data['graph']=data_graph

        return data