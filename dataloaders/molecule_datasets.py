import os

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_gz)
# from torch_geometric.datasets import MoleculeNet
from ogb.utils import smiles2graph
import re
import os.path as osp
from torch_geometric.datasets.molecule_net import x_map,e_map


#origin moleculenet from torch_geometric; we add cyp450 dataset

class MyMoleculeNet(InMemoryDataset):
    r"""The `MoleculeNet <http://moleculenet.ai/datasets-1>`_ benchmark
    collection  from the `"MoleculeNet: A Benchmark for Molecular Machine
    Learning" <https://arxiv.org/abs/1703.00564>`_ paper, containing datasets
    from physical chemistry, biophysics and physiology.
    All datasets come with the additional node and edge features introduced by
    the `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"ESOL"`,
            :obj:`"FreeSolv"`, :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`,
            :obj:`"HIV"`, :obj:`"BACE"`, :obj:`"BBPB"`, :obj:`"Tox21"`,
            :obj:`"ToxCast"`, :obj:`"SIDER"`, :obj:`"ClinTox"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'

    # Format: name: [display_name, url_name, csv_name, smiles_idx, y_idx]
    names = {
        'esol': ['ESOL', 'delaney-processed.csv', 'delaney-processed', -1, -2],
        'freesolv': ['FreeSolv', 'SAMPL.csv', 'SAMPL', 1, 2],
        'lipo': ['Lipophilicity', 'Lipophilicity.csv', 'Lipophilicity', 2, 1],
        'pcba': ['PCBA', 'pcba.csv.gz', 'pcba', -1,
                 slice(0, 128)],
        'muv': ['MUV', 'muv.csv.gz', 'muv', -1,
                slice(0, 17)],
        'hiv': ['HIV', 'HIV.csv', 'HIV', 0, -1],
        'bace': ['BACE', 'bace.csv', 'bace', 0, 2],
        'bbbp': ['BBPB', 'BBBP.csv', 'BBBP', -1, -2],
        'tox21': ['Tox21', 'tox21.csv.gz', 'tox21', -1,
                  slice(0, 12)],
        'toxcast':
        ['ToxCast', 'toxcast_data.csv.gz', 'toxcast_data', 0,
         slice(1, 618)],
        'sider': ['SIDER', 'sider.csv.gz', 'sider', 0,
                  slice(1, 28)],
        'clintox': ['ClinTox', 'clintox.csv.gz', 'clintox', 0,
                    slice(1, 3)],
        'cyp450':  ['CYP450', 'CYP450.csv.gz', 'CYP450', -1,
                      slice(2,7)]
    }

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):

        if Chem is None:
            raise ImportError('`MoleculeNet` requires `rdkit`.')

        self.name = name.lower()
        assert self.name in self.names.keys()
        super(MyMoleculeNet, self).__init__(root, transform, pre_transform,
                                          pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line in dataset:
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')

            smiles = line[self.names[self.name][3]]
            ys = line[self.names[self.name][4]]
            ys = ys if isinstance(ys, list) else [ys]

            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            xs = []
            for atom in mol.GetAtoms():
                x = []
                x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
                x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
                x.append(x_map['degree'].index(atom.GetTotalDegree()))
                x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
                x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
                x.append(x_map['num_radical_electrons'].index(
                    atom.GetNumRadicalElectrons()))
                x.append(x_map['hybridization'].index(
                    str(atom.GetHybridization())))
                x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
                x.append(x_map['is_in_ring'].index(atom.IsInRing()))
                xs.append(x)

            x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                e = []
                e.append(e_map['bond_type'].index(str(bond.GetBondType())))
                e.append(e_map['stereo'].index(str(bond.GetStereo())))
                e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

                edge_indices += [[i, j], [j, i]]
                edge_attrs += [e, e]

            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        smiles=smiles)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))





# Original torch_geometric dataset use different tokenizer to graph, so we need rewrite the preprocess with ogb smiles2graph function
class MoleculeDatasetRich(MyMoleculeNet):
    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None,return_id=False,return_smiles=False,recache_preprocess=False,rich_features=True):

        super(MoleculeDatasetRich, self).__init__(root, name, transform, pre_transform,
                 pre_filter)
        self.return_id=return_id
        self.return_smiles=return_smiles
        self.rich_features=rich_features


        if recache_preprocess:
            os.makedirs(self.processed_dir)
            self.process()

    def get(self, idx):
        data=super(MoleculeDatasetRich,self).get(idx)
        # data['y']=data['y']*2-1
        data['y'][data['y'].isnan()]=-100
        # if self.name=='esol':
        #     data['y']=-data['y']
        if self.return_id:
            data['id']=idx
        if not self.return_smiles:
            data.__delattr__('smiles')
        if not self.rich_features:
            data['x']=data['x'][:,0:3]
            data['edge_attr'] = data['edge_attr'][:,0:2]
        return data


    def download(self):
        if self.name!='cyp450':
            super(MoleculeDatasetRich,self).download()
        else:
            pass

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        data_smiles_list = []
        for line in dataset:
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')

            smiles = line[self.names[self.name][3]]

            try:
                graph=smiles2graph(smiles)
            except:
                print('invalid smiles ',smiles)
                continue

            ys = line[self.names[self.name][4]]
            ys = ys if isinstance(ys, list) else [ys]

            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)


            data = Data(x=torch.tensor(graph['node_feat']), edge_index=torch.tensor(graph['edge_index']), edge_attr=torch.tensor(graph['edge_feat']), y=y,
                        smiles=smiles)
            data_smiles_list.append(smiles)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = os.path.join(self.processed_dir, 'smiles.csv')
        data_smiles_series.to_csv(saver_path, index=False, header=False)


    @property
    def processed_file_names(self):
        return 'rich_data_ogb_encoding.pt'



class MoleculeDatasetSplitLabel(MoleculeDatasetRich):
    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None,return_id=False,return_smiles=False,split_label=False,single_split=None,recache_preprocess=False):

        super(MoleculeDatasetSplitLabel, self).__init__(root, name, transform, pre_transform,
                 pre_filter,return_id=return_id,return_smiles=return_smiles,recache_preprocess=recache_preprocess)
        # if split_label:
        self.split_label=split_label
        self.single_split=single_split

        if split_label and single_split is None:
            assert self.data.y.ndim==2
            self.label_number=self.data.y.shape[1]
        else:
            self.label_number=1


    def get(self, idx):
        # if split label, the first #data datas will return data with the first label, and next #data datas will return data with the second label……
        if self.split_label:
            idx_data=idx%self.len_oridata()
        else:
            idx_data=idx

        if self.split_label:
            idx_label = idx // self.len_oridata()
        if self.single_split is not None:
            idx_label = self.single_split

        data=super(MoleculeDatasetSplitLabel,self).get(idx)

        if self.split_label or self.single_split is not None:
            data['y'] = data['y'][:, idx_label] if (data['y'].ndim == 2 and data['y'].shape[0] == 1) else data['y'][
                idx_label]

        if self.split_label or self.single_split is not None:
            data['y_idx']=torch.tensor([idx_label])

        return data


    def download(self):
        super(MoleculeDatasetSplitLabel,self).download()

    def process(self):
        super(MoleculeDatasetSplitLabel,self).process()

    # Result of self.__len__ changes after Subset selection because self.__len__ return the length of indices

    def len(self) -> int: # Result of this function do not change after Subset selection, becuse self.slices do not change
        for item in self.slices.values():
            return (len(item) - 1)*self.label_number
        return 0

    def len_oridata(self) -> int: # Result of this function do not change after Subset selection, becuse self.slices do not change
        for item in self.slices.values():
            return len(item) - 1
        return 0

    def len_data(self): # Result of this function changes after Subset selection because self.__len__() return the length of indices
        assert len(self)%self.label_number == 0
        return len(self)//self.label_number

    def set_single_split(self,single_split):
        self.single_split = single_split