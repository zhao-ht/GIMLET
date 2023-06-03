import pickle
import os
import argparse
from tqdm import tqdm
import pandas as pd
from ogb.utils import smiles2graph
import commentjson
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='chembl_full')
parser.add_argument('--dataset', type=str, default='chembl_raw')

parser.add_argument('--generate_mole_text',action='store_true')
parser.add_argument('--generate_assay_text',action='store_true')

parser.add_argument('--split_non_overlap',action='store_true')

parser.add_argument('--use_augmented_prompt',action='store_true')
parser.add_argument('--augmented_type',type=str,default=None)

parser.add_argument('--add_negation',action='store_true')
parser.add_argument('--negation_p',type=float,default=0.5)

parser.add_argument('--split_file',type=str,default='split_0')
parser.add_argument('--file_save_name',type=str,default='.csv')


args = parser.parse_args()

if args.split_non_overlap:
    args.split_file = 'non_overlap_' + args.split_file
else:
    args.split_file += 'with_molecule_overlap_' + args.split_file

map_dict={'N':'No','Y':'Yes'}

def is_decimal_zero(number):
    if isinstance(number,int):
        return True
    elif isinstance(number,float):
        return number % 1 == 0
    else:
        return False


def map_label(label,dtype):
    if dtype=='int':
        return str(int(label))
    elif dtype=='float':
        return str(round(label, 2))
    elif dtype=='str':
        if label in map_dict:
            return map_dict[label]
        else:
            return label
    else:
        raise ValueError('not supported dtype: {}'.format(dtype))


def map_negation(label):
    if isinstance(label,int) or isinstance(label,float):
        return -label
    else:
        try:
            label = eval(label)
            if is_decimal_zero(label):
                return str(-int(label))
            else:
                return str(round(-label,2))
        except:
            assert label in {'No','Yes'}
            return {'No':'Yes','Yes':'No'}[label]

map_dict_graph_only={'N':0,'Y':1}
def map_label_graph_only(label):
    if label in map_dict_graph_only:
        return map_dict_graph_only[label]
    elif isinstance(label,str):
        return eval(label)
    else:
        return label


def load_split_file(mol_smiles,assays):
    split_file_dir = os.path.join('..', 'pretrain_datasets', args.split_file)

    if args.split_file is not None and os.path.exists(split_file_dir):
        print('loading existing ' + split_file_dir)

        df = pd.read_csv(os.path.join(split_file_dir, 'train_assays.csv'))
        train_assays = df['values'].values.tolist()
        df = pd.read_csv(os.path.join(split_file_dir, 'valid_assays.csv'))
        valid_assays = df['values'].values.tolist()
        df = pd.read_csv(os.path.join(split_file_dir, 'train_graph_index.csv'))
        train_graph_index = df['values'].values.tolist()
        df = pd.read_csv(os.path.join(split_file_dir, 'valid_graph_index.csv'))
        valid_graph_index = df['values'].values.tolist()

    else:
        print('Warning! Creating new split')
        train_index = np.random.choice(np.arange(len(assays)), size=int(len(assays) * 0.8), replace=False)
        valid_index = np.delete(np.arange(len(assays)), train_index)

        train_assays = [assays[i] for i in train_index]
        valid_assays = [assays[i] for i in valid_index]

        if args.split_non_overlap:
            train_graph_index = np.random.choice(np.arange(len(mol_smiles)), size=int(len(mol_smiles) * 0.8),
                                                 replace=False)
            valid_graph_index = np.delete(np.arange(len(mol_smiles)), train_graph_index)
        else:
            train_graph_index = np.arange(len(mol_smiles))
            valid_graph_index = np.arange(len(mol_smiles))
        train_graph_index = train_graph_index.tolist()
        valid_graph_index = valid_graph_index.tolist()
        split_index = {}
        split_index['train_assays'] = train_assays
        split_index['valid_assays'] = valid_assays
        split_index['train_graph_index'] = train_graph_index
        split_index['valid_graph_index'] = valid_graph_index
        os.mkdir(split_file_dir)
        for key in split_index:
            df = pd.DataFrame(split_index[key], columns=['values'])
            df.to_csv(os.path.join(split_file_dir, key + '.csv'))
    return train_graph_index, valid_graph_index, train_assays, valid_assays


def generate_assay_data(assay_ids, molecule_index_set,split_name):
    data_all = []
    for _, assay_chembl_id in tqdm(enumerate(assay_ids)):

        if args.augmented_type is not None:
            type = args.augmented_type
            text_data = prompts_augmented['chembl'][assay_chembl_id][type][
                np.random.randint(len(prompts_augmented['chembl'][assay_chembl_id][type]))]
        else:
            text_data = prompts['chembl'][assay_chembl_id][np.random.randint(len(prompts['chembl'][assay_chembl_id]))]

        for graph_id in set(
                targetMat[:, targetAnnInd[assay_chembl_id]].nonzero()[0]) & molecule_index_set & valid_graph_ids:

            label = targetMat[graph_id, targetAnnInd[assay_chembl_id]]
            assert label != 0
            label = 'Yes' if label > 0 else 'No'

            if graph_id in mol_prop['chembl']:
                graph_data = mol_smiles[graph_id]

                chemid = mol_prop['chembl'][graph_id]
                data_all += [(graph_data, text_data, label, chemid, assay_chembl_id)]

    print(len(data_all))

    file_prefix = "assay_graph_text_"
    if args.augmented_type is not None:
        file_prefix += args.augmented_type + '_'
    df = pd.DataFrame(data_all,
                      columns=['graph', 'text', 'label', 'chemid', 'assayid'
                               ])
    df.to_csv(os.path.join('..', 'pretrain_datasets', file_prefix + split_name+'_' + args.split_file + args.file_save_name))


def generate_property_data(molecule_index_set,split_name):

    data_all = []
    for _, graph_id in tqdm(enumerate(molecule_index_set)):

        if graph_id in mol_prop['chembl']:
            graph_chembl_id=mol_prop['chembl'][graph_id]

            if graph_chembl_id in data_mol_property.index:
                properties = data_mol_property.loc[graph_chembl_id]
                graph_data = properties['graph']
                for key in data_mol_property.columns:
                    if not (key in [] or pd.isnull((properties[key])) or properties[key] in ['[]', []] or (
                            properties[key] is None)):

                        key_prompt = 'Molecule_molecules_molecule_properties_' + key
                        if args.add_negation:
                            add_negation = (np.random.rand() < args.negation_p) & (
                                        key_prompt + '_negative' in prompts['chembl_property'])
                        else:
                            add_negation = False

                        if add_negation:
                            key_prompt += '_negative'

                        if key_prompt in prompts['chembl_property']:
                            if args.augmented_type is not None:
                                text_prompts = prompts_augmented['chembl_property'][key_prompt][args.augmented_type]
                                prompt_id = np.random.randint(0, len(text_prompts))
                                text_prompt = text_prompts[prompt_id]
                            else:
                                text_prompts = prompts['chembl_property'][key_prompt]
                                prompt_id = np.random.randint(0, len(text_prompts))
                                text_prompt = text_prompts[prompt_id]
                            label = map_label(properties[key], column_type[key])
                            if add_negation:
                                label = map_negation(label)
                            data_all.append((graph_data, text_prompt, label, graph_chembl_id))


    print(len(data_all))

    df = pd.DataFrame(data_all,
                      columns=['graph', 'text', 'label', 'chemid'
                               ])
    file_prefix = "property_graph_text_"
    if args.add_negation:
        file_prefix += 'negative' + str(args.negation_p).replace('.', '') + '_'
    if args.augmented_type is not None:
        file_prefix += args.augmented_type + '_'
    df.to_csv(os.path.join('..', 'pretrain_datasets', file_prefix + split_name+ '_' + args.split_file + args.file_save_name))

if __name__ == '__main__':

    root = '../prompt_data/{}/raw/'.format(args.dataset)
    with open(os.path.join(root, 'labelsHard.pckl'), 'rb') as f:
        targetMat = pickle.load(f)
        sampleAnnInd = pickle.load(f)
        targetAnnInd = pickle.load(f)

    assay_ids = list(targetAnnInd.index)

    mol_prop = pd.read_csv(os.path.join('..','prompt_data',args.input_dir, 'mol_properties.csv'), index_col=0)
    print('{} molecules with SMILES'.format(mol_prop.shape[0]))

    with open(os.path.join('../prompt_data/{}/raw/'.format(args.dataset), 'chembl20Smiles.pckl'), 'rb') as f:
        mol_smiles = pickle.load(f)

    if not os.path.exists("../pretrain_datasets"):
        os.makedirs("../pretrain_datasets")

    with open("prompt_pretrain.json", 'r') as load_f:
        prompts = commentjson.load(load_f)
    with open("augmented_prompt_pretrain.json", 'r') as load_f:
        prompts_augmented = commentjson.load(load_f)

    assays = list(prompts['chembl'].keys())

    train_graph_index, valid_graph_index, train_assays, valid_assays = load_split_file(mol_smiles,assays)



    if args.generate_assay_text:
        if os.path.exists('../pretrain_datasets/valid_graph_ids.csv'):
            print('loading existing valid_graph_ids.csv')
            valid_graph_ids=pd.read_csv('../pretrain_datasets/valid_graph_ids.csv')

        else:
            valid_graph_ids=[]
            for graph_id in tqdm(range(len(mol_smiles))):
                graph_data = mol_smiles[graph_id]

                try:
                    graph = smiles2graph(graph_data)
                    if graph['node_feat'].shape[0] == 0 or graph['edge_feat'].shape[0] == 0 or \
                            graph['edge_index'].shape[1] == 0:
                        print('empty graph', graph)
                        continue
                    valid_graph_ids.append(graph_id)
                except:
                    print('graph data error ', graph_data)
                    continue

            valid_graph_ids=pd.DataFrame(valid_graph_ids,columns=['graph_ind'])
            valid_graph_ids.to_csv('../pretrain_datasets/valid_graph_ids.csv')

        valid_graph_ids=set(valid_graph_ids['graph_ind'].values.tolist())
        print('number of valid graphs: ', len(valid_graph_ids))

         # a few assay do not have instruction, so we only use the assies with instructions

        generate_assay_data(train_assays,set(train_graph_index),'train')
        generate_assay_data(valid_assays,set(valid_graph_index),'valid')

        if args.use_augmented_prompt:
            for type in ['rewrite','expand','detail','shorten']:
                args.augmented_type=type
                generate_assay_data(train_assays,set(train_graph_index),'train')
                generate_assay_data(valid_assays,set(valid_graph_index),'valid')
            args.augmented_type=None

    if args.generate_mole_text:

        data_mol_property=pd.read_csv(os.path.join('..','prompt_data','mole_graph_property.csv'),header=0)
        data_mol_property=data_mol_property.set_index('graph_chembl_id')
        column_type={}

        for column in data_mol_property.columns:
            non_nan_values = data_mol_property[column][~data_mol_property[column].isna()]
            if all(non_nan_values.apply(is_decimal_zero)):
                column_type[column]= 'int'
            elif all(isinstance(value, float) for value in non_nan_values):
                column_type[column] = 'float'
            else:
                column_type[column] = 'str'

        generate_property_data(set(train_graph_index),'train')
        generate_property_data(set(valid_graph_index),'valid')

        if args.use_augmented_prompt:
            for type in ['rewrite','expand','detail','shorten']:
                args.augmented_type=type
                generate_property_data(set(train_graph_index), 'train')
                generate_property_data(set(valid_graph_index), 'valid')
            args.augmented_type=None


