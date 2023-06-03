import os
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from tqdm import tqdm
from multiprocessing import Pool
import argparse
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


def transform(input_dir, output_dir, n_proc):
    """
    Data from 'Large-scale comparison of machine learning methods for drug target prediction on ChEMBL'
    :param input_dir: path to the folder containing the reduced chembl dataset
    """
    # adapted from https://github.com/ml-jku/lsc/blob/master/pythonCode/lstm/loadData.py
    # first need to download the files and unzip:
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced.zip
    # unzip and rename to chembl_with_labels
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Smiles.pckl
    # into the dataPythonReduced directory
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20LSTM.pckl

    # 1. load folds and labels
    mol_info = pd.read_csv(os.path.join(input_dir, 'mol_cluster.csv'), header=0, index_col='filteredID')
    print(mol_info.head())
    with open(os.path.join(input_dir, 'labelsHard.pckl'), 'rb') as f:
        targetMat=pickle.load(f)  # possible values are {-1, 0, 1}
        sampleAnnInd=pickle.load(f)
        targetAnnInd=pickle.load(f)

    targetMat.sort_indices()
    targetAnnInd.to_csv(os.path.join(output_dir, 'assay_id.tsv'), sep='\t', header=None)

    # 2. load and filter molecules
    with open(os.path.join(input_dir, 'chembl20LSTM.pckl'), 'rb') as f:
        rdkitArr = pickle.load(f)

    assert len(rdkitArr) == targetMat.shape[0] == mol_info.shape[0]

    with Pool(n_proc) as p:
        filter_result = p.starmap(filter_mol, tqdm(enumerate(rdkitArr), total=len(rdkitArr)))
    filtered_mol_statistics = [elem for elem in filter_result if not isinstance(elem, tuple)]
    print("==", np.unique(filtered_mol_statistics, return_counts=True))
    filter_result = [elem for elem in filter_result if isinstance(elem, tuple)]
    valid_indices, molecule_list, smiles_list, scaffold_smiles_list = zip(*filter_result)

    valid_indices = np.array(valid_indices)

    mol_info = mol_info.loc[valid_indices]
    mol_info['smiles'] = smiles_list
    mol_info['scaffold_smiles'] = scaffold_smiles_list
    mol_info.to_csv(os.path.join(output_dir, 'mol_properties.csv'))

    labels = targetMat.A[valid_indices]
    np.savez_compressed(os.path.join(output_dir, 'labels.npz'), labels=labels)

    with open(os.path.join(output_dir, 'rdkit_molecule.pkl'), 'wb') as f:
        pickle.dump(molecule_list, f)

    print('{} out of {} are valid'.format(len(valid_indices), len(rdkitArr)))
    print('shape of labels: {}'.format(labels.shape))
    print(mol_info.head())


def filter_mol(i, mol):
    if mol is None:
        return -1  # 22
    mol_species_list = split_rdkit_mol_obj(mol)
    # 20119 mols would have > 1 species
    largest_mol = get_largest_mol(mol_species_list)
    if len(largest_mol.GetAtoms()) <= 2:
        return -2  # 9
    mw = Descriptors.MolWt(largest_mol)
    if mw < 50:
        return -3  # 929
    if mw > 900:
        return -4  # 9
    smiles = AllChem.MolToSmiles(largest_mol)
    # if not AllChem.MolFromInchi(AllChem.MolToInchi(largest_mol)):
    #     return -5  # 11
    return (i, largest_mol, smiles, generate_scaffold(smiles))


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if AllChem.MolFromSmiles(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default=os.path.join('chembl_raw', 'raw'), help='dir path to the raw dataset')
    parser.add_argument('--output-dir', type=str, default='chembl_full', help='dir path to the transformed dataset')
    parser.add_argument('--n-proc', type=int, default=8, help='number of processes to run when downloading assay & target information')
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    transform(args.input_dir, args.output_dir, args.n_proc)