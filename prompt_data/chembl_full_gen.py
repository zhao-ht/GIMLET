import pickle
from rdkit.Chem import AllChem
from tqdm import tqdm


if __name__ == '__main__':
    dataPathSave = './chembl_raw/raw/'

    f = open(dataPathSave + 'labelsHard.pckl', 'rb')
    targetMat = pickle.load(f)
    sampleAnnInd = pickle.load(f)
    targetAnnInd = pickle.load(f)
    f.close()

    index_list = targetAnnInd.index.tolist()

    fin = open('chembl_full/assay_id.tsv', 'w')
    for idx, chembl_idx in enumerate(index_list):
        print('{}\t{}'.format(chembl_idx, idx), file=fin)

    ########## Double-check smiles and rdkit ##########
    root_path = './chembl_full'
    fout = open('{}/smiles.csv'.format(root_path), 'r')
    smiles_list = []
    for line in fout:
        smiles_list.append(line.strip())

    f = open('{}/rdkit_molecule.pkl'.format(root_path), 'rb')
    rdkit_mol_list = pickle.load(f)

    for i, (smiles, rdkit_mol) in enumerate(zip(smiles_list, rdkit_mol_list)):
        s = AllChem.MolToSmiles(rdkit_mol)
        print(i, '\t', smiles, '\t', rdkit_mol)
        assert smiles == s
