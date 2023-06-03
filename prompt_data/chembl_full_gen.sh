mkdir -p chembl_full
cp ChEMBL/smiles.csv chembl_full/
cp ChEMBL/rdkit_molecule.pkl chembl_full/
cp ChEMBL/labels.npz chembl_full/
cp ChEMBL/fold*txt chembl_full/
cp ChEMBL_STRING/filtered_task_score.tsv chembl_full/
python chembl_full_gen.py # get mapping from assay id (ChEMBL id) to id
