# Step 1: Download dataset

```bash
mkdir -p ./datasets
cd datasets
wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced.zip
unzip dataPythonReduced.zip
cd dataPythonReduced
wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Smiles.pckl
wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20LSTM.pckl
cd ..
rm dataPythonReduced.zip
mkdir -p chembl_raw/raw
mv dataPythonReduced/* chembl_raw/raw
wget 'https://www.dropbox.com/s/vi084g0ol06wzkt/mol_cluster.csv?dl=1'
mv 'mol_cluster.csv?dl=1' chembl_raw/raw/mol_cluster.csv
```

#### Task description
Target prediction: predict a binary assay outcome, that indicates whether a certain compound, for example, binds to a specific receptor, inhibits some pathway or induces toxic effects.
#### Contents of files in ```chembl_raw/raw```
+ ```chembl20LSTM.pckl```: A ```list``` of 456331 ```rdchem.Mol``` objects, representing the compounds.
+ ```chembl20Smiles.pckl```: A ```list``` of 456331 SMILES ```str```s of the compounds.
+ ```folds0.pckl```: A ```list``` of 3 ```ndarray```s, containing the indices (from 0 to 456330) for train/valid/test sets.
+ ```mol_cluster.csv```: A csv file storing the CHEMBL ID, cluster ID, ID in the raw LSC dataset (rawID), fold ID, ID in the reduced LSC dataset (filteredID)
+ ```labelsHard.pckl```: Contains three python objects described below. Labels are assigned using hard threshold.
  +  ```targetMat```: A 456331 x 1310 sparse matrix, representing the drug-target matrix.
  +  ```sampleAnnInd```: An ```int64 Series``` of length 456331, containing the indices of the compounds. This is in fact redundant, as its index and values are both ```0, 1, ..., 456330```.
  +  ```targetAnnInd```: An ```int64 Series``` of length 1310 containing column names for ```targetMat```, indexed by [the ChEMBL id](http://chembl.blogspot.com/2011/08/chembl-identifiers.html) of the target and containing the indices of the target ```0, 1, ..., 1310```.


# Step 2: Transform and ChEMBL

```bash
python transform.py --input-dir chembl_raw/raw --output-dir chembl_full > transform.out
```
Preprocess the raw dataset, and transform it into more friendly and organized formats.

1. Discard the 22 ```None```s in the compound list.
2. Filter out the 9 molecules with ≤ 2 non-H atoms.
3. Retain only the largest molecule in the SMILES string. E.g. if the compound is a organic chlorium salt, say CH3NH3+Cl-, we retain only the organic compound after removing HCL, in this case CH3NH2.
4. Filter out molecules with molecular weight < 50 (929) or > 900 (9).

#### Contents of the output files
+ ```assay_id.tsv```: a table storing ChEMBL indices for the assays
+ ```labels.npz```: a numpy compressed file storing the dense 455362 x 1310 drug-target matrix.
+ ```rdkit_molecule.pkl```: a ```list``` of 455362 ```rdchem.Mol``` objects.
+ ```mol_properties.csv```: a table storing filteredID, chembl, clusterID,rawID, fold, smiles, and scaffold_smiles for the filtered molecules.

# Step 3: Mapping ChEMBL To STRING

```bash
cd ChEMBL_STRING
python step_01.py --dataset chembl_raw > step_01.out
python step_02.py
python step_03.py --dataset chembl_raw
python step_04.py --dataset chembl_raw
python step_05.py --dataset chembl_raw
cp filtered_task_score.tsv ../chembl_full/
cd ..
```
+ step 1 generates ```assay2target.tsv```, a 1310 x 6 tsv file storing the assay_id, target_id, target_name, organism and uniprot_list for the assays.
  + We are using this comand `https://www.ebi.ac.uk/chembl/api/data/assay` to get the assay information from ChEMBL. By default, this will use the latest ChEMBL version, which was ChEMBL-27 for this project. We also save a copy on the [google drive](https://drive.google.com/drive/folders/14DpCynww2NEroKV2W3g0F-tXQpV2-PhC?usp=sharing).
+ step 2 generates ```uniprot2string.tsv```, storing 2630 uniprot and its corresponding STRING ID. 40 uniprots without available STRING IDs are stored in ```uniprot_without_stringid.txt```. This step also creates ```string_ppi_score.tsv```, which stores the STRING protein-protein interaction score for 6178 protein (represented as STRING ID) pairs.
+ step 3 generates ```filtered_assay_score.tsv```, which stores 9172 non-zero scores for assay pairs (represented by CHEMBL ID). The assay score is defined as $\mathrm{score}\,(A_i, A_j) = \max \{ \mathrm{score}\,(p_i, p_j) | p_i\in A_i, p_j\in A_j \}$, where $A$ denotes assay and $p$ denotes protein.
+ step 4 generates ```filtered_task_score.tsv```, which is identical to ```filtered_assay_score.tsv``` except each assay is represented by an integer from 0 to 1310.
+ step 5 verifies the task/assay ordering.

# Step 4: Filtering Based on ChEMBL-STRING
Construct the `chembl_connected` dataset, which only contains tasks appearing in the KG.

+ ChEMBL-Dense-{10, 50, 100}:
```bash
bash chembl_gen.sh
mkdir -p chembl_dense_10/processed
mkdir -p chembl_dense_50/processed
mkdir -p chembl_dense_100/processed
cp -r chembl_dense_10 chembl_dense_10_ablation
cp -r chembl_dense_50 chembl_dense_50_ablation
cp -r chembl_dense_100 chembl_dense_100_ablation
```
This script filters the ```CHEMBL``` dataset by the degree of drug and assays.
#### Shape of the filtered datasets
+ chembl_dense_10: (13343, 390)
+ chembl_dense_50: (932, 152)
+ chembl_dense_100: (518, 132)
