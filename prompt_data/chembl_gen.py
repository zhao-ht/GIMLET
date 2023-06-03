import pickle
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='chembl_full', help='dir path to the raw dataset')
    parser.add_argument('--dataset', type=str, default='chembl_dense')
    parser.add_argument('--threshold', type=int, default=10)
    parser.add_argument('--threshold_pos_neg', type=int, default=10)
    args = parser.parse_args()
    root = '{}_{}'.format(args.dataset, args.threshold)

    if not os.path.exists(root):
        os.makedirs(root)

    # ------ Loading ------
    # Load labels and target index
    targetAnnInd = pd.read_csv(os.path.join(args.input_dir, 'assay_id.tsv'), header=None, sep='\t', index_col=0)[1]
    targetMat = np.load(os.path.join(args.input_dir, 'labels.npz'))['labels']    # Important! Some molecules have been filtered in transform.py

    # Load STRING
    task_score = pd.read_csv(os.path.join(args.input_dir, 'filtered_task_score.tsv'), header=None, sep='\t')
    task_score.columns = ['task1', 'task2', 'score']

    # Load molecule property data
    mol_prop = pd.read_csv(os.path.join(args.input_dir, 'mol_properties.csv'), index_col=0)
    print('{} molecules with SMILES'.format(mol_prop.shape[0]))

    # Load rdkit_molecule.pkl
    with open(os.path.join(args.input_dir, 'rdkit_molecule.pkl'), 'rb') as f:
        rdkit_mol_list = pickle.load(f)
    print('{} rdkit molecules'.format(len(rdkit_mol_list)))
    assert mol_prop.shape[0] == len(rdkit_mol_list)
    rdkit_mol_list = np.asarray(rdkit_mol_list)

    # ------ Filtering -------
    # Filter out assays not appearing in STRING
    scored_tasks = set(task_score.task1).union(task_score.task2)
    task_bool_idx = [taskID in scored_tasks for taskID in targetAnnInd.values]

    targetAnnInd = targetAnnInd[task_bool_idx]
    targetMat = targetMat[:, task_bool_idx]
    print('matrix shape after first pruning step:', targetMat.shape)

    # Filter out molecules and tasks whose degree are no more than a threshold
    lst_shape = (0, 0)
    while lst_shape != targetMat.shape:
        lst_shape = targetMat.shape
        
        # Filter out molecules
        mol_deg = np.sum(targetMat != 0, axis=1)
        mol_bool_idx = mol_deg >= args.threshold
        targetMat = targetMat[mol_bool_idx, :]
        mol_prop = mol_prop.loc[mol_bool_idx]
        rdkit_mol_list = rdkit_mol_list[mol_bool_idx]

        # Filter out tasks
        task_pos_deg = np.sum(targetMat > 0, axis=0)
        task_neg_deg = np.sum(targetMat < 0, axis=0)
        task_bool_idx = (task_neg_deg + task_pos_deg >= args.threshold) & \
            (task_pos_deg >= args.threshold_pos_neg) & (task_neg_deg >= args.threshold_pos_neg)
        targetMat = targetMat[:, task_bool_idx]
        targetAnnInd = targetAnnInd[task_bool_idx]

    # ------ Outputing ------
    # Output mol_properties.csv
    print('{} molecules with SMILES after pruing'.format(mol_prop.shape[0]))
    mol_prop.reset_index(inplace=True)
    mol_prop.to_csv(os.path.join(root, "mol_properties.csv"))

    # Output rdkit_molecule.pkl
    print('{} rdkit molecule list after pruning'.format(len(rdkit_mol_list)))
    rdkit_mol_list = rdkit_mol_list.tolist()
    with  open(os.path.join(root, "rdkit_molecule.pkl"), 'wb') as f:
        pickle.dump(rdkit_mol_list, f)

    # Output labels.npz
    print('matrix shape after second pruning step:', targetMat.shape)
    np.savez_compressed(os.path.join(root, "labels.npz"), labels=targetMat)

    # Output assay_id.tsv
    task_df = targetAnnInd.to_frame(name='oldID')
    task_df['newID'] = np.arange(task_df.shape[0])
    task_df.newID.to_csv(os.path.join(root, "assay_id.tsv"), sep='\t', header=None)
    print('{} tasks after pruning'.format(len(targetAnnInd)))

    # Output filtered_task_score.tsv
    task_oldID_to_newID = {oldID: newID for oldID, newID in zip(task_df.oldID, task_df.newID)}
    n_edges = 0
    with open(os.path.join(root, "filtered_task_score.tsv"), 'w') as f:
        for rowID, row in task_score.iterrows():
            if row.task1 in task_oldID_to_newID and row.task2 in task_oldID_to_newID:
                f.write('{}\t{}\t{}\n'.format(
                    task_oldID_to_newID[row.task1],
                    task_oldID_to_newID[row.task2],
                    row.score
                ))
                n_edges += 1
    print("# edges in STRING after pruing:", n_edges)

