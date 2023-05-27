import pickle
import os
from multiprocessing import Pool
from collections import defaultdict, OrderedDict
import urllib
from urllib.error import HTTPError
from http.client import IncompleteRead
import urllib.request
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm
import pandas as pd
import json as js
parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, default='../chembl_full', help='dir path to the raw dataset')
parser.add_argument('--dataset', type=str, default='chembl_raw')
# parser.add_argument('--n-proc', type=int, default=12, help='number of processes to run when downloading assay & target information')
parser.add_argument('--output-file', default='assay2target.tsv', help='file path to save the assay2target table')
parser.add_argument('--parallel',action='store_true')
parser.add_argument('--n-proc', type=int, default=100, help='number of processes to run when downloading assay & target information')

parser.add_argument('--store-assay', action='store_true')
parser.add_argument('--store-molecular', action='store_true')
parser.add_argument('--save-each',  type=int, default=100)
parser.add_argument('--items', type=str, nargs='+', default=None, help="")


args = parser.parse_args()
import lmdb
'''
http://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/
'''


def get_record(website):
    try:
        data = urllib.request.urlopen(website).read().decode("utf-8")
    except:
        return {}
    root = ET.fromstring(data)
    record = defaultdict(list)
    total_length=0
    for child in root:
        record[child.tag] = child.text
        total_length+=len(child.text)
    record['total_length'] = total_length


    # for item in root.findall('target_components/target_component/target_component_xrefs/target'):
    #     xref_id = item.find('xref_id').text
    #     xref_src_db = item.find('xref_src_db').text
    #     record[xref_src_db].append(xref_id)

    return record


def generate_table_row_str(assay_id):
    assay_record = get_record('https://www.ebi.ac.uk/chembl/api/data/assay/{}'.format(assay_id))
    if not assay_record:
        return '{}\t\t\t\t\t\t'.format(assay_id)
    target_id = assay_record['target_chembl_id']
    target_record = get_record('https://www.ebi.ac.uk/chembl/api/data/target/{}'.format(target_id))
    if not target_record:
        return '{}\t\t\t\t\t\t'.format(assay_id)
    target_name = target_record['pref_name']
    target_type = target_record['target_type']
    assay_type = assay_record['assay_type']
    organism = target_record['organism']
    uniprot_list = target_record['UniProt']


    print("assay_id\t", assay_id, '\tuniprot\t', uniprot_list)
    return '{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
        assay_id,
        assay_type,
        target_id,
        target_type,
        target_name,
        organism,
        ','.join(uniprot_list)
    )

Keys_meta=['Activity',
           'Assay',
           'Document',
           'Drug',
           'Drug_Indication',
           'Drug_Indication_PARENT',
           'Drug_Warning',
           'Drug_Warning_PARENT',
           'Mechanism',
           'Mechanism_PARENT',
           'Mechanism_TARGET',
           'Metabolism',
           'Molecule',
           'Molecule_PARENT',
           'Molecule_Form',
           'Molecule_Form_PARENT',
           'Similarity',
           'Substructure',
           'Target']


def generate_table_all_information(assay_id):
    items = args.items
    if 'Activity' in items:
        Activity = js.loads(urllib.request.urlopen('https://www.ebi.ac.uk/chembl/api/data/activity.json?assay_chembl_id={}'.format(assay_id)).read().decode("utf-8"))
    else:
        Activity=None
    if 'Assay' in items:
        Assay= js.loads(urllib.request.urlopen('https://www.ebi.ac.uk/chembl/api/data/assay.json?assay_chembl_id={}'.format(assay_id)).read().decode("utf-8"))
    else:
        Assay=None
    if 'Document' in items:
        Document= js.loads(urllib.request.urlopen('https://www.ebi.ac.uk/chembl/api/data/document.json?document_chembl_id={}'.format(assay_id)).read().decode("utf-8"))
    else:
        Document=None
    if 'Drug' in items:
        Drug = js.loads(urllib.request.urlopen('https://www.ebi.ac.uk/chembl/api/data/drug.json?molecule_chembl_id={}'.format(assay_id)).read().decode("utf-8"))
    else:
        Drug = None
    if 'Drug_Indication' in items:
        Drug_Indication = js.loads(urllib.request.urlopen('https://www.ebi.ac.uk/chembl/api/data/drug_indication.json?molecule_chembl_id={}'.format(assay_id)).read().decode("utf-8"))
    else:
        Drug_Indication=None
    if 'Drug_Indication_PARENT' in items:
        Drug_Indication_PARENT = js.loads(urllib.request.urlopen(
            'https://www.ebi.ac.uk/chembl/api/data/drug_indication.json?parent_molecule_chembl_id={}'.format(
                assay_id)).read().decode("utf-8"))
    else:
        Drug_Indication_PARENT=None
    if 'Drug_Warning' in items:
        Drug_Warning = js.loads(urllib.request.urlopen('https://www.ebi.ac.uk/chembl/api/data/drug_warning.json?molecule_chembl_id={}'.format(assay_id)).read().decode("utf-8"))
    else:
        Drug_Warning=None
    if 'Drug_Indication_PARENT' in items:
        Drug_Warning_PARENT = js.loads(urllib.request.urlopen(
            'https://www.ebi.ac.uk/chembl/api/data/drug_warning.json?parent_molecule_chembl_id={}'.format(assay_id)).read().decode(
            "utf-8"))
    else:
        Drug_Warning_PARENT=None
    # Image = js.loads(urllib.request.urlopen('https://www.ebi.ac.uk/chembl/api/data/image/{}?format=svg'.format(assay_id)).read().decode("utf-8"))
    if 'Mechanism' in items:
        Mechanism = js.loads(urllib.request.urlopen('https://www.ebi.ac.uk/chembl/api/data/mechanism.json?molecule_chembl_id={}'.format(assay_id)).read().decode("utf-8"))
    else:
        Mechanism = None
    if 'Mechanism_PARENT' in items:
        Mechanism_PARENT = js.loads(urllib.request.urlopen(
            'https://www.ebi.ac.uk/chembl/api/data/mechanism.json?parent_molecule_chembl_id={}'.format(assay_id)).read().decode(
            "utf-8"))
    else:
        Mechanism_PARENT = None
    if 'Mechanism_TARGET' in items:
        Mechanism_TARGET = js.loads(urllib.request.urlopen(
            'https://www.ebi.ac.uk/chembl/api/data/mechanism.json?target_chembl_id={}'.format(assay_id)).read().decode(
            "utf-8"))
    else:
        Mechanism_TARGET = None
    if 'Metabolism' in items:
        Metabolism = js.loads(urllib.request.urlopen('https://www.ebi.ac.uk/chembl/api/data/metabolism.json?metabolite_chembl_id={}'.format(assay_id)).read().decode("utf-8"))
    else:
        Metabolism = None
    if 'Molecule' in items:
        Molecule = js.loads(urllib.request.urlopen('https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_chembl_id={}'.format(assay_id)).read().decode("utf-8"))
    else:
        Molecule = None
    if 'Molecule_PARENT' in items:
        Molecule_PARENT = js.loads(urllib.request.urlopen(
            'https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_hierarchy__parent_chembl_id={}'.format(assay_id)).read().decode(
            "utf-8"))
    else:
        Molecule_PARENT = None
    if 'Molecule_Form' in items:
        Molecule_Form = js.loads(urllib.request.urlopen('https://www.ebi.ac.uk/chembl/api/data/molecule_form.json?molecule_chembl_id={}'.format(assay_id)).read().decode("utf-8"))
    else:
        Molecule_Form = None
    if 'Molecule_Form_PARENT' in items:
        Molecule_Form_PARENT =js.loads( urllib.request.urlopen('https://www.ebi.ac.uk/chembl/api/data/molecule_form.json?parent_chembl_id={}'.format(assay_id)).read().decode("utf-8"))
    else:
        Molecule_Form_PARENT = None

    if 'Similarity' in items:
        try:
            Similarity = js.loads(urllib.request.urlopen('https://www.ebi.ac.uk/chembl/api/data/similarity/{}/80.json'.format(assay_id)).read().decode("utf-8"))
        except:
            Similarity=None
    else:
        Similarity = None

    if 'Substructure' in items:
        try:
            Substructure = js.loads(urllib.request.urlopen('https://www.ebi.ac.uk/chembl/api/data/substructure/{}.json'.format(assay_id)).read().decode("utf-8"))
        except:
            Substructure=None
    else:
        Substructure = None

    if 'Target' in items:
        Target = js.loads(urllib.request.urlopen('https://www.ebi.ac.uk/chembl/api/data/target.json?target_chembl_id={}'.format(assay_id)).read().decode("utf-8"))
    else:
        Target = None
    # Target_Component = urllib.request.urlopen('https://www.ebi.ac.uk/chembl/api/data/target_component.json?targets__target__target_chembl_id={}'.format(assay_id)).read().decode("utf-8")

    result={
        'Activity':Activity,
            'Assay':Assay,
           'Document':Document,
           'Drug':Drug,
           'Drug_Indication':Drug_Indication,
           'Drug_Indication_PARENT':Drug_Indication_PARENT,
           'Drug_Warning':Drug_Warning,
           'Drug_Warning_PARENT':Drug_Warning_PARENT,
           'Mechanism':Mechanism,
           'Mechanism_PARENT':Mechanism_PARENT,
           'Mechanism_TARGET':Mechanism_TARGET,
           'Metabolism':Metabolism,
           'Molecule':Molecule,
           'Molecule_PARENT':Molecule_PARENT,
           'Molecule_Form':Molecule_Form,
           'Molecule_Form_PARENT':Molecule_Form_PARENT,
           'Similarity':Similarity,
           'Substructure':Substructure,
           'Target':Target}

    counts={}
    for key in result.keys():
        if result[key] is not None:
            counts[key]=result[key]['page_meta']['total_count']
        else:
            counts[key]=0

    return result,counts



if __name__ == '__main__':
    root = '../{}/raw/'.format(args.dataset)
    with open(os.path.join(root, 'labelsHard.pckl'), 'rb') as f:
        targetMat = pickle.load(f)
        sampleAnnInd = pickle.load(f)
        targetAnnInd = pickle.load(f)

    assay_ids = list(targetAnnInd.index)
    # with Pool(args.n_proc) as p:
    #     table_rows_str = p.map(generate_table_row_str, tqdm(assay_ids))
    with open(os.path.join(args.input_dir, 'rdkit_molecule.pkl'), 'rb') as f:
        rdkit_mol_list = pickle.load(f)
    mol_prop = pd.read_csv(os.path.join(args.input_dir, 'mol_properties.csv'), index_col=0)
    print('{} molecules with SMILES'.format(mol_prop.shape[0]))



    if args.store_assay:
        path_lmdb = 'assay_info'
        print('saving to ', path_lmdb)

        if not os.path.exists(path_lmdb):
            os.makedirs(path_lmdb)
        lmdb_env = lmdb.open(path_lmdb, map_size=1e12)
        txn = lmdb_env.begin(write=True)

        for id in tqdm(assay_ids):
            result,counts=generate_table_all_information(id)
            result_list = [result, counts]
            txn.put(key=(str(id)).encode(), value=str(result_list).encode())
        print('********************************commit substructure sampling caching **************************')
        txn.commit()
        lmdb_env.close()


    if args.store_molecular:
        path_lmdb = '../prompt_data/ChEMBL_STRING/mole_info'
        print('saving to ', path_lmdb)

        if not os.path.exists(path_lmdb):
            os.makedirs(path_lmdb)
        lmdb_env = lmdb.open(path_lmdb, map_size=1e12)
        txn = lmdb_env.begin(write=True)

        keys = list(txn.cursor().iternext(values=False))
        start=-1
        print('finding start........')
        for cnt,id in tqdm(enumerate(mol_prop['chembl'])):
            if not (str(id)).encode() in keys:
                start = cnt
                break
        print('start from ',start)
        if not args.parallel:
            for cnt, id in tqdm(enumerate(mol_prop['chembl'][start:])):
                result,counts=generate_table_all_information(id)
                result_list = [result, counts]
                txn.put(key=(str(id)).encode(), value=str(result_list).encode())
                if (cnt+1) % args.save_each == 0:
                    print('********************************commit substructure sampling caching **************************')
                    txn.commit()
                    txn = lmdb_env.begin(write=True)
        else:
            for i in tqdm(range(len(mol_prop['chembl'][start:])//args.save_each)):
                end=min(start + args.save_each,len(mol_prop['chembl']))
                with Pool(args.n_proc) as p:
                    results = p.map(generate_table_all_information, (mol_prop['chembl'][start:end]))
                    for cnt, id in enumerate(mol_prop['chembl'][start:end]):
                        result=results[cnt]
                        txn.put(key=(str(id)).encode(), value=str(result).encode())
                        txn.commit()
                        txn = lmdb_env.begin(write=True)
                start = start+args.save_each


    # with Pool(args.n_proc) as p:
    #     assay_info = p.map(generate_table_all_information, tqdm(assay_ids))
    # with open('file.txt', 'a') as f:
    #     f.write('Thank you.\n')
    # f = open('assay_info.json', 'w')
    # f.write(js.dumps(assay_info))
    # f.close()


    # table_rows_str={}
    # counts_total={}
    # for id in tqdm(assay_ids):
    #     id_res,result,counts=generate_table_all_information(id)
    #     # for key in counts.keys():
    #     #     if not key in counts_total:
    #     #         counts_total[key]=[counts[key]]
    #     #     else:
    #     #         counts_total[key].append(counts[key])
    #     table_rows_str[id_res] = result
    #     counts_total[id_res]=counts

    # with Pool(args.n_proc) as p:
    #     table_rows_str_1 = p.map(generate_table_all_information, tqdm(mol_prop['chembl'][0:1000]))

    # table_rows_str_1=[]
    # for idx in mol_prop['chembl'][0:100]:
    #     table_rows_str_1.append(generate_table_all_information(idx))


    # f = open('table_rows_str_1_tem.json', 'w')
    # f.write(js.dumps(table_rows_str_1))
    # f.close()


    # f = open('counts_total.json', 'w')
    # f.write(js.dumps(counts_total))
    # f.close()
