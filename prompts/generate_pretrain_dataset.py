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
import lmdb
# from src.datasets.molecule_dataset import mol_to_graph_data_obj_simple,graph_data_obj_to_mol_simple,graph_data_obj_to_nx_simple,nx_to_graph_data_obj_simple
from ogb.utils import smiles2graph
import commentjson

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='../chembl_full', help='dir path to the raw dataset')
parser.add_argument('--dataset', type=str, default='chembl_raw')
# parser.add_argument('--n-proc', type=int, default=12, help='number of processes to run when downloading assay & target information')
parser.add_argument('--output_file', default='assay2target.tsv', help='file path to save the assay2target table')
parser.add_argument('--n_proc', type=int, default=12, help='number of processes to run when downloading assay & target information')
parser.add_argument('--load_json', action='store_true' )
parser.add_argument('--load_lmdb', action='store_true' )
parser.add_argument('--load_excel', action='store_true' )

parser.add_argument('--load_assay',action='store_true')
parser.add_argument('--assay_type',type=str,default='excel')
parser.add_argument('--load_mole',action='store_true')
parser.add_argument('--mole_type',type=str,default='lmdb')

parser.add_argument('--prompt_type',type=str,default='mlm')

parser.add_argument('--generate_mole_text',action='store_true')

parser.add_argument('--generate_assay_prompt',action='store_true')

parser.add_argument('--generate_assay_text',action='store_true')

parser.add_argument('--generate_mole_graph_only',action='store_true')

parser.add_argument('--generate_assay_graph_only',action='store_true')

parser.add_argument('--assay_split_transductive',action='store_true')

parser.add_argument('--split_file',type=str,default=None)

parser.add_argument('--use_augmented_prompt',type=str,default=None)

parser.add_argument('--split_index',type=str,default='split.json')


parser.add_argument('--file_save_name',type=str,default='split_0.csv')



parser.add_argument('--add_negation',action='store_true')

parser.add_argument('--negation_p',type=float,default=0.1)




args = parser.parse_args()
import numpy as np
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




def load_json():
    f = open('table_rows_str_1_tem.json', 'r')
    content = f.read()
    f.close()
    table_rows_str_2 = js.loads(content)

    text_dict={}
    count_dict={}

    text_panda={}
    count_panda={}

    for data in table_rows_str_2:
        idx=data[0]
        text=data[1]
        count=data[2]

        text_dict[idx]=text
        count_dict[idx]=count

        for key in text.keys():
            if not key in text_panda:
                text_panda[key]=[text[key]]
            else:
                text_panda[key].append(text[key])


        for key in count.keys():
            if not key in count_panda:
                count_panda[key]=[count[key]]
            else:
                count_panda[key].append(count[key])

    count_panda=pd.DataFrame(count_panda)

    print(1)



    for key_meta in text_panda.keys():
        text_mole_panda = {}
        text_mole_panda['id'] = []
        for id,data in enumerate(text_panda[key_meta]):

            keys=list(text_panda[key_meta][0].keys())
            keys.remove('page_meta')
            key_tem=keys[0]
            if data is not None and len(data[key_tem])>0:
                for item in data[key_tem]:
                    text_mole_panda['id'].append(table_rows_str_2[id][0])

                    for key in item.keys():
                        key_dic=key
                        if not key_dic in text_mole_panda:
                            text_mole_panda[key_dic]=[item[key]]*len(text_mole_panda['id'])
                        else:
                            text_mole_panda[key_dic].append(item[key])
                    for key in text_mole_panda.keys():
                        if key!='id' and not key in item:
                            text_mole_panda[key].append(None)

        text_assay_panda=pd.DataFrame(text_mole_panda)
        text_assay_panda.to_excel('text_mole'+key_meta+'.xlsx', sheet_name='data1')

    # return count_panda, text_assay_panda


def apply_prompt(prompts,key,value):
    if key in prompts:
        return prompts[key].format(value)

map_dict={'N':'No','Y':'Yes'}
def map_label(label):
    if label in map_dict:
        return map_dict[label]
    else:
        return str(label)

def map_negation(label):
    if isinstance(label,int):
        return -label
    else:
        try:
            label = eval(label)
            assert isinstance(label,float)
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

if __name__ == '__main__':

    root = '../{}/raw/'.format(args.dataset)
    with open(os.path.join(root, 'labelsHard.pckl'), 'rb') as f:
        targetMat = pickle.load(f)
        sampleAnnInd = pickle.load(f)
        targetAnnInd = pickle.load(f)

    assay_ids = list(targetAnnInd.index)


    # with open(os.path.join(args.input_dir, 'rdkit_molecule.pkl'), 'rb') as f:
    #     rdkit_mol_list = pickle.load(f)
    # mol_prop is the complete graph-id mapping. On the contrary,targetMat is not complete. So traversal of graph should be down by mol_prop
    mol_prop = pd.read_csv(os.path.join(args.input_dir, 'mol_properties.csv'), index_col=0)
    print('{} molecules with SMILES'.format(mol_prop.shape[0]))
    # graph_chem=mol_prop['chembl']
    # graph_chem_to_id_dict={}
    # for id in graph_chem.index:
    #     graph_chem_to_id_dict[graph_chem[id]]=id

    with open(os.path.join('../chembl_raw/raw', 'chembl20Smiles.pckl'), 'rb') as f:
        mol_smiles = pickle.load(f)




    if args.load_assay or  args.generate_assay_text or args.generate_assay_prompt:
        if args.assay_type=='excel':
            for key_meta in Keys_meta:
                df = pd.read_excel('text_mole' + key_meta + '.xlsx')

            data_T0=assay_ids

            df = pd.read_excel('text_assay.xlsx')
        else:
            raise ValueError('assay data not supported')

        assay_dict={}
        for id,chembl_id in enumerate(list(df['assay_chembl_id'])):
            item_dict={}
            for key in df.keys():
                item_dict[key]=df[key][id]
            assay_dict[chembl_id]=item_dict

        if args.generate_assay_prompt:
            prompts_assay={}
            for _, assay_chembl_id in tqdm(enumerate(assay_dict.keys())):
                df_assay = assay_dict[assay_chembl_id]
                if not (pd.isnull((df_assay['description'])) or df_assay['description'] in ['[]', []]):
                    text_assay = 'The assay is '

                    text_assay += df_assay['description'] + ' '
                    if 'default value' in df_assay['confidence_description'].lower():
                        text_assay += '. '
                    else:
                        text_assay += ', and it is ' + df_assay['confidence_description'] + ' . '
                else:
                    text_assay = ''

                text_assay += 'The assay has properties: '
                for key in df_assay.keys():
                    if not (pd.isnull((df_assay[key])) or df_assay[key] in ['[]', []] or key in ['Unnamed: 0',
                                                                                                 'assay_chembl_id',
                                                                                                 'document_chembl_id',
                                                                                                 'description',
                                                                                                 'confidence_description',
                                                                                                 'relationship_description',
                                                                                                 'target_chembl_id',
                                                                                                 'bao_format',
                                                                                                 'confidence_score',
                                                                                                 'relationship_type',
                                                                                                 'src_id', 'assay_type',
                                                                                                 "assay_tax_id",
                                                                                                 "bao_label",
                                                                                                 "cell_chembl_id",
                                                                                                 "src_assay_id",
                                                                                                 "tissue_chembl_id",
                                                                                                 "document_chembl_id", ]):
                        text_assay += key.replace('_', ' ') + ' is '
                        text_assay += str(df_assay[key]) + ' ; '
                text_data = text_assay[:-2] + '. '
                text_data += 'Is the molecule effective to this assay?'
                prompts_assay[assay_chembl_id]=[text_data]
            with open('../prompts_backup/prompt_assay.json', 'w') as f:
                js.dump(prompts_assay, f,indent=2)

        if args.generate_assay_text:
            with open("../prompts_backup/prompt_assay.json", 'r') as load_f:
                prompts = commentjson.load(load_f)
            with open("../prompts_backup/augmented_prompt_assay.json", 'r') as load_f:
                prompts_augmented = commentjson.load(load_f)


            if os.path.exists('../pretrain_datasets/valid_graph_ids.csv'):
                print('loading existing valid_graph_ids.csv')
                # with open('valid_graph_ids.csv') as f:
                #     valid_graph_ids=commentjson.load(f)
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
                        # graph_data = mol_prop['smiles'][graph_chem_to_id_dict[graph_chembl_id.decode()]]
                        continue

                valid_graph_ids=pd.DataFrame(valid_graph_ids,columns=['graph_ind'])
                valid_graph_ids.to_csv('valid_graph_ids.csv')

                # with open('valid_graph_ids.json','w') as f:
                #     commentjson.dump(valid_graph_ids,f)
            valid_graph_ids=set(valid_graph_ids['graph_ind'].values.tolist())
            print('number of valid graphs: ', len(valid_graph_ids))

            def generate_assay_data(assay_ids,molecule_index_set):
                data_all=[]
                for _,assay_chembl_id in  tqdm(enumerate(assay_ids)):
                    # df_assay = assay_dict[assay_chembl_id]
                    # if not (pd.isnull((df_assay['description'])) or df_assay['description'] in ['[]', []]):
                    #     text_assay='The assay is '
                    #
                    #     text_assay+=df_assay['description']+' '
                    #     if 'default value' in df_assay['confidence_description'].lower():
                    #         text_assay+='. '
                    #     else:
                    #         text_assay+=', and it is '+ df_assay['confidence_description']+' . '
                    # else:
                    #     text_assay=''
                    #
                    # text_assay += 'The assay has properties: '
                    # for key in df_assay.keys():
                    #     if not (pd.isnull((df_assay[key]))  or df_assay[key]in ['[]',[]] or key in ['Unnamed: 0','assay_chembl_id','document_chembl_id','description','confidence_description','relationship_description','target_chembl_id','bao_format','confidence_score','relationship_type','src_id','assay_type',"assay_tax_id","bao_label","cell_chembl_id","src_assay_id","tissue_chembl_id","document_chembl_id",]):
                    #         text_assay += key.replace('_', ' ')+' is '
                    #         text_assay += str(df_assay[key])+ ' ; '
                    # text_data = text_assay[:-2] + '. '
                    # text_data += 'Is Molecular effective to this assay ? '


                    if args.use_augmented_prompt is not None:
                        type=args.use_augmented_prompt
                        text_data=prompts_augmented[assay_chembl_id][type][np.random.randint(len(prompts_augmented[assay_chembl_id][type]))]
                    else:
                        text_data = prompts[assay_chembl_id][np.random.randint(len(prompts[assay_chembl_id]))]

                    for graph_id in set(targetMat[:,targetAnnInd[assay_chembl_id]].nonzero()[0]) & molecule_index_set & valid_graph_ids:
                        graph_data=mol_smiles[graph_id]
                        # try:
                        #     graph=smiles2graph(graph_data)
                        # except:
                        #     print('graph data error ',graph_data)
                        #     # graph_data = mol_prop['smiles'][graph_chem_to_id_dict[graph_chembl_id.decode()]]
                        #     continue
                        # if graph['node_feat'].shape[0]==0 or graph['edge_feat'].shape[0]==0 or graph['edge_index'].shape[1]==0:
                        #     print('empty graph',graph)
                        #     continue

                        label=targetMat[graph_id,targetAnnInd[assay_chembl_id]]
                        assert label!=0
                        label='Yes' if label>0 else 'No'

                        # graph_data=mol_to_graph_data_obj_simple(rdkit_mol_list[graph_id])
                        # graph_data=mol_smiles[graph_id]
                        # try:
                        if  graph_id in mol_prop['chembl']:
                            graph_data = mol_smiles[graph_id]
                            # graph_data = mol_prop['smiles'][graph_id]
                            # if text_graph != graph_data:
                            #     print(graph_id,text_graph,graph_data)
                            chemid=mol_prop['chembl'][graph_id]
                            data_all += [(graph_data, text_data, label, chemid,assay_chembl_id)]
                        # except:
                        #     pass
                return data_all

            assays=list(assay_dict.keys())

            if args.assay_split_transductive:
                args.split_file='transductive_'+args.split_file
            else:
                args.split_file += 'inductive_'+args.split_file
            if args.split_file is not None and os.path.exists(args.split_file):
                print('loading existing '+args.split_file)

                df=pd.read_csv(os.path.join(args.split_file,'train_assays.csv'))
                train_assays=df['values'].values.tolist()
                df=pd.read_csv(os.path.join(args.split_file,'valid_assays.csv'))
                valid_assays=df['values'].values.tolist()
                df=pd.read_csv(os.path.join(args.split_file,'train_graph_index.csv'))
                train_graph_index=df['values'].values.tolist()
                df=pd.read_csv(os.path.join(args.split_file,'valid_graph_index.csv'))
                valid_graph_index=df['values'].values.tolist()

                # with open(args.split_file) as f:
                #     split_index=commentjson.load(f)
                # train_assays=split_index['train_assays']
                # valid_assays=split_index['valid_assays']
                # train_graph_index=split_index['train_graph_index']
                # valid_graph_index=split_index['valid_graph_index']
            else:
                print('Warning! Creating new split')
                train_index = np.random.choice(np.arange(len(assays)), size=int(len(assays)*0.8), replace=False)
                valid_index=np.delete(np.arange(len(assays)), train_index)

                train_assays=[assays[i] for i in train_index]
                valid_assays=[assays[i] for i in valid_index]

                if args.assay_split_transductive:
                    train_graph_index =np.random.choice(np.arange(len(mol_smiles)), size=int(len(mol_smiles)*0.8), replace=False)
                    valid_graph_index = np.delete(np.arange(len(mol_smiles)), train_graph_index)
                else:
                    train_graph_index =np.arange(len(mol_smiles))
                    valid_graph_index = np.arange(len(mol_smiles))
                train_graph_index=train_graph_index.tolist()
                valid_graph_index=valid_graph_index.tolist()
                split_index={}
                split_index['train_assays']=train_assays
                split_index['valid_assays']=valid_assays
                split_index['train_graph_index']=train_graph_index
                split_index['valid_graph_index']=valid_graph_index
                os.mkdir(args.split_file)
                for key in split_index:
                    df=pd.DataFrame(split_index[key],columns=['values'])
                    df.to_csv(os.path.join(args.split_file,key+'.csv'))
                # with open(args.split_file,'w') as f:
                #     commentjson.dump(split_index,f)

            data_all_train=generate_assay_data(train_assays,set(train_graph_index))
            data_all_valid=generate_assay_data(valid_assays,set(valid_graph_index))
            print(len(data_all_train))
            print(len(data_all_valid))
                #     if len(data_all)>1000:
                #         break
                # if len(data_all) > 1000:
                #     break
            # print(len(data_all))
            # train_number=int(len(data_all)*0.8)
            # data_all_train=data_all[0:train_number]
            # data_all_valid=data_all[train_number:]
            file_prefix="assay_graph_text_T0_"
            if args.assay_split_transductive:
                file_prefix+="transductive_"
            else:
                file_prefix += "inductive_"
            if args.use_augmented_prompt is not None:
                file_prefix+=args.use_augmented_prompt+'_'
            df = pd.DataFrame(data_all_train,
                              columns=['graph', 'text','label','chemid','assayid'
                                       ])
            df.to_csv(file_prefix+'train_'+args.file_save_name)
            df = pd.DataFrame(data_all_valid,
                              columns=['graph', 'text','label','chemid','assayid'
                                       ])
            df.to_csv(file_prefix+'valid_'+args.file_save_name)
            # f = open('assay_graph_text.txt', 'w')
            # for data_line in data_all:
            #     f.writelines([data_line, '\n'])
            #
            # f.close()

            # print(1)


    if args.load_mole or args.generate_mole_text:
        if args.mole_type=='lmdb':
            path_lmdb = '../prompt_data/ChEMBL_STRING/mole_info'
            print('loading from ', path_lmdb)

            lmdb_env = lmdb.open(path_lmdb, map_size=1e12, readonly=True,
                lock=False, readahead=False, meminit=False)
            txn = lmdb_env.begin(write=False)

            mole_in_crawler = list(txn.cursor().iternext(values=False))
            tem=txn.get(mole_in_crawler[0])
            if tem is not None:
                mol = eval(tem)
            else:
                raise ValueError("substructure_tensor not prepared")
            print(1)
        else:
            raise ValueError('mole type not supported')

        if args.generate_mole_text:

            if os.path.exists('valid_graph_ids_mole.csv'):
                print('loading existing valid_graph_ids_mole.csv')
                # with open('valid_graph_ids.csv') as f:
                #     valid_graph_ids=commentjson.load(f)
                valid_graph_ids_mole=pd.read_csv('valid_graph_ids_mole.csv')
            else:

                valid_graph_ids_mole=[]
                for _,graph_chembl_id in  tqdm(enumerate(mole_in_crawler)):
                    try:
                        df_mole = eval(txn.get(graph_chembl_id))
                        graph_data = df_mole[0]['Molecule']['molecules'][0]['molecule_structures']['canonical_smiles']
                        graph = smiles2graph(graph_data)
                        if graph['node_feat'].shape[0] == 0 or graph['edge_feat'].shape[0] == 0 or \
                                graph['edge_index'].shape[1] == 0:
                            print('empty graph', graph)
                            continue
                        valid_graph_ids_mole.append(graph_chembl_id)
                    except:
                        # print('graph data error ', graph_data)
                        # graph_data = mol_prop['smiles'][graph_chem_to_id_dict[graph_chembl_id.decode()]]
                        continue

                valid_graph_ids_mole=pd.DataFrame(valid_graph_ids_mole,columns=['graph_chembl_id'])
                valid_graph_ids_mole.to_csv('valid_graph_ids_mole.csv')

                # with open('valid_graph_ids.json','w') as f:
                #     commentjson.dump(valid_graph_ids,f)
            valid_graph_ids_mole=set(valid_graph_ids_mole['graph_chembl_id'].values.tolist())
            print('number of valid graphs: ', len(valid_graph_ids_mole))


            if args.prompt_type=='mlm':
                with open("../prompts_backup/prompt_mlm.json", 'r') as load_f:
                    prompts = commentjson.load(load_f)

                data_all=[]
                # mol_prop.set_index("chembl", inplace=True)
                # graph_data = mol_to_graph_data_obj_simple()
                for _,graph_chembl_id in  tqdm(enumerate(mole_in_crawler)):

                    # graph=mol_prop.loc[graph_chembl_id.decode()]
                    # graph_data=mol_to_graph_data_obj_simple(rdkit_mol_list[graph_chem_to_id_dict[graph_chembl_id.decode()]])

                    df_mole = eval(txn.get(graph_chembl_id))
                    text_graph=''

                    if len(df_mole[0]['Molecule']['molecules'])>0:
                        try:
                            graph_data = df_mole[0]['Molecule']['molecules'][0]['molecule_structures']['canonical_smiles']
                        except:
                            # graph_data = mol_prop['smiles'][graph_chem_to_id_dict[graph_chembl_id.decode()]]
                            continue

                        # properties=df_mole[0]['Molecule']['molecules'][0]
                        # text_graph += 'The graph has properties: '
                        #
                        # for key in properties.keys():
                        #     try:
                        #         if not (key in ['molecule_properties','cross_references','molecule_hierarchy','molecule_synonyms','molecule_structures'] or pd.isnull((properties[key]))  or properties[key]in ['[]',[]] or (properties[key] is None)):
                        #             text_graph += key.replace('_', ' ')+' is '
                        #             text_graph += str(properties[key])+ ' ; '
                        #     except:
                        #         if not (key in ['molecule_properties','cross_references','molecule_hierarchy','molecule_synonyms','molecule_structures'] or pd.isnull((properties[key])).all()  or properties[key]in ['[]',[]] or (properties[key] is None)):
                        #             text_graph += key.replace('_', ' ')+' is '
                        #             text_graph += str(properties[key])+ ' ; '
                        properties=df_mole[0]['Molecule']['molecules'][0]['molecule_properties']
                        if properties is not None:
                            for key in properties.keys():
                                if not (key in [] or pd.isnull((properties[key])) or properties[key] in ['[]', []] or (
                                        properties[key] is None)):
                                    key_prompt='Molecule_molecules_molecule_properties_'+key
                                    if key_prompt in prompts:
                                        text_graph = prompts[key_prompt][0].format(properties[key])
                                        data_all.append((graph_data,text_graph))
                    # if len(data_all)>1000:
                    #     break
                df = pd.DataFrame(data_all,
                                  columns=['graph', 'text',
                                           ])
                df.to_csv('mole_graph_text_mlm.csv')

                # f = open('mole_graph_text_all.txt', 'w')
                # for data_line in data_all:
                #     f.writelines([data_line, '\n'])
                #
                # f.close()

                # print(1)
            elif args.prompt_type=='T0':
                with open("../prompts_backup/prompt_T0.json", 'r') as load_f:
                    prompts = commentjson.load(load_f)
                with open("../prompts_backup/augmented_prompt_T0.json", 'r') as load_f:
                    prompts_augmented = commentjson.load(load_f)

                data_all=[]
                # mol_prop.set_index("chembl", inplace=True)
                # graph_data = mol_to_graph_data_obj_simple()
                for _,graph_chembl_id in  tqdm(enumerate(mole_in_crawler)):

                    # graph=mol_prop.loc[graph_chembl_id.decode()]
                    # graph_data=mol_to_graph_data_obj_simple(rdkit_mol_list[graph_chem_to_id_dict[graph_chembl_id.decode()]])

                    df_mole = eval(txn.get(graph_chembl_id))


                    if len(df_mole[0]['Molecule']['molecules'])>0 and (str(graph_chembl_id) in valid_graph_ids_mole):
                        # try:
                        #     graph_data = df_mole[0]['Molecule']['molecules'][0]['molecule_structures']['canonical_smiles']
                        #     graph=smiles2graph(graph_data)
                        #
                        #
                        # except:
                        #     print('graph data error ',graph_data)
                        #
                        #     # graph_data = mol_prop['smiles'][graph_chem_to_id_dict[graph_chembl_id.decode()]]
                        #     continue
                        # if graph['node_feat'].shape[0]==0 or graph['edge_feat'].shape[0]==0 or graph['edge_index'].shape[1]==0:
                        #     print('empty graph',graph)
                        #     continue
                        # item = graph
                        # item_new = Data()
                        # for key in item.keys():
                        #     item_new[key] = torch.tensor(item[key]) if (
                        #         not isinstance(item[key], torch.Tensor)) else item[key]
                        # res=graphormer_data_transform_tensor(item_new)
                        # properties=df_mole[0]['Molecule']['molecules'][0]
                        # text_graph += 'The graph has properties: '
                        #
                        # for key in properties.keys():
                        #     try:
                        #         if not (key in ['molecule_properties','cross_references','molecule_hierarchy','molecule_synonyms','molecule_structures'] or pd.isnull((properties[key]))  or properties[key]in ['[]',[]] or (properties[key] is None)):
                        #             text_graph += key.replace('_', ' ')+' is '
                        #             text_graph += str(properties[key])+ ' ; '
                        #     except:
                        #         if not (key in ['molecule_properties','cross_references','molecule_hierarchy','molecule_synonyms','molecule_structures'] or pd.isnull((properties[key])).all()  or properties[key]in ['[]',[]] or (properties[key] is None)):
                        #             text_graph += key.replace('_', ' ')+' is '
                        #             text_graph += str(properties[key])+ ' ; '
                        graph_data = df_mole[0]['Molecule']['molecules'][0]['molecule_structures']['canonical_smiles']
                        properties=df_mole[0]['Molecule']['molecules'][0]['molecule_properties']
                        if properties is not None:
                            for key in properties.keys():
                                if not (key in [] or pd.isnull((properties[key])) or properties[key] in ['[]', []] or (
                                        properties[key] is None)):

                                    key_prompt='Molecule_molecules_molecule_properties_'+key

                                    if args.add_negation:
                                        add_negation=(np.random.rand()<args.negation_p) & ( key_prompt+'_negative' in prompts)
                                    else:
                                        add_negation=False

                                    if add_negation:
                                        key_prompt+='_negative'

                                    if key_prompt in prompts:
                                        if args.use_augmented_prompt is not None:
                                            text_prompts = prompts_augmented[key_prompt][args.use_augmented_prompt]
                                            prompt_id = np.random.randint(0, len(text_prompts))
                                            text_prompt = text_prompts[prompt_id]
                                        else:
                                            text_prompts = prompts[key_prompt]
                                            prompt_id=np.random.randint(0, len(text_prompts))
                                            text_prompt=text_prompts[prompt_id]
                                        label=map_label(properties[key])
                                        if add_negation:
                                            label=map_negation(label)
                                        data_all.append((graph_data,text_prompt,label,graph_chembl_id.decode()))
                    # if len(data_all)>1000:
                    #     break
                df = pd.DataFrame(data_all,
                                  columns=['graph', 'text','label','chemid'
                                           ])
                file_prefix = "mole_graph_text_T0_"
                if args.add_negation:
                    file_prefix+='negative'+str(args.negation_p).replace('.','')+'_'
                if args.use_augmented_prompt is not None:
                    file_prefix+=args.use_augmented_prompt+'_'
                df.to_csv(file_prefix  + args.file_save_name)
                # df.to_csv('mole_graph_text_T0_rich.csv')


    if args.generate_mole_graph_only:
        if args.mole_type=='lmdb':
            path_lmdb = '../prompt_data/ChEMBL_STRING/mole_info'
            print('loading from ', path_lmdb)

            lmdb_env = lmdb.open(path_lmdb, map_size=1e12, readonly=True,
                lock=False, readahead=False, meminit=False)
            txn = lmdb_env.begin(write=False)

            mole_in_crawler = list(txn.cursor().iternext(values=False))
            tem=txn.get(mole_in_crawler[0])
            if tem is not None:
                mol = eval(tem)
            else:
                raise ValueError("substructure_tensor not prepared")
            print(1)
        else:
            raise ValueError('mole type not supported')

        data_all = []
        with open("../prompts_backup/prompt_T0.json", 'r') as load_f:
            prompts = commentjson.load(load_f)


        from mole_key import key_int,key_float


        # mol_prop.set_index("chembl", inplace=True)
        # graph_data = mol_to_graph_data_obj_simple()
        data_int_all=[]
        data_float_all = []
        for _, graph_chembl_id in tqdm(enumerate(mole_in_crawler)):

            # graph=mol_prop.loc[graph_chembl_id.decode()]
            # graph_data=mol_to_graph_data_obj_simple(rdkit_mol_list[graph_chem_to_id_dict[graph_chembl_id.decode()]])

            df_mole = eval(txn.get(graph_chembl_id))

            if len(df_mole[0]['Molecule']['molecules']) > 0:
                try:
                    graph_data = df_mole[0]['Molecule']['molecules'][0]['molecule_structures']['canonical_smiles']
                    graph = smiles2graph(graph_data)


                except:
                    print('graph data error ', graph_data)

                    # graph_data = mol_prop['smiles'][graph_chem_to_id_dict[graph_chembl_id.decode()]]
                    continue
                if graph['node_feat'].shape[0] == 0 or graph['edge_feat'].shape[0] == 0 or graph['edge_index'].shape[
                    1] == 0:
                    print('empty graph', graph)
                    continue
                # item = graph
                # item_new = Data()
                # for key in item.keys():
                #     item_new[key] = torch.tensor(item[key]) if (
                #         not isinstance(item[key], torch.Tensor)) else item[key]
                # res=graphormer_data_transform_tensor(item_new)
                # properties=df_mole[0]['Molecule']['molecules'][0]
                # text_graph += 'The graph has properties: '
                #
                # for key in properties.keys():
                #     try:
                #         if not (key in ['molecule_properties','cross_references','molecule_hierarchy','molecule_synonyms','molecule_structures'] or pd.isnull((properties[key]))  or properties[key]in ['[]',[]] or (properties[key] is None)):
                #             text_graph += key.replace('_', ' ')+' is '
                #             text_graph += str(properties[key])+ ' ; '
                #     except:
                #         if not (key in ['molecule_properties','cross_references','molecule_hierarchy','molecule_synonyms','molecule_structures'] or pd.isnull((properties[key])).all()  or properties[key]in ['[]',[]] or (properties[key] is None)):
                #             text_graph += key.replace('_', ' ')+' is '
                #             text_graph += str(properties[key])+ ' ; '

                properties = df_mole[0]['Molecule']['molecules'][0]['molecule_properties']
                if properties is not None:
                    labels_int=[]
                    labels_float=[]
                    for key_prompt in prompts.keys():
                        key=key_prompt.replace('Molecule_molecules_molecule_properties_','')

                        if key in properties and not (pd.isnull((properties[key])) or properties[key] in ['[]', []] or (properties[key] is None)): # exclude None
                            label=properties[key]
                        else:
                            label=-100
                        label=map_label_graph_only(label)
                        if key in key_int:
                            labels_int.append(label)
                        elif key in key_float:
                            labels_float.append(label)
                    # data_int_all.append(labels_int)
                    # data_float_all.append(labels_float)
                    data_all.append([graph_data]+labels_int+labels_float+[graph_chembl_id.decode()])
                    # if len(data_all)>100:
                    #     break

                    #

                    # for key in properties.keys():
                    #     if not (key in [] or pd.isnull((properties[key])) or properties[key] in ['[]', []] or (
                    #             properties[key] is None)):
                    #         key_prompt = 'Molecule_molecules_molecule_properties_' + key
                    #         if key_prompt in prompts:
                    #             text_prompts = prompts[key_prompt]
                    #             prompt_id = np.random.randint(0, len(text_prompts))
                    #             text_prompt = text_prompts[prompt_id]
                    #             label = map_label(properties[key])
                    #             data_all.append((graph_data, text_prompt, label, graph_chembl_id.decode()))
            # if len(data_all)>10000:
            #     break

        # data_int_all=np.array(data_int_all)
        # data_float_all=np.array(data_float_all)

        # Sort keys with the same order as data
        key_int_sorted=[]
        key_float_sorted=[]
        for key_prompt in prompts.keys():
            key = key_prompt.replace('Molecule_molecules_molecule_properties_', '')
            if key in key_int:
                key_int_sorted.append('cla_'+key)
            elif key in key_float:
                key_float_sorted.append('reg_'+key)

        columes_name=['graph']+key_int_sorted+key_float_sorted+['chemid']
        df = pd.DataFrame(data_all,
                          columns=columes_name)
        def max_min_scaler(x):
            index_not_none=x!=-100
            x[index_not_none]=(x[index_not_none] - np.min(x[index_not_none])) / (np.max(x[index_not_none]) - np.min(x[index_not_none]))
            return x

        def add_minimum(x):
            index_not_none = x != -100
            minimum=np.min(x[index_not_none])
            if minimum<0:
                x[index_not_none]=x[index_not_none]-minimum
            return x

        # max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        df[key_float_sorted]=df[key_float_sorted].apply(max_min_scaler)
        df[key_int_sorted]=df[key_int_sorted].apply(add_minimum)


        file_prefix = "mole_graph_only_"
        df.to_csv(file_prefix + args.file_save_name)


        # df.to_csv('mole_graph_text_T0_rich.csv')



    if args.generate_assay_graph_only:
        if args.assay_type=='excel':
            for key_meta in Keys_meta:
                df = pd.read_excel('text_mole' + key_meta + '.xlsx')

            data_T0=assay_ids

            df = pd.read_excel('text_assay.xlsx')
        else:
            raise ValueError('assay data not supported')


        assay_dict={}
        for id,chembl_id in enumerate(list(df['assay_chembl_id'])):
            item_dict={}
            for key in df.keys():
                item_dict[key]=df[key][id]
            assay_dict[chembl_id]=item_dict

        def generate_data(assay_ids,molecule_index_set):
            data_all=[]

            for graph_id in molecule_index_set:
                if graph_id in mol_prop['chembl']:
                    graph_data = mol_smiles[graph_id]
                    try:
                        graph = smiles2graph(graph_data)
                    except:
                        print('graph data error ', graph_data)
                        # graph_data = mol_prop['smiles'][graph_chem_to_id_dict[graph_chembl_id.decode()]]
                        continue
                    if graph['node_feat'].shape[0] == 0 or graph['edge_feat'].shape[0] == 0 or graph['edge_index'].shape[
                        1] == 0:
                        print('empty graph', graph)
                        continue
                    labels=targetMat[graph_id,:]
                    chemid = mol_prop['chembl'][graph_id]
                    data_all.append([graph_data]+np.array(labels.todense()).tolist()[0]+[chemid])
                    # if len(data_all)>100:
                    #     break


            # for _,assay_chembl_id in  tqdm(enumerate(assay_ids)):
            #     df_assay = assay_dict[assay_chembl_id]
            #     if not (pd.isnull((df_assay['description'])) or df_assay['description'] in ['[]', []]):
            #         text_assay='The assay is '
            #
            #         text_assay+=df_assay['description']+' '
            #         if 'default value' in df_assay['confidence_description'].lower():
            #             text_assay+='. '
            #         else:
            #             text_assay+=', and it is '+ df_assay['confidence_description']+' . '
            #     else:
            #         text_assay=''
            #
            #     text_assay += 'The assay has properties: '
            #     for key in df_assay.keys():
            #         if not (pd.isnull((df_assay[key]))  or df_assay[key]in ['[]',[]] or key in ['Unnamed: 0','assay_chembl_id','document_chembl_id','description','confidence_description','relationship_description','target_chembl_id','bao_format','confidence_score','relationship_type','src_id','assay_type',"assay_tax_id","bao_label","cell_chembl_id","src_assay_id","tissue_chembl_id","document_chembl_id",]):
            #             text_assay += key.replace('_', ' ')+' is '
            #             text_assay += str(df_assay[key])+ ' ; '
            #     for graph_id in set(targetMat[:,targetAnnInd[assay_chembl_id]].nonzero()[0]) & molecule_index_set:
            #         graph_data=mol_smiles[graph_id]
            #         try:
            #             graph=smiles2graph(graph_data)
            #         except:
            #             print('graph data error ',graph_data)
            #             # graph_data = mol_prop['smiles'][graph_chem_to_id_dict[graph_chembl_id.decode()]]
            #             continue
            #         if graph['node_feat'].shape[0]==0 or graph['edge_feat'].shape[0]==0 or graph['edge_index'].shape[1]==0:
            #             print('empty graph',graph)
            #             continue
            #
            #
            #         text_data=text_assay[:-2]+ '. '
            #         # text_data+='Is Molecular '+text_graph+' effective to this assay ? '
            #         text_data += 'Is Molecular effective to this assay ? '
            #         label=targetMat[graph_id,targetAnnInd[assay_chembl_id]]
            #         assert label!=0
            #         label='Yes' if label>0 else 'No'
            #
            #         # graph_data=mol_to_graph_data_obj_simple(rdkit_mol_list[graph_id])
            #         # graph_data=mol_smiles[graph_id]
            #         # try:
            #         if  graph_id in mol_prop['chembl']:
            #             graph_data = mol_smiles[graph_id]
            #             # graph_data = mol_prop['smiles'][graph_id]
            #             # if text_graph != graph_data:
            #             #     print(graph_id,text_graph,graph_data)
            #             chemid=mol_prop['chembl'][graph_id]
            #             data_all += [(graph_data, text_data, label, chemid,assay_chembl_id)]
            #         # except:
            #         #     pass
            return data_all

        assays=list(assay_dict.keys())

        # train_index = np.random.choice(np.arange(len(assays)), size=int(len(assays)*0.8), replace=False)
        # valid_index=np.delete(np.arange(len(assays)), train_index)
        #
        # train_assays=[assays[i] for i in train_index]
        # valid_assays=[assays[i] for i in valid_index]
        #
        # if args.assay_split_transductive:
        #     train_graph_index =np.random.choice(np.arange(len(mol_smiles)), size=int(len(mol_smiles)*0.8), replace=False)
        #     valid_graph_index = np.delete(np.arange(len(mol_smiles)), train_graph_index)
        # else:
        train_graph_index =np.arange(len(mol_smiles))
        #     valid_graph_index = np.arange(len(mol_smiles))
        #
        # data_all_train=generate_data(train_assays,set(train_graph_index))
        # data_all_valid=generate_data(valid_assays,set(valid_graph_index))
        # print(len(data_all_train))
        # print(len(data_all_valid))
            #     if len(data_all)>1000:
            #         break
            # if len(data_all) > 1000:
            #     break
        # print(len(data_all))
        # train_number=int(len(data_all)*0.8)
        # data_all_train=data_all[0:train_number]
        # data_all_valid=data_all[train_number:]

        data_all_train = generate_data(assays, set(train_graph_index))

        file_prefix="assay_graph_only_"

        assay_ids_column_name=['cla_'+id for id in assay_ids]
        column_names=['graph']+assay_ids_column_name+['chemid']
        df = pd.DataFrame(data_all_train,
                          columns=column_names)

        def transform_column(x):
            x[x==0]=-100
            x[x==-1]=0
            return x

        df[assay_ids_column_name]=df[assay_ids_column_name].apply(transform_column)
        df.to_csv(file_prefix+'train_'+args.file_save_name)


        # f = open('assay_graph_text.txt', 'w')
        # for data_line in data_all:
        #     f.writelines([data_line, '\n'])
        #
        # f.close()

        # print(1)

