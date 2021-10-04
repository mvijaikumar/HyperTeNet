# tested
import torch
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import itertools,pdb,json,pickle
'''
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
'''
def store_npy(fname,obj):
    '''store npy <obj> into <fname>'''
    with open(fname, "wb") as fout:
        np.save(fout,obj)

def load_npy(fname):
    '''load npy obj from <fname>'''
    with open(fname, "rb") as fin:
        obj = np.load(fin)
    return obj

def store_json(fname,obj):
    '''store json <obj> into <fname>'''
    with open(fname, "w") as fout:
        json.dump(obj,fout)

def load_json(fname):
    '''load json obj from <fname>'''
    with open(fname, "r") as fin:
        obj = json.load(fin)
    return obj

def store_pickle(fname,obj):
    '''store pickle <obj> into <fname>'''
    with open(fname, "wb") as fout:
        #pickle.dump(obj,fout,protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(obj,fout,protocol=2)#pickle.HIGHEST_PROTOCOL)

def load_pickle(fname):
    '''load pickle obj from <fname>'''
    try:
        with open(fname, "rb") as fin:
            obj = pickle.load(fin)
    except:
        obj = torch.load(fname, map_location='cpu')
    return obj

def get_keys_as_set(dct):
    '''get <dct> keys as set'''
    return set(dct.keys())

def get_value_lists_as_set(dct):
    '''get <dct> list of values as set'''
    return set(itertools.chain.from_iterable(dct.values()))

def get_value_lists_as_list(dct):
    '''get <dct> list of values as set'''
    return list(itertools.chain.from_iterable(dct.values()))

def get_values_as_set(dct):
    '''get <dct> values as set'''
    return set(dct.values())

def get_file_to_dict(fname,delimiter=None):
    '''get dictionary from given key-value <fname>'''
    dct = defaultdict(list)
    with open(fname, "rb") as fin:
        for line in fin:
            if delimiter == None:
                key, val = line.decode().strip().split()
            else:
                key, val = line.decode().strip().split(delimiter)
            dct[int(key)].append(int(val))
    return dct

def store_dct_as_file(fname,dct):
    with open(fname,'w') as fout:
        for key in dct:
            fout.write(str(key) + ' ' + str(dct[key]) + "\n")

def store_dct_list_as_file(fname,dct,reverse_key_val=False):
    with open(fname,'w') as fout:
        for key in dct:
            for val in dct[key]:
                if reverse_key_val == True:
                    fout.write(str(val) + ' ' + str(key) + "\n")
                else:
                    fout.write(str(key) + ' ' + str(val) + "\n")

def get_ids(entity_set,one_index=False):
    '''get assigned ids for given <entity_set>'''
    id_dct = dict()
    id_val = 0
    if one_index == True: id_val += 1
    for entity in entity_set:
        id_dct[entity] = id_val
        id_val += 1

    return id_dct

def assign_ids(in_dct, key_id_dct, value_id_dct):
    '''assign ids for given <in_dct> using <key_id_dct> and <value_id_dct>'''
    out_dct = dict()
    for key in in_dct:
        value_lst = []
        for value in in_dct[key]:
            value_lst.append(value_id_dct[value])
        out_dct[key_id_dct[key]] = value_lst
    return out_dct

def get_dct_to_mat(dct, num_row, num_col, padding_value=0,dtype=int):
        mat = np.full((num_row,num_col),padding_value,dtype=dtype)
        for lst in dct:
            items_arr = np.array(dct[lst])
            leng      = len(items_arr)
            if leng >= num_col:
                mat[lst,:]      = items_arr[-num_col:]
            else:
                mat[lst,-leng:] = items_arr
        return mat

def get_dct_for_user(dct):
    res_dct = dict()
    for key in dct:
        res_dct[key[0]] = dct[key]
    return res_dct,len(dct[key])

def remove_duplicates_from_list(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def remove_duplicates_from_dct_of_list(dct):
    newdct = defaultdict(list)
    for key in dct:
        newdct[key] = remove_duplicates_from_list(dct[key])
    return newdct

def get_item_embed_dim(filename):
    item_embed = dict()
    with open(filename, "r") as f:
        line = f.readline().strip()
        toks = line.replace("\n","").split("::")
        itemid = int(toks[0])
        embed  = np.array(toks[1].split(" ")).astype(np.float)
        attr_dim = len(embed)
        return attr_dim

def load_embed_file_as_dict(filename):
    item_embed = dict()
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            toks = line.replace("\n","").split("::")
            itemid = int(toks[0])
            embed  = np.array(toks[1].split(" ")).astype(np.float)
            item_embed[itemid] = embed
            line = f.readline()
    return item_embed

def get_max_entity_id(filename):
    dct = load_embed_file_as_dict(filename)
    return max(dct.keys()) + 1

def load_embed_as_mat(filename, row, col):
    mat = np.zeros((row, col),dtype=np.float32)
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            toks    = line.replace("\n","").split("::")
            itemid  = int(toks[0])
            embed   = np.array(toks[1].split(" ")).astype(np.float)
            mat[itemid] = embed
            line = f.readline()
    return mat

def store_npy_to_csv(fname, obj, delimiter='\t'):
    np.savetxt(fname, obj, delimiter)
