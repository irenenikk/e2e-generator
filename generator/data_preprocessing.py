
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import sys
from collections import OrderedDict

DELEXICALIZED_MRS = ['name', 'near']
MR_SUFFIX = '_place'

def preprocess_mr(mr_info):
    slots = get_slots(mr_info)
    mr_info = " , ".join([slot + "[" + slot_value.replace(" ", "_") + "]" for slot, slot_value in slots.items()])
    mr_info = delexicalize_mr(mr_info, slots)
    #mr_info = add_space_to_punctuation(mr_info)
    mr_info = mr_info.lower()
#    print('PROCESSED MR INFO', mr_info)
    return mr_info

def delexicalize_mr(mr_info, slots):
    for mr in DELEXICALIZED_MRS:
        if mr in slots.keys():
            mr_info = mr_info.replace(slots[mr], mr + MR_SUFFIX)
    return mr_info

def get_slots(mr_info, remove_whitespace=True):
    """ Build a dict containing all the slots in an MR. """
    slots = OrderedDict()
    MRs = mr_info.split(', ')
    for mr in MRs:
        slot = mr[:mr.find('[')]
        value = mr[mr.find('[')+1:mr.find(']')]
        if remove_whitespace:
            value = value.replace(' ', '_')
        slots[slot] = value
    return slots

def delexicalize_ref(row, mr):
    mr_val = row[mr]
    if mr_val is None:
        return row['ref']
    return row['ref'].replace(mr_val.replace('_', ' '), mr + MR_SUFFIX)

def delexicalize(data):
    """ Delexicalizes the dataset using slot columns. """
    for mr in DELEXICALIZED_MRS:
        data['ref'] = data[['ref', mr]].apply(lambda row: delexicalize_ref(row, mr), axis=1)
        data[mr] = mr + MR_SUFFIX
    return data

def add_space_to_punctuation(text):
    text = re.sub(r"([?.!,:;'\"])", r" \1 ", text)
    return re.sub(r"[' ']+", " ", text)

def preprocess_data(data):
    """ Add start and end tags to reference text and delexicalize. """
    data['ref'] = data['ref'].apply(add_space_to_punctuation)
    data['ref'] = data['ref'].apply(lambda x: '<start> ' + x + ' <end>')
    data = delexicalize(data)
    return data

def build_slot_columns(data, remove_whitespace=True):
    """ Build a dataframe with each slot as a column. """
    data.columns = map(str.lower, data.columns)
    slot_infos = data['mr'].apply(lambda x: get_slots(x, remove_whitespace=remove_whitespace)).values
    # get all possible slots from data
    # ensure that the order stays the same by sorting
    # a seq2seq model expects structure in input
    # this same method is used to load training and test data
    all_slots = set([key for d in slot_infos for key in d.keys()])
    # create a new pandas dataframe of interesting columns
    mr_slots = {}
    for slot in all_slots:
        slot_values = []
        for i in range(len(slot_infos)):
            slot_info = slot_infos[i]
            slot_value = slot_info[slot] if slot in slot_info else None
            slot_values.append(slot_value)
        mr_slots[slot] = slot_values
    return pd.DataFrame.from_dict(mr_slots)

def tokenize(texts):
    text_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    text_tokenizer.fit_on_texts(texts)
    tensor = text_tokenizer.texts_to_sequences(texts)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, text_tokenizer.word_index, text_tokenizer.index_word

def reconstruct_mr(data, mrs):
    new_mr = [''] * len(data)
    # ensure the data is always in the same order
    for mr in mrs:
        for i in range(len(data)):
            if data.iloc[i][mr] is not None:
                info = mr + '[' + data.iloc[i][mr]+ ']'
                if new_mr[i] == '':
                    new_mr[i] = info
                else:
                    new_mr[i] +=  ', ' + info
    return new_mr
