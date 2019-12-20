
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import sys

DELEXICALIZED_MRS = ['name', 'near']
MR_SUFFIX = '_place'

def preprocess_mr(mr_info):
    slots = get_slots(mr_info)
    mr_info = " , ".join([slot + "[" + slot_value.replace(" ", "_") + "]" for slot, slot_value in slots.items()])
    mr_info = delexicalize_mr(mr_info, slots)
    mr_info = add_space_to_punctuation(mr_info)
    mr_info = mr_info.lower()
#    print('PROCESSED MR INFO', mr_info)
    return mr_info

def delexicalize_mr(mr_info, slots):
    for mr in DELEXICALIZED_MRS:
        if mr in slots.keys():
#            print('replace', slots[mr])
            mr_info = mr_info.replace(slots[mr], mr + MR_SUFFIX)
#            print('Processed', mr_info)
    return mr_info

def get_slots(mr_info):
    """ Build a dict containing all the slots in an MR. """
    slots = {}
    MRs = mr_info.split(', ')
    for mr in MRs:
        slot = mr[:mr.find('[')]
        value = mr[mr.find('[')+1:mr.find(']')]
        slots[slot] = value.replace(' ', '_')
    return slots

def delexicalize_ref(row, mr):
    mr_val = row[mr]
    if mr_val is None:
        return row['ref']
    return row['ref'].replace(mr_val, mr + MR_SUFFIX)

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

def build_slot_columns(data):
    """ Build a dataframe with each slot as a column. """
    data.columns = map(str.lower, data.columns)
    slot_infos = data['mr'].apply(get_slots).values
    # get all possible slots from data
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
    for mr in mrs:
        for i in range(len(data)):
            if data.iloc[i][mr] is not None:
                new_mr[i] +=  mr + '[' + data.iloc[i][mr]+ '], '
    data['new_mr'] = new_mr
    return data
