
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys
from data_preprocessing import build_slot_columns, preprocess_data, reconstruct_mr, tokenize, add_space_to_punctuation

def load_data_tensors(data_file, num_examples=None):
    """ Read data from a CSV file, and convert into lookup tensors pointing to tokens in the text. """
    raw_data = pd.read_csv(data_file)
    if num_examples is not None: 
        raw_data = raw_data.head(num_examples)
    # extract the slot infromation into separate columns
    data_columns = build_slot_columns(raw_data)
    # add the slot columns into the dataframe
    data = pd.concat([raw_data, data_columns], axis=1)
    data = preprocess_data(data)
    data = reconstruct_mr(data, data_columns.columns)
    data['new_mr'] = data['new_mr'].apply(add_space_to_punctuation)
    input_tensor, mr_word2idx, mr_idx2word = tokenize(data['new_mr'])
#    print(mr_word2idx)
    target_tensor, ref_word2idx, ref_idx2word = tokenize(data['ref'])
    return input_tensor, target_tensor, ref_word2idx, ref_idx2word, mr_word2idx, mr_idx2word

def load_text_data(data_file, num_examples=None):
    """ Load text data and return as a dataframe. """
    raw_data = pd.read_csv(data_file)
    if num_examples is not None: 
        raw_data = raw_data.head(num_examples)
    # and start and end tags to data
    data_columns = build_slot_columns(raw_data)
    data = pd.concat([raw_data, data_columns], axis=1)
    return data

def tiny_analysis(data):
    columns = [col for col in data.columns if col not in ['ref', 'mr']]
    for column in columns:
        print('Unique values in slot', column, ':', data[column].unique(), '(', data[column].nunique(), ')')

def max_length(tensor):
    return max(len(t) for t in tensor)

def get_dataset(input_tensor, output_tensor, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor)).shuffle(len(input_tensor))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def create_dataset(input_tensor, target_tensor, batch_size, embedding_dim, units, test_size=0.2):    
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=test_size)
    steps_per_epoch = len(input_tensor_train)//batch_size
    train_dataset = get_dataset(input_tensor_train, target_tensor_train, batch_size)
    val_dataset = get_dataset(input_tensor_val, target_tensor_val, batch_size)
    return train_dataset, val_dataset, steps_per_epoch

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Please give path to data file')
        exit()
    data_path = sys.argv[1]
    data = load_text_data(data_path)
    tiny_analysis(data)