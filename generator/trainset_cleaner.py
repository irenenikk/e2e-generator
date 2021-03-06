####### PLEASE RUN THIS IN THE SLUG2SLUG FOLDER TO USE THEIR MODULES #######

import pandas as pd
import argparse
from slug2slug.slot_aligner.data_analysis import score_slot_realizations
import os
from data_preprocessing import build_slot_columns, reconstruct_mr

parser = argparse.ArgumentParser(description='Clean training data using the Slug2Slug slot aligner')
parser.add_argument("training_data_folder", type=str,
                    help="The path to the folder containing training data")
parser.add_argument("training_data_file", type=str, default='trainset.csv',
                    help="The name of the training data file to be cleaned")
parser.add_argument("-i", "--ignore-old", default=False, action='store_true',
                    help="Redo error analysis even if old analysis file is found")

slot_column_order = ['name', 'eatType', 'food', 'priceRange', 'customer rating', 'area', 'familyFriendly', 'near']

def main(folder_name, filename, ignore_old):
    # use slug2slug to analyse which slots are present
    # this creates a new csv file
    error_file_suffix = '_errors'
    cleaned_file_suffix = '_cleaned'
    filename_prefix = os.path.splitext(filename)[0]
    filename_out = filename_prefix + error_file_suffix + '.csv'
    if ignore_old:
        filename_out = score_slot_realizations(folder_name, filename, error_file_suffix)
    print('Using error file', filename_out)
    error_df = pd.read_csv(os.path.join(folder_name, filename_out))
    print('Error df length', len(error_df))
    orig_data = pd.read_csv(os.path.join(folder_name, filename))
    print('Original data length', len(orig_data))
    slot_column_data = build_slot_columns(error_df, remove_whitespace=False)
    comb_data = pd.concat([error_df, slot_column_data], axis=1)
    column_order = comb_data.columns
    if len(comb_data) != len(error_df):
        raise ValueError('Analysed trainset and original trainset are not the same length')
    # find data points which were marked to have an error
    erronous_indices = comb_data[comb_data['errors'] > 0].index
    for i, err_row in comb_data.loc[erronous_indices].iterrows():
        incorrect_slots = err_row['incorrect slots'].split(', ')
        # this has to be done because slug2slug gives the column name in lowercase
        incorrect_cols = comb_data.columns.map(lambda x: x.lower()).isin(incorrect_slots)
        # remove the incorrect slot values
        comb_data.loc[i, incorrect_cols] = None
    comb_data = comb_data[column_order]
    print('Cleaned slots from', len(erronous_indices), 'instances')
    # reconstruct the slot column data frame with the cleaned values
    new_mr = reconstruct_mr(comb_data, slot_column_order)
    comb_data['new_mr'] = new_mr
    cleaned_data = comb_data[~(comb_data['new_mr'] == '')]
    cleaned_data = cleaned_data[['new_mr', 'ref']]
    cleaned_data = cleaned_data.rename(columns={'new_mr': 'mr'})
    # cleaned data may now have sentences where there are no valid mrs
    # remove those before saving
    cleaned_out = os.path.join(folder_name, filename_prefix + cleaned_file_suffix + '.csv')
    print('Final cleaned data length', len(cleaned_data))
    print('Writing cleaned dataset to', cleaned_out)
    cleaned_data.to_csv(cleaned_out, index=False)

if __name__ == '__main__':
    args = parser.parse_args()
    folder_name = args.training_data_folder
    filename = args.training_data_file
    ignore_old = args.ignore_old
    main(folder_name, filename, ignore_old)