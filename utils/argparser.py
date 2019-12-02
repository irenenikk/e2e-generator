import argparse

parser = argparse.ArgumentParser(description='Train SCLSTM')
parser.add_argument('data_path', type=string,
                    help='path to the nlg dataset to use in training')

def get_args():
    return parser.parse_args()
