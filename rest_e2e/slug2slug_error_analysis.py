import pandas as pd
import numpy as np

if __name__ == '__main__':
    data = pd.read_csv('trainset_errors.csv')
    N = 50
    for i, row in data.sample(N).iterrows():
        print('mr:', row['mr'])
        print('utterance:', row['ref'])
        print('err:', row['errors'])
        print('incorrect slots:', row['incorrect slots'])
        import ipdb; ipdb.set_trace()
        print('-------------')