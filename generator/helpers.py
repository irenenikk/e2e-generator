import pickle

def save_to_pickle(data, target_file):
    with open(target_file, 'wb') as out_file:
        pickle.dump(data, out_file)

def load_from_pickle(file):
    with open(file, 'rb') as out_file:
        return pickle.load(out_file)
