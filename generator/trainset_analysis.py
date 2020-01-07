import pandas as pd
import nltk
import argparse
from data_preprocessing import build_slot_columns
import pgmpy
from pgmpy.estimators import ConstraintBasedEstimator, MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling
from util import save_to_pickle

parser = argparse.ArgumentParser(description='Analyse the probabilities of slots appearing from training data')
parser.add_argument("training_data", type=str,
                    help="The path to the training data file")
parser.add_argument("target_file", type=str, default='cpd_model.pkl',
                    help="Where to store the conditional probability tables created")

def check_pairwise_independences(correlation_cols, est):
    independent = []
    dependent = []
    for s1 in correlation_cols:
        for s2 in correlation_cols:
            tupl = (s1, s2)
            dep = est.test_conditional_independence(s1, s2)
            if dep:
                dependent.append(tupl)
            else:
                independent.append(tupl)
            print(s1, 'and', s2, 'independence :', dep)
    return dependent, independent

def main(data_file, target_file):
    raw_data = pd.read_csv(data_file)
    print('Found data of shape', raw_data.shape)
    slot_cols = build_slot_columns(raw_data)
    data = pd.concat([raw_data, slot_cols], axis=1)
    # don't calculate correlations for name
    correlation_cols = slot_cols.columns.drop(['name']).values
    print('Detecting correlations between', correlation_cols)
    for col in correlation_cols:
        not_present = data[col].isnull()
        data.loc[not_present, col] = 0
        data.loc[~not_present, col] = 1
    slot_data = data[correlation_cols]
    # build a model to infer underlying DAG
    # based on these examples: https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/9.%20Learning%20Bayesian%20Networks%20from%20Data.ipynb
    est = ConstraintBasedEstimator(slot_data)
    dag_edges = est.estimate(significance_level=0.01).edges()
    # create a new model to estimate conditional probability tables
    model = BayesianModel(dag_edges)
    print('Estimating conditional probability tables for each variable')
    mle = MaximumLikelihoodEstimator(model, slot_data)
    cpd_tables = []
    for slot in correlation_cols:
        cpd = mle.estimate_cpd(slot)
        cpd_tables.append(cpd)
    model.add_cpds(*cpd_tables)
    if not target_file.endswith('.pkl'):
        target_file += '.pkl'
    print('Storing the Bayesian network to', target_file)
    save_to_pickle(model, target_file)

if __name__ == '__main__':
    args = parser.parse_args()
    data_file = args.training_data
    target_file = args.target_file
    main(data_file, target_file)

