from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
import numpy as np
import helpers

all_slots = ['area', 'customer rating', 'eatType', 'familyFriendly', 'food', 'near', 'priceRange']

def construct_model(model_info, mr_slot_names):
    edges = list(model_info['edges'])
    cpd_tables = model_info['cpd_tables']
    model = BayesianModel(edges)
    model.add_cpds(*cpd_tables)
    return model

def sample_slots(model_file, mr_slot_names):
    #model_info = helpers.load_from_pickle(model_info_file)
    #model = construct_model(model_info, mr_slot_names)
    model = helpers.load_from_pickle(model_file)
    inference = BayesianModelSampling(model)
    # use the missing mr slots as evidence
    missing_slots = [mr for mr in all_slots if mr not in mr_slot_names]
    print('missing slots', missing_slots)
    evidence = [State(mr, 0) for mr in missing_slots]
    print(evidence)
    #if model is None:
    #    print('not sampling')
    #    return mr_slot_names
    inference = BayesianModelSampling(model)
    sampled_slots = []
    while(sampled_slots == []):
        sample = inference.rejection_sample(evidence=evidence, size=1, return_type='recarray')
        # return a list of the column names which had presence
        sampled_slots = [name for var, name in zip(sample.view('<i8'), sample.dtype.names) if var == 1]
    return sampled_slots
