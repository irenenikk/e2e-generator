from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
import numpy as np
import helpers

#all_slots = ['area', 'customer rating', 'eatType', 'familyFriendly', 'food', 'near', 'priceRange']

def sample_slots(model_info_file, mr_slot_names):
    model_info = helpers.load_from_pickle(model_info_file)
    model = model_info['model']
    inference = BayesianModelSampling(model)
    # use the missing mr slots as evidence
    all_slots = model_info['all_slots']
    missing_slots = [mr for mr in all_slots if mr not in mr_slot_names]
    evidence = [State(mr, 0) for mr in missing_slots]
    inference = BayesianModelSampling(model)
    # don't allow empty samples
    sampled_slots = []
    while(sampled_slots == []):
        sample = inference.rejection_sample(evidence=evidence, size=1, return_type='recarray')
        # return a list of the column names which had presence
        sampled_slots = [name for var, name in zip(sample.view('<i8'), sample.dtype.names) if var == 1]
    return sampled_slots
