from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
import helpers

def sample_slots(model_file):
    model = helpers.load_from_pickle(model_file)
    inference = BayesianModelSampling(model)
    sample = inference.forward_sample(size=1, return_type='recarray')
    # return a list of the column names which had presence
    return [name for var, name in zip(sample.view('<i8'), sample.dtype.names) if var == 1]
