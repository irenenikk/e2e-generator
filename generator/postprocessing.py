from data_preprocessing import get_slots, MR_SUFFIX

def dedelexicalise_ref(ref, mr_slots):
    for mr in mr_slots.keys():
        mr_val = mr_slots[mr]
        ref = ref.replace(mr + MR_SUFFIX, mr_val.replace('_', ' '))
    return ref

def postprocess_utterance(utt, mr_slots):
    """ Postprocess a generated utterance. mr_slots should be a dict where keys are slots and values are slot values. """
    dedelexicalised_utt = dedelexicalise_ref(utt, mr_slots)
    without_end = dedelexicalised_utt.replace('<end>', '')
    return without_end