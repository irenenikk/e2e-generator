import re
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import os
import json
from nltk.corpus import wordnet

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slug2slug')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EVAL_DIR = os.path.join(ROOT_DIR, 'eval')
METRICS_DIR = os.path.join(ROOT_DIR, 'metrics')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
PREDICTIONS_DIR = os.path.join(ROOT_DIR, 'predictions')
PREDICTIONS_BATCH_DIR = os.path.join(PREDICTIONS_DIR, 'batch')
PREDICTIONS_BATCH_LEX_DIR = os.path.join(PREDICTIONS_DIR, 'batch_lex')
PREDICTIONS_BATCH_EVENT_DIR = os.path.join(PREDICTIONS_DIR, 'batch_event')
SLOT_ALIGNER_DIR = os.path.join(ROOT_DIR, 'slot_aligner')
SLOT_ALIGNER_ALTERNATIVES = os.path.join(SLOT_ALIGNER_DIR, 'alignment', 'alternatives.json')
T2T_DIR = os.path.join(ROOT_DIR, 't2t')
TOOLS_DIR = os.path.join(ROOT_DIR, 'tools')
TTEST_DIR = os.path.join(ROOT_DIR, 'ttest')
TTEST_DATA_DIR = os.path.join(ROOT_DIR, 'ttest', 'data')
TTEST_SCORES_DIR = os.path.join(ROOT_DIR, 'ttest', 'scores')

# Dataset paths
E2E_DATA_DIR = os.path.join(DATA_DIR, 'rest_e2e')
TV_DATA_DIR = os.path.join(DATA_DIR, 'tv')
LAPTOP_DATA_DIR = os.path.join(DATA_DIR, 'laptop')
HOTEL_DATA_DIR = os.path.join(DATA_DIR, 'hotel')
VIDEO_GAME_DATA_DIR = os.path.join(DATA_DIR, 'video_game')

# Script paths
METRICS_SCRIPT_PATH = os.path.join(METRICS_DIR, 'measure_scores.py')

# Constants
COMMA_PLACEHOLDER = ' __comma__'
DELEX_PREFIX = '__slot_'    # Important to use special symbols that do not get tokenized (such as '_')
DELEX_SUFFIX = '__'
EMPH_TOKEN = '__emph__'
CONTRAST_TOKEN = '__contrast__'
CONCESSION_TOKEN = '__concession__'

def get_unaligned_and_hallucinated_slots(utt, mr):
    """Counts unrealized and hallucinated slots in an utterance."""

    slots_found, slots_hallucinated = find_all_slots(utt, mr)
    # Identify slots that were realized incorrectly or not mentioned at all in the utterance
    unaligned_slots = [slot for slot in mr if slot not in slots_found]

    return unaligned_slots, slots_hallucinated

def find_all_slots(utt, mr):
    slots_found = set()
    slots_hallucinated = set()

    utt, utt_tok = __preprocess_utterance(utt)
    delex_placeholders = extract_delex_placeholders(utt)

    for slot, value in mr.items():
        slot_root = slot.rstrip(string.digits)
        value = value.lower()

        pos, is_hallucinated = find_slot_realization(utt, utt_tok, slot_root, value, delex_placeholders)

        if pos >= 0:
            slots_found.add(slot)

        if is_hallucinated:
            slots_hallucinated.add(slot)

    return slots_found, slots_hallucinated


def __preprocess_utterance(utt):
    """Removes certain special symbols from the utterance, and reduces all whitespace to a single space.
    Returns the utterance both in string form and tokenized.
    """

    utt = re.sub(r'[-/]', ' ', utt.lower())
    utt = re.sub(r'\s+', ' ', utt)
    utt_tok = [w.strip('.,!?') if len(w) > 1 else w for w in word_tokenize(utt)]

    return utt, utt_tok

def extract_delex_placeholders(utt):
    """Extracts delexicalized placeholders from the utterance."""

    pattern = DELEX_PREFIX + '.*?' + DELEX_SUFFIX

    return set(re.findall(pattern, utt))

def find_slot_realization(text, text_tok, slot, value_orig, delex_slot_placeholders,
                          soft_align=False, match_name_ref=False):
    pos = -1
    is_hallucinated = False

    value = re.sub(r'[-/]', ' ', value_orig)

    # TODO: remove auxiliary slots ('da' and '__.*?__') beforehand
    if slot == 'da':
        pos = 0
    elif re.match(r'__.*?__', slot):
        pos = 0
    else:
        delex_slot = check_delex_slots(slot, delex_slot_placeholders)
        if delex_slot is not None:
            pos = text.find(delex_slot)
            delex_slot_placeholders.remove(delex_slot)

            # Find hallucinations of the delexed slot
            slot_cnt = text.count(delex_slot)
            if slot_cnt > 1:
                print('HALLUCINATED SLOT:', slot)
                is_hallucinated = True
        else:
            # Universal slot values
            if value == 'dontcare':
                if dontcare_realization(text, slot, soft_match=True):
                    # TODO: get the actual position
                    pos = 0
                    for slot_stem in reduce_slot_name(slot):
                        slot_cnt = text.count(slot_stem)
                        if slot_cnt > 1:
                            print('HALLUCINATED SLOT:', slot)
                            is_hallucinated = True
            elif value == 'none':
                if none_realization(text, slot, soft_match=True):
                    # TODO: get the actual position
                    pos = 0
                    for slot_stem in reduce_slot_name(slot):
                        slot_cnt = text.count(slot_stem)
                        if slot_cnt > 1:
                            print('HALLUCINATED SLOT:', slot)
                            is_hallucinated = True
            elif value == '':
                for slot_stem in reduce_slot_name(slot):
                    pos = text.find(slot_stem)
                    if pos >= 0:
                        slot_cnt = text.count(slot_stem)
                        if slot_cnt > 1:
                            print('HALLUCINATED SLOT:', slot)
                            is_hallucinated = True
                        break

            elif slot == 'name' and match_name_ref:
                pos = text.find(value)
                if pos < 0:
                    for pronoun in ['it', 'its', 'they', 'their', 'this']:
                        _, pos = find_first_in_list(pronoun, text_tok)
                        if pos >= 0:
                            break

            # E2E restaurant dataset slots
            elif slot == 'familyfriendly':
                pos = align_boolean_slot(text, text_tok, slot, value)
            elif slot == 'food':
                pos = foodSlot(text, text_tok, value)
            elif slot in ['area', 'eattype']:
                if soft_align:
                    pos = align_categorical_slot(text, text_tok, slot, value,
                                                 mode='first_word')
                else:
                    pos = align_categorical_slot(text, text_tok, slot, value,
                                                 mode='exact_match')
            elif slot == 'pricerange':
                if soft_align:
                    pos = align_scalar_slot(text, text_tok, slot, value,
                                            slot_stem_only=True)
                else:
                    pos = align_scalar_slot(text, text_tok, slot, value,
                                            slot_stem_only=False)
            elif slot == 'customerrating':
                if soft_align:
                    pos = align_scalar_slot(text, text_tok, slot, value,
                                            slot_mapping=customerrating_mapping['slot'],
                                            value_mapping=customerrating_mapping['values'],
                                            slot_stem_only=True)
                else:
                    pos = align_scalar_slot(text, text_tok, slot, value,
                                            slot_mapping=customerrating_mapping['slot'],
                                            value_mapping=customerrating_mapping['values'],
                                            slot_stem_only=False)

            # Fall back to finding verbatim slot realization
            elif value in text:
                if len(value) > 4 or ' ' in value:
                    pos = text.find(value)
                else:
                    _, pos = find_first_in_list(value, text_tok)

                # value_cnt = text.count(value)
                # if value_cnt > 1:
                #     print('HALLUCINATED SLOT:', slot, value)
                #     is_hallucinated = True

    return pos, is_hallucinated

def check_delex_slots(slot, delex_slots):
    if delex_slots is None:
        return None

    for delex_slot in delex_slots:
        if slot in delex_slot:
            return delex_slot

    return None

def dontcare_realization(text, slot, soft_match=False):
    text = re.sub('\'', '', text.lower())
    text_tok = word_tokenize(text)

    for slot_stem in reduce_slot_name(slot):
        slot_stem_plural = get_plural(slot_stem)

        if slot_stem in text_tok or slot_stem_plural in text_tok or slot in text_tok:
            if soft_match:
                return True

            for x in ['any', 'all', 'vary', 'varying', 'varied', 'various', 'variety', 'different',
                      'unspecified', 'irrelevant', 'unnecessary', 'unknown', 'n/a', 'particular', 'specific', 'priority', 'choosy', 'picky',
                      'regardless', 'disregarding', 'disregard', 'excluding', 'unconcerned', 'matter', 'specification',
                      'concern', 'consideration', 'considerations', 'factoring', 'accounting', 'ignoring']:
                if x in text_tok:
                    return True
            for x in ['no preference', 'no predetermined', 'no certain', 'wide range', 'may or may not',
                      'not an issue', 'not a factor', 'not important', 'not considered', 'not considering', 'not concerned',
                      'without a preference', 'without preference', 'without specification', 'without caring', 'without considering',
                      'not have a preference', 'dont have a preference', 'not consider', 'dont consider', 'not mind', 'dont mind',
                      'not caring', 'not care', 'dont care', 'didnt care']:
                if x in text:
                    return True
            if ('preference' in text_tok or 'specifics' in text_tok) and ('no' in text_tok):
                return True
    
    return False

def reduce_slot_name(slot):
    reduction_map = {
        'availableonsteam': ['steam'],
        'batteryrating': ['battery'],
        'customerrating': ['customer'],
        'driverange': ['drive'],
        'ecorating': ['eco'],
        'eattype': ['eat'],
        'familyfriendly': ['family', 'families', 'kid', 'kids', 'child', 'children'],
        'genres': ['genre'],
        'haslinuxrelease': ['linux'],
        'hasmacrelease': ['mac'],
        'hasmultiplayer': ['multiplayer', 'friends', 'others'],
        'hasusbport': ['usb'],
        'hdmiport': ['hdmi'],
        'isforbusinesscomputing': ['business'],
        'playerperspective': ['perspective'],
        'platforms': ['platform'],
        'powerconsumption': ['power'],
        'pricerange': ['price'],
        'releaseyear': ['year'],
        'screensize': ['screen'],
        'screensizerange': ['screen'],
        'weightrange': ['weight']
    }

    return reduction_map.get(slot, [slot])

def none_realization(text, slot, soft_match=False):
    text = re.sub('\'', '', text.lower())
    text_tok = word_tokenize(text)

    for slot_stem in reduce_slot_name(slot):
        if slot_stem in text_tok:
            if soft_match:
                return True

            for x in ['information', 'info', 'inform', 'results', 'requirement', 'requirements', 'specification', 'specifications']:
                if x in text_tok and ('no' in text_tok or 'not' in text_tok or 'any' in text_tok):
                    return True
    
    return False

def align_categorical_slot(text, text_tok, slot, value, mode='exact_match'):
    # TODO: load alternatives only once
    alternatives = get_slot_value_alternatives(slot)

    pos = find_value_alternative(text, text_tok, value, alternatives, mode=mode)

    return pos

def get_slot_value_alternatives(slot):
    with open(SLOT_ALIGNER_ALTERNATIVES, 'r') as f_alternatives:
        alternatives_dict = json.load(f_alternatives)

    return alternatives_dict.get(slot, {})

def find_value_alternative(text, text_tok, value, alternatives, mode):
    leftmost_pos = -1

    # Parse the item into tokens according to the selected mode
    if mode == 'first_word':
        value_alternatives = [value.split(' ')[0]]  # Single-element list
    elif mode == 'any_word':
        value_alternatives = value.split(' ')  # List of elements
    elif mode == 'all_words':
        value_alternatives = [value.split(' ')]  # List of single-element lists
    else:
        value_alternatives = [value]  # Single-element list

    # Merge the tokens with the item's alternatives
    if value in alternatives:
        value_alternatives += alternatives[value]

    # Iterate over individual tokens of the item
    for value_alt in value_alternatives:
        # If the item is composed of a single token, convert it to a single-element list
        if not isinstance(value_alt, list):
            value_alt = [value_alt]

        # Keep track of the positions of all the item's tokens
        positions = []
        for tok in value_alt:
            if len(tok) > 4 or ' ' in tok:
                # Search for long and multi-word values in the string representation
                pos = text.find(tok)
            else:
                # Search for short single-word values in the tokenized representation
                _, pos = find_first_in_list(tok, text_tok)
            positions.append(pos)

        # If all tokens of one of the value's alternatives are matched, record the match and break
        if all([p >= 0 for p in positions]):
            leftmost_pos = min(positions)
            break

    return leftmost_pos

def find_first_in_list(val, lst):
    idx = -1
    pos = -1

    for i, elem in enumerate(lst):
        if val == elem:
            idx = i

    if idx >= 0:
        # Calculate approximate character position of the matched value
        punct_cnt = lst[:idx].count('.') + lst[:idx].count(',')
        pos = len(' '.join(lst[:idx])) + 1 - punct_cnt

    return idx, pos

# TODO @food has 24 failures which are acceptable to remove the slot
def foodSlot(text, text_tok, value):
    value = value.lower()

    pos = text.find(value)
    if pos >= 0:
        return pos
    elif value == 'english':
        return text.find('british')
    elif value == 'fast food':
        return text.find('american style')
    else:
        text_tok = word_tokenize(text)
        for token in text_tok:
            # FIXME warning this will be slow on start up
            synsets = wordnet.synsets(token, pos='n')
            synset_ctr = 0

            for synset in synsets:
                synset_ctr += 1
                hypernyms = synset.hypernyms()

                # If none of the first 3 meanings of the word has "food" as hypernym, then we do not want to
                #   identify the word as food-related (e.g. "center" has its 14th meaning associated with "food",
                #   or "green" has its 7th meaning accociated with "food").
                while synset_ctr <= 3 and len(hypernyms) > 0:
                    lemmas = [l.name() for l in hypernyms[0].lemmas()]

                    if 'food' in lemmas:
                        # DEBUG PRINT
                        # print(token)

                        return text.find(token)
                    # Skip false positives (e.g. "a" in the meaning of "vitamin A" has "food" as a hypernym,
                    #   or "coffee" in "coffee shop" has "food" as a hypernym). There are still false positives
                    #   triggered by proper nouns containing a food term, such as "Burger King" or "The Golden Curry".
                    elif 'vitamin' in lemmas:
                        break
                    elif 'beverage' in lemmas:
                        break

                    # Follow the hypernyms recursively up to the root
                    hypernyms = hypernyms[0].hypernyms()

    return pos
