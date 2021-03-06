import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_mr, get_slots
from postprocessing import postprocess_utterance
import sys
from data_manager import load_data_tensors, load_text_data
from models import Encoder, BahdanauAttention, Decoder
import time
import os
import nltk
from slug2slug_aligner import get_unaligned_and_hallucinated_slots
sys.path.append('./')
import argparse
import helpers
import bayesian_sampler

BATCH_SIZE = 1
# optimal beam size found by Juraska
END_SYMBOL = '<end>'
START_SYMBOL = '<start>'

parser = argparse.ArgumentParser(description='Generate utterances from a trained E2E description generator')
parser.add_argument("test_data", type=str,
                    help="The path to the test data file")
parser.add_argument("-id", "--identifier", default='',
                    help="The identifier used to define checkpoint and training info directories")
parser.add_argument("-b", "--beam-width", type=int, default=0,
                    help="Size of beam to use in generation. If not specified use sampling.")
parser.add_argument("-s", "--sample-content", default=False, action="store_true",
                    help="Sample slots used in MR of utterance.")
parser.add_argument("-cpd", "--cpd-model-file", default='cpd_model.pkl',
                    help="Pickle file where the cpd model is stored")
parser.add_argument("-p", "--print-utt", default=False, action="store_true",
                    help="Print generations for dataset before estimating bleu score")


class BeamObj:
    def __init__(self, utterance, probability, last_id, last_hidden):
        self.utterance = utterance
        self.probability = probability
        # save the id of the last word
        # this will be used as input for the next timestep
        self.last_id = last_id
        # this is the decoder hidden state obtained from the previous prediction
        self.last_hidden = last_hidden

    def __repr__(self):
        return '{}({})'.format(self.utterance, self.probability)

def search_candidates(curr_prob, curr_utterance, dec_hidden, predictions, ref_idx2word, beam_size):
    """ Find the possible extensions to an utterance and their log-probabilities. """ 
    candidates = []
    # calculate probabilites if new predictions were added
    # check for all new predictions in the beam 
    if predictions.shape[0] != 1:
        raise ValueError('Batches not supported in beam search. ')
    preds = predictions[0]
    ids_by_prob = np.argsort(-preds)[:beam_size]
    for word_id in ids_by_prob:
        pred = preds[word_id]
        new_prob = curr_prob + pred
        new_utterance = (curr_utterance + " " + ref_idx2word[word_id]).strip()
        candidates += [BeamObj(new_utterance, new_prob, word_id, dec_hidden)]
    return candidates

def score_prediction(prediction, mr_slots):
    """ 
        Scores a complete utterance based on slot realisation.
        mr_slots should be a dict where keys are slots and values are slot values.
        The score function is taken from Juraska et al.
    """
    N = len(mr_slots.keys())
    # remove the whitespace placeholders
    orig_mr_slots = { k.replace(' ', ''): v.replace('_', ' ') for k, v in mr_slots.items() }
    # use Juraska's code to get erronous slots
    unaligned_slots, hallucinated_slots = get_unaligned_and_hallucinated_slots(prediction, orig_mr_slots)
    score = N/((len(unaligned_slots)+1)*(len(hallucinated_slots)+1))
    return score

def generation_done(beam_obj, training_info, end_token):
    """ Stop when end token is reached or the sentence is the maximal length. """
    return beam_obj.last_id == end_token or len(beam_obj.utterance) == training_info['max_length_targ']

def get_length_normalisation_denominator(utterance, alpha=0.9):
    """ As done in Google's NMT paper, who found optimal alpha to be between 0.6 and 0.7. """
    utt_len = len(utterance.split(" "))
    return ((5 + utt_len)**alpha)/((5+1)**alpha)

def evaluate_with_beam(encoder, decoder, mr_info, training_info, beam_size):
    attention_plot = np.zeros((training_info['max_length_targ'], training_info['max_length_inp']))
    processed_mr_info = preprocess_mr(mr_info)
    inputs = [training_info['mr_word2idx'][i.strip()] for i in processed_mr_info.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=training_info['max_length_inp'],
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)
    start_token = training_info['ref_word2idx'][START_SYMBOL]
    end_token = training_info['ref_word2idx'][END_SYMBOL]
    hidden = encoder.initialize_hidden_state(1)
    enc_out, forward_hidden, backward_hidden = encoder(inputs, hidden)
    dec_hidden = tf.keras.layers.Concatenate()([forward_hidden, backward_hidden])
    beam = [BeamObj('', 0, start_token, dec_hidden)]
    while(True):
        new_beams = []
        for beam_obj in beam:
            if generation_done(beam_obj, training_info, end_token):
                new_beams += [beam_obj]
                continue
            predictions, dec_hidden, attention_weights = decoder(tf.expand_dims([beam_obj.last_id], 0), beam_obj.last_hidden, enc_out)
            curr_prob = beam_obj.probability
            curr_utterance = beam_obj.utterance
            # the network gives back logits instead of probabilities
            # so start by calculating probabilities
            preds = tf.nn.log_softmax(predictions, axis=1).numpy()
            # find the candidates for this prediction
            candidates = search_candidates(curr_prob, curr_utterance, dec_hidden, preds, training_info['ref_idx2word'], beam_size)
            new_beams += candidates
        normalised_beams = [BeamObj(b.utterance, b.probability/get_length_normalisation_denominator(b.utterance), b.last_id, b.last_hidden) for b in new_beams]
        beam = sorted(normalised_beams, key=lambda b: -b.probability)[:beam_size]
        all_generated = [generation_done(beam_obj, training_info, end_token) for beam_obj in beam]
        if np.all(all_generated):
            break
    return beam, processed_mr_info, attention_plot

def create_results(predicted_ids, results, training_info):
    """ Create utterances from predicted ids """
    for i in range(len(predicted_ids)):
        idd = predicted_ids[i]
        utt = results[i]
        # don't add anything if the utterance already has an end symbol
        if not utt.endswith(END_SYMBOL) and idd != 0:
            results[i] = (utt +  " " + training_info['ref_idx2word'][idd]).strip()
    return results

def evaluate_with_sampling(encoder, decoder, mr_info, training_info, batch_size):
    attention_plot = np.zeros((training_info['max_length_targ'], training_info['max_length_inp']))
    processed_mr_info = preprocess_mr(mr_info)
    inputs = [training_info['mr_word2idx'][i.strip()] if i.strip() in training_info['mr_word2idx'] else 0 for i in processed_mr_info.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=training_info['max_length_inp'],
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)
    inputs = tf.tile(inputs, tf.constant([batch_size, 1]))
    hidden = encoder.initialize_hidden_state(batch_size)
    enc_out, forward_hidden, backward_hidden = encoder(inputs, hidden)
    dec_hidden = tf.keras.layers.Concatenate()([forward_hidden, backward_hidden])
    dec_input = tf.expand_dims([training_info['ref_word2idx'][START_SYMBOL]], 0)
    dec_input = tf.tile(dec_input, tf.constant([batch_size, 1]))
    results = ['']*batch_size
    for t in range(training_info['max_length_targ']):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        pred_dist = tfp.distributions.Multinomial(total_count=1, logits=predictions)
        predicted_ids = tf.argmax(pred_dist.sample(1)[0], axis=1).numpy()
        results = create_results(predicted_ids, results, training_info)
        dec_input = tf.expand_dims(predicted_ids, 1)
    return results, mr_info, attention_plot

def generate_reference_using_beam(encoder, decoder, mr_info, training_info, beam_size=1):
    """ Generate new reference, and postprocess it to form a complete sentence using beam search."""
    beam, processed_mr_info, attention_plot = evaluate_with_beam(encoder, decoder, mr_info, training_info, beam_size)
    mr_slots = get_slots(mr_info, remove_whitespace=False)
    # postprocess and score the beam
    for beam_obj in beam:
        processed_utterance = postprocess_utterance(beam_obj.utterance, mr_slots)
        score = score_prediction(processed_utterance, mr_slots)
        beam_obj.utterance = processed_utterance
        beam_obj.probability += np.log(score)
    # order again by probability
    sorted_beam = sorted(beam, key=lambda x: -x.probability)
    return sorted_beam[0].utterance

def generate_reference_with_sampling(encoder, decoder, mr_info, training_info):
    """ Generate new reference, and postprocess it to form a complete sentence by sampling the next token from a probability distribution."""
    results, processed_mr_info, attention_plot = evaluate_with_sampling(encoder, decoder, mr_info, training_info, batch_size=10)
    mr_slots = get_slots(mr_info, remove_whitespace=False)
    scores = np.zeros(len(results))
    utterances = []
    for i, ref in enumerate(results):
        processed_utterance = postprocess_utterance(ref, mr_slots)
        score = score_prediction(processed_utterance, mr_slots)
        scores[i] = score
        utterances.append(processed_utterance)
    # postprocess and score the beam
    best_pred_id = np.argsort(-scores)[0]
    return utterances[best_pred_id]

def sample_mr_content(mr_info, content_selection_model_file):
    mr_slots = get_slots(mr_info, remove_whitespace=False)
    # don't sample over name
    sample_mrs = [k for k in mr_slots.keys() if k != 'name']
    sampled_slots = bayesian_sampler.sample_slots(content_selection_model_file, sample_mrs)
    # always include name
    sampled_slots += ['name']
    mr_slots_to_use = { mr_key: mr_slots[mr_key] for mr_key in mr_slots.keys() if mr_key in sampled_slots }
    return ', '.join(k + '[' + v + ']' for k, v in mr_slots_to_use.items())

def print_generations(test_data, encoder, decoder, training_info, beam_width, sample_content, cpd_model_file):
    print('Beam width is', beam_width)
    for i in range(len(test_data)):
        print(test_data['mr'].iloc[i])
        mr_input = test_data['mr'].iloc[i]
        mr_info = ''
        if sample_content:
            mr_info = sample_mr_content(mr_input, cpd_model_file)
            print('Sampled mr', mr_info)
        else:
            mr_info = mr_input
        generated = ''
        if beam_width > 0:
            generated = generate_reference_using_beam(encoder, decoder, mr_info, training_info, beam_width)
        else:
            generated = generate_reference_with_sampling(encoder, decoder, mr_info, training_info)
        print(generated)
        if 'ref' in test_data.columns:
            print(test_data['ref'].iloc[i])
            bleu = nltk.translate.bleu_score.sentence_bleu([test_data['ref'].iloc[i]], generated)
            print('bleu score for the best prediction', bleu)
        print('-------------------------')

def calculate_mean_bleu_score(test_data, encoder, decoder, training_info, beam_width, sample_content, cpd_model_file=None):
    print('Calculating mean BLEU score for validation set of size', len(test_data))
    bleus = np.zeros(len(test_data))
    if sample_content and cpd_model_file is None:
        raise ValueError('Please give CPD model file if sampling content')
    for i in range(len(test_data)):
        mr_input = test_data['mr'].iloc[i]
        mr_info = ''
        if sample_content:
            mr_info = sample_mr_content(mr_input, cpd_model_file)
        else:
            mr_info = mr_input
        generated = ''
        if beam_width > 0:
            generated = generate_reference_using_beam(encoder, decoder, mr_info, training_info, beam_width)
        else:
            generated = generate_reference_with_sampling(encoder, decoder, mr_info, training_info)
        bleu = nltk.translate.bleu_score.sentence_bleu([test_data['ref'].iloc[i]], generated)
        bleus[i] = bleu
        if i % 50 == 0:
            print(i)
        if i % 500 == 0:
            print(generated)
            print(test_data['ref'].iloc[i])
            print('mean bleu', np.mean(bleus[bleus > 0]))
    return np.mean(bleus), np.var(bleus)

def main(test_data_file, checkpoint_dir, training_info_file, beam_width, sample_content, cpd_model_file, print_utt):
    training_info = helpers.load_from_pickle(training_info_file)
    encoder = Encoder(len(training_info['mr_word2idx'])+1, 
                        training_info['embedding_dim'], 
                        training_info['units'])
    decoder = Decoder(len(training_info['ref_word2idx'])+1, 
                        training_info['embedding_dim'], 
                        training_info['units']*2, 
                        training=False)
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)
    print('Restoring checkpoint from', checkpoint_dir)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # get test data
    test_data = load_text_data(test_data_file, 2000)
    if print_utt:
        print_generations(test_data, encoder, decoder, training_info, beam_width, sample_content, cpd_model_file)
    bleu_mean, bleu_var = calculate_mean_bleu_score(test_data, encoder, decoder, training_info, beam_width, sample_content, cpd_model_file)
    print(bleu_mean, bleu_var)

if __name__ == "__main__":
    # restoring the latest checkpoint in checkpoint_dir
    args = parser.parse_args()
    test_data_file = args.test_data
    identifier = args.identifier
    checkpoint_dir = 'training_checkpoints' + identifier
    training_info_file = 'training_info' + identifier + '.pkl'
    beam_width = args.beam_width
    sample_content = args.sample_content
    print_utt = args.print_utt
    print('Sampling content', sample_content)
    cpd_model_file = args.cpd_model_file
    main(test_data_file, checkpoint_dir, training_info_file, beam_width, sample_content, cpd_model_file, print_utt)