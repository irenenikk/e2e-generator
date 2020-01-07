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

BATCH_SIZE = 1
# optimal beam size found by Juraska
BEAM_SIZE = 10
END_SYMBOL = '<end>'
START_SYMBOL = '<start>'

parser = argparse.ArgumentParser(description='Generate utterances from a trained E2E description generator')
parser.add_argument("test_data", type=str,
                    help="The path to the test data file")
parser.add_argument("-id", "--identifier", default='',
                    help="The identifier used to define checkpoint and training info directories")


class BeamObj:
    def __init__(self, utterance, probability, last_id):
        self.utterance = utterance
        self.probability = probability
        # save the id of the last word
        # this will be used as input for the next timestep
        self.last_id = last_id

    def __repr__(self):
        return '{}({})'.format(self.utterance, self.probability)

def search_candidates2(curr_prob, curr_utterance, predictions, ref_idx2word):
    """ Find the possible extensions to an utterance and their log-probabilities. """ 
    candidates = []
    # calculate probabilites if new predictions were added
    # check for all new predictions in the beam 
    if predictions.shape[0] != 1:
        raise ValueError('Batches not supported in beam search. ')
    for word_id in range(1, predictions.shape[1]):
        pred = predictions[0][word_id]
        new_prob = curr_prob + np.log(pred)
        new_utterance = curr_utterance + " " + ref_idx2word[word_id]
        candidates += [BeamObj(new_utterance, new_prob, word_id)]
    return candidates

def search_candidates(curr_prob, curr_utterance, predictions, ref_idx2word, beam_size):
    """ Find the possible extensions to an utterance and their log-probabilities. """ 
    candidates = []
    # calculate probabilites if new predictions were added
    # check for all new predictions in the beam 
    if predictions.shape[0] != 1:
        raise ValueError('Batches not supported in beam search. ')
    preds = predictions[0]
    ids_by_prob = np.argsort(-preds)[:5]
    for word_id in ids_by_prob:
        pred = preds[word_id]
        new_prob = curr_prob + pred
        new_utterance = curr_utterance + " " + ref_idx2word[word_id]
        candidates += [BeamObj(new_utterance, new_prob, word_id)]
    return candidates

def search_candidates3(curr_prob, curr_utterance, predictions, ref_idx2word, beam_size):
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
        new_utterance = curr_utterance + " " + ref_idx2word[word_id]
        candidates += [BeamObj(new_utterance, new_prob, word_id)]
    return candidates

def score_prediction(prediction, mr_slots):
    """ 
    Scores a complete utterance based on slot realisation.
    mr_slots should be a dict where keys are slots and values are slot values.
    The score function is taken from Juraska et al.
    """
    N = len(mr_slots.keys())
    #print('-------------')
    #print(mr_slots)
    # use Juraska's code to get erronous slots
    unaligned_slots, hallucinated_slots = get_unaligned_and_hallucinated_slots(prediction, mr_slots)
    #print('Prediction', prediction)
    #print('Unaligned slots:', unaligned_slots)
    #print('Hallucinated slots:', hallucinated_slots)
    return N/((len(unaligned_slots)+1)*(len(hallucinated_slots)+1))

def generation_done(beam_obj, training_info, end_token):
    """ Stop when end token is reached or the sentence is the maximal length. """
    return beam_obj.last_id == end_token or len(beam_obj.utterance) == training_info['max_length_targ']

def get_length_normalisation_denominator(utterance, alpha=20):
    """ As done in Google's NMT paper. """
    n = ((5 + len(utterance.split(" ")))**alpha)/((5+1)**alpha)
    return n

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
    beam = [BeamObj('', 0, start_token)]
    hidden = encoder.initialize_hidden_state(1)
    enc_out, forward_hidden, backward_hidden = encoder(inputs, hidden)
    dec_hidden = tf.keras.layers.Concatenate()([forward_hidden, backward_hidden])
    while(True):
        new_beams = []
        for beam_obj in beam:
            #print('---------')
            #print('beam obj', beam_obj)
            if generation_done(beam_obj, training_info, end_token):
                new_beams += [beam_obj]
            else:
                predictions, dec_hidden, attention_weights = decoder(tf.expand_dims([beam_obj.last_id], 0), dec_hidden, enc_out)
                curr_prob = beam_obj.probability
                curr_utterance = beam_obj.utterance
                # the network gives back logits instead of probabilities
                # so start by calculating probabilities
                preds = tf.nn.log_softmax(predictions, axis=1).numpy()
                # find the candidates for this prediction
                candidates = search_candidates(curr_prob, curr_utterance, preds, training_info['ref_idx2word'], beam_size)
                #print('candidates', candidates)
                new_beams += candidates
            #print('new beams', new_beams)
        # normalize the probability in order not to favor short sequences.
        #print('VANILLA')
        #print([BeamObj(b.utterance, b.probability, b.last_id) for b in new_beams])
        normalised_beams = [BeamObj(b.utterance, b.probability/get_length_normalisation_denominator(b.utterance), b.last_id) for b in new_beams]
        #print('NORMALISED')
        #print(normalised_beams)
        beam = sorted(normalised_beams, key=lambda b: -b.probability)[:beam_size]
        #print('beam', beam)
        #same = len(set([len(b.utterance.split(" "))for b in new_beams])) == 1
        all_generated = [generation_done(beam_obj, training_info, end_token) for beam_obj in beam]
        if np.all(all_generated):
            break
    return beam, processed_mr_info, attention_plot

def create_results(predicted_ids, results, training_info):
    for i in range(len(predicted_ids)):
        idd = predicted_ids[i]
        utt = results[i]
        if not utt.endswith(END_SYMBOL) and idd != 0:
            results[i] = (utt +  " " + training_info['ref_idx2word'][idd]).strip()
    return results

def evaluate_with_sampling(encoder, decoder, mr_info, training_info, beam_size):
    attention_plot = np.zeros((training_info['max_length_targ'], training_info['max_length_inp']))
    processed_mr_info = preprocess_mr(mr_info)
    inputs = [training_info['mr_word2idx'][i.strip()] if i.strip() in training_info['mr_word2idx'] else 0 for i in processed_mr_info.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=training_info['max_length_inp'],
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)
    inputs = tf.tile(inputs, tf.constant([beam_size, 1]))
    hidden = encoder.initialize_hidden_state(beam_size)
    enc_out, forward_hidden, backward_hidden = encoder(inputs, hidden)
    dec_hidden = tf.keras.layers.Concatenate()([forward_hidden, backward_hidden])
    dec_input = tf.expand_dims([training_info['ref_word2idx'][START_SYMBOL]], 0)
    dec_input = tf.tile(dec_input, tf.constant([beam_size, 1]))
    results = ['']*beam_size
    for t in range(training_info['max_length_targ']):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)
        pred_dist = tfp.distributions.Multinomial(total_count=1, logits=predictions)
        predicted_ids = tf.argmax(pred_dist.sample(1)[0], axis=1).numpy()
        results = create_results(predicted_ids, results, training_info)
        dec_input = tf.expand_dims(predicted_ids, 1)
    return results, mr_info, attention_plot

def generate_reference_using_beam(encoder, decoder, mr_info, training_info, beam_size=10):
    """ Generate new reference, and postprocess it to form a complete sentence."""
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
    #best_prediction = beam[0].processed_utterance
    return beam[0].utterance, beam[0].probability

def generate_reference_with_sampling(encoder, decoder, mr_info, training_info):
    """ Generate new reference, and postprocess it to form a complete sentence."""
    results, processed_mr_info, attention_plot = evaluate_with_sampling(encoder, decoder, mr_info, training_info, 5)
    mr_slots = get_slots(mr_info, remove_whitespace=False)
    scores = np.zeros(len(results))
    utterances = []
    for i, ref in enumerate(results):
        processed_utterance = postprocess_utterance(ref, mr_slots)
        score = score_prediction(processed_utterance, mr_slots)
        scores[i] = score
        utterances.append(processed_utterance)
    #for s, r in zip(scores, utterances):
    #    print(r, '(', s, ')')
    #print('----')
    # postprocess and score the beam
    best_pred_id = np.argsort(-scores)[0]
    return utterances[best_pred_id]

def print_generations(test_data, encoder, decoder, training_info):
    for i in range(len(test_data)):
        print(test_data['mr'].iloc[i])
        generated = generate_reference_with_sampling(encoder, decoder, test_data['mr'].iloc[i], training_info)
        #print('with beam 1', generated)
        #generated = generate_reference(encoder, decoder, test_data['mr'].iloc[i], training_info, 3)
        print(generated)
        if 'ref' in test_data.columns:
            print(test_data['ref'].iloc[i])
            bleu = nltk.translate.bleu_score.sentence_bleu([test_data['ref'].iloc[i]], generated)
            print(bleu)
        print('------------')

def calculate_mean_bleu_score(test_data, encoder, decoder, training_info, sampling=True):
    print('Calculating mean BLEU score for validation set of size', len(test_data))
    bleus = np.zeros(len(test_data))
    for i in range(len(test_data)):
        generated = None
        if sampling:
            generated = generate_reference_with_sampling(encoder, decoder, test_data['mr'].iloc[i], training_info)
        else:
            generated = generate_reference(encoder, decoder, test_data['mr'].iloc[i], training_info)
        bleu = nltk.translate.bleu_score.sentence_bleu([test_data['ref'].iloc[i]], generated)
        bleus[i] = bleu
    return np.mean(bleus), np.var(bleus)

def main(test_data_file, checkpoint_dir, training_info_file):
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
    test_data = load_text_data(test_data_file)
    print_generations(test_data, encoder, decoder, training_info)
    bleu_mean, bleu_var = calculate_mean_bleu_score(test_data, encoder, decoder, training_info)
    print(bleu_mean, bleu_var)

if __name__ == "__main__":
    # restoring the latest checkpoint in checkpoint_dir
    args = parser.parse_args()
    test_data_file = args.test_data
    identifier = args.identifier
    checkpoint_dir = 'training_checkpoints' + identifier
    training_info_file = 'training_info' + identifier + '.pkl'
    main(test_data_file, checkpoint_dir, training_info_file)