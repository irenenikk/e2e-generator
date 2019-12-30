import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_mr, get_slots
from postprocessing import postprocess_utterance
import sys
from data_loader import load_data_tensors, load_text_data
from models import Encoder, BahdanauAttention, Decoder
import time
import os
import pickle
import nltk
#from slug2slug.slot_aligner.slot_alignment import get_unaligned_and_hallucinated_slots
from slug2slug_aligner import get_unaligned_and_hallucinated_slots
sys.path.append('./')

BATCH_SIZE = 64
# optimal beam size found by Juraska
BEAM_SIZE = 10

class BeamObj:
    def __init__(self, utterance, probability, last_id):
        self.utterance = utterance
        self.probability = probability
        # save the id of the last word
        # this will be used as input for the next timestep
        self.last_id = last_id

    def __repr__(self):
        return '{}({})'.format(self.utterance, self.probability)


def beam_search(previous_beam, new_predictions, ref_idx2word):
    """ Calculate the log probability of each sequence and return in descending order. """ 
    new_beams = []
    for i in range(len(new_predictions)):
        pred = new_predictions.numpy()[i]
        #print(pred)
        #print('pred length', len(pred))
        old_beam = previous_beam[i]
        #print('old beam', old_beam)
        # sort in ascending order
        ids_by_prob = np.argsort(-pred)[:BEAM_SIZE]
        #print('most probable ids', ids_by_prob)
        words_by_prob = [ref_idx2word[idd] for idd in ids_by_prob if idd != 0]
        #print('most probable words', words_by_prob)
        probs = pred[ids_by_prob]
        #print(probs)
        #print('ordered probs', probs)
        new_beams += [BeamObj(old_beam.utterance + " " + ref_idx2word[idd], old_beam.probability + np.log(prob), idd) for idd, prob in zip(ids_by_prob, probs) if idd > 0]
    #print('new_beams', new_beams)
    new_beams.sort(key=lambda x: x.probability, reverse=True)
    #print('new_beams sorted', new_beams)
    return new_beams[:BEAM_SIZE]

def score_prediction(prediction, mr_slots):
    """ 
    Scores a complete utterance based on slot realisation.
    mr_slots should be a dict where keys are slots and values are slot values.
    The score function is taken from Juraska et al.
    """
    N = len(mr_slots.keys())
    # use Juraska's code to get erronous slots
    unaligned_slots, hallucinated_slots = get_unaligned_and_hallucinated_slots(prediction, mr_slots)
    #print('Prediction', prediction)
    #print('Unaligned slots:', unaligned_slots)
    #print('Hallucinated slots:', hallucinated_slots)
    return N/((len(unaligned_slots)+1)*(len(hallucinated_slots)+1))

def evaluate(encoder, decoder, mr_info, training_info):
    attention_plot = np.zeros((training_info['max_length_targ'], training_info['max_length_inp']))
    processed_mr_info = preprocess_mr(mr_info)
#    print(mr_info)
#    print(training_info['mr_word2idx'].keys())
    inputs = [training_info['mr_word2idx'][i.strip()] for i in processed_mr_info.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=training_info['max_length_inp'],
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)
    beam = [BeamObj('', 0, -1)]*BEAM_SIZE
    hidden = [tf.zeros((1, training_info['units']))]*2
    enc_out, forward_hidden, backward_hidden = encoder(inputs, hidden)
    dec_hidden = tf.keras.layers.Concatenate()([forward_hidden, backward_hidden])
    dec_input = tf.expand_dims([training_info['ref_word2idx']['<start>']], 0)
    # TODO: stop only at a stop word
    for t in range(training_info['max_length_targ']):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)
        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()
        # use beam search to keep n best predictions
        beam = beam_search(beam, predictions, training_info['ref_idx2word'])
        next_inputs = [[b.last_id] for b in beam if training_info['ref_idx2word'][b.last_id] != '<end>']
        #for n in next_inputs:
        #    print(training_info['ref_idx2word'][n[0]])
        #print('----')
        if next_inputs == []:
            # TODO: rerank final beams
            return beam, processed_mr_info, attention_plot
        # the predicted ID is fed back into the model
        dec_input = np.asarray(next_inputs)
    return beam, processed_mr_info, attention_plot

# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    fontdict = {'fontsize': 14}
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
    plt.show()

def generate_reference(encoder, decoder, mr_info, training_info):
    """ Generate new reference, and postprocess it to form a complete sentence."""
    beams, processed_mr_info, attention_plot = evaluate(encoder, decoder, mr_info, training_info)
    mr_slots = get_slots(mr_info)
    # postprocess and score the beam
    for beamObj in beams:    
        processed_utterance = postprocess_utterance(beamObj.utterance, mr_slots)
        beamObj.processed_utterance = processed_utterance
        score = score_prediction(processed_utterance, mr_slots)
        beamObj.probability += np.log(score)
    # order again by probability
    beams.sort(key=lambda x: x.probability, reverse=True)
    best_prediction = beams[0].processed_utterance
    #attention_plot = attention_plot[:len(best_prediction.split(' ')), :len(mr_info.split(' '))]
    #plot_attention(attention_plot, mr_info.split(' '), best_prediction.split(' '))
    return best_prediction

def load_training_info(training_info_file):
    with open(training_info_file, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # restoring the latest checkpoint in checkpoint_dir
    if len(sys.argv) < 3:
        print('Please give path to data file as argument')
        exit()
    checkpoint_dir = sys.argv[1]
    test_data_file = sys.argv[2]
    training_info_file = 'training_info.pkl' if len(sys.argv) < 4 else sys.argv[3]
    training_info = load_training_info(training_info_file)
    encoder = Encoder(len(training_info['mr_word2idx'])+1, 
                        training_info['embedding_dim'], 
                        training_info['units'], 
                        BATCH_SIZE)
    decoder = Decoder(len(training_info['ref_word2idx'])+1, 
                        training_info['decoder_layers'], 
                        training_info['embedding_dim'], 
                        training_info['units']*2, 
                        BATCH_SIZE, 
                        training=False)
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # get test data
    test_data = load_text_data(test_data_file)
    for i in range(len(test_data)):
        generated = generate_reference(encoder, decoder, test_data['mr'].iloc[i], training_info)
        print(test_data['mr'].iloc[i])
        print(generated)
        bleu = nltk.translate.bleu_score.sentence_bleu([test_data['mr'].iloc[i]], generated)
        print(bleu)
        print('------------')
