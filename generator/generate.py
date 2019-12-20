import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_mr
import sys
from data_loader import load_data_tensors, load_text_data
from models import Encoder, BahdanauAttention, Decoder
import time
import os
import pickle
import ipdb
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
        return f'{self.utterance} ({self.probability})'


def beam_search(previous_beam, new_predictions, ref_idx2word):
    """ Calculate the log probability of each sequence and return in descending order. """ 
    new_beams = []
    for i in range(len(new_predictions)):
        pred = new_predictions.numpy()[i]
        #print('pred length', len(pred))
        old_beam = previous_beam[i]
        #print('old beam', old_beam)
        # sort in ascending order
        ids_by_prob = np.argsort(-pred)[:BEAM_SIZE]
        #print('most probable ids', ids_by_prob)
        words_by_prob = [ref_idx2word[idd] for idd in ids_by_prob]
        #print('most probable words', words_by_prob)
        probs = pred[ids_by_prob]
        #print('ordered probs', probs)
        new_beams += [BeamObj(old_beam.utterance + " " + ref_idx2word[idd], old_beam.probability + np.log(prob), idd) for idd, prob in zip(ids_by_prob, probs) if idd > 0]
    #print('new_beams', new_beams)
    new_beams.sort(key=lambda x: x.probability, reverse=True)
    #print('new_beams sorted', new_beams)
    return new_beams[:BEAM_SIZE]

def score_predictions(predictions):
    """ Scores a complete utterance based on slot realisation. """
    pass

def evaluate(encoder, decoder, mr_info, training_info):
    attention_plot = np.zeros((training_info['max_length_targ'], training_info['max_length_inp']))
    mr_info = preprocess_mr(mr_info)
#    print(mr_info)
#    print(training_info['mr_word2idx'].keys())
    inputs = [training_info['mr_word2idx'][i.strip()] for i in mr_info.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=training_info['max_length_inp'],
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)
    beam = [BeamObj('', 0, -1)]*BEAM_SIZE
    hidden = [tf.zeros((1, training_info['units']))]*4
    enc_out, forward_hidden, forward_mem, backward_hidden, backward_mem = encoder(inputs, hidden)
    state_h = tf.keras.layers.Concatenate()([forward_hidden, backward_hidden])
    state_c = tf.keras.layers.Concatenate()([forward_mem, backward_mem])
    dec_hidden = [state_h, state_c]
    dec_input = tf.expand_dims([training_info['ref_word2idx']['sssss']], 0)
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
        #print('beam', beam)
        for b in beam:
            if training_info['ref_idx2word'][b.last_id] == 'eeeee':
                print('FOUND END MARK')
        next_inputs = [[b.last_id] for b in beam if training_info['ref_idx2word'][b.last_id] != 'eeeee']
        if next_inputs == []:
            return beam[0].utterance, mr_info, attention_plot
        # the predicted ID is fed back into the model
        dec_input = np.asarray(next_inputs)
        #dec_input = next_inputs
    return beam[0].utterance, mr_info, attention_plot

# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    fontdict = {'fontsize': 14}
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
    plt.show()

def generate_reference(encoder, decoder, sentence, training_info):
    result, mr_info, attention_plot = evaluate(encoder, decoder, sentence, training_info)
    #attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    #plot_attention(attention_plot, sentence.split(' '), result.split(' '))
    return result

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
    encoder = Encoder(len(training_info['mr_word2idx'])+1, training_info['embedding_dim'], training_info['units'], BATCH_SIZE)
    decoder = Decoder(len(training_info['ref_word2idx'])+1, 4, training_info['embedding_dim'], training_info['units']*2, BATCH_SIZE)
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
        print('------------')
