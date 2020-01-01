import tensorflow as tf
import tensorflow_probability as tfp
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
import ipdb; 
#from slug2slug.slot_aligner.slot_alignment import get_unaligned_and_hallucinated_slots
from slug2slug_aligner import get_unaligned_and_hallucinated_slots
sys.path.append('./')

BATCH_SIZE = 1
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
    # for each beamobj in batch
    # batch size is the length of the beam
    for prev_beam_obj in previous_beam:
        # calculate probabilites if new predictions were added
        curr_prob = prev_beam_obj.probability
        curr_utterance = prev_beam_obj.utterance
        # check for all new predictions in the beam
        for beam_id in range(new_predictions.shape[0]):
            for word_id in range(1, new_predictions.shape[1]):
                pred = new_predictions[beam_id,word_id]
                new_prob = curr_prob + np.log(pred)
                new_utterance = curr_utterance + " " + ref_idx2word[word_id]
                new_beams += [(BeamObj(new_utterance, new_prob, word_id))]
        # add the results to the previous beam
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
    #beam = [BeamObj('', 0, -1)]
    hidden = encoder.initialize_hidden_state(1)
    enc_out, forward_hidden, backward_hidden = encoder(inputs, hidden)
    dec_hidden = tf.keras.layers.Concatenate()([forward_hidden, backward_hidden])
    dec_input = tf.expand_dims([training_info['ref_word2idx']['<start>']], 0)
    result = ''
    for t in range(training_info['max_length_targ']):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)
        preds = predictions.numpy()[0]
        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()
        # use beam search to keep n best predictions
        #beam = beam_search(beam, predictions, training_info['ref_idx2word'])
        predicted_id = tf.argmax(predictions[0]).numpy()
        #pred_dist = tfp.distributions.Multinomial(total_count=1, logits=predictions[0])
        #predicted_id = tf.argmax(pred_dist.sample(1), axis=1).numpy()[0]
        result += training_info['ref_idx2word'][predicted_id] + ' '
        #next_inputs = [[b.last_id] for b in beam if training_info['ref_idx2word'][b.last_id] != '<end>']
        #for n in next_inputs:
        #    print(training_info['ref_idx2word'][n[0]])
        #print('----')
        if training_info['ref_idx2word'][predicted_id] == '<end>':
            return result, mr_info, attention_plot
        #if next_inputs == []:
        #    return beam, processed_mr_info, attention_plot
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)
    return result, mr_info, attention_plot
    '''
        dec_input = np.asarray(next_inputs)
    return beam, processed_mr_info, attention_plot
    '''

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
    beam, processed_mr_info, attention_plot = evaluate(encoder, decoder, mr_info, training_info)
    mr_slots = get_slots(mr_info)
    # postprocess and score the beam
    for beamObj in beam:    
        processed_utterance = postprocess_utterance(beamObj.utterance, mr_slots)
        beamObj.processed_utterance = processed_utterance
        score = score_prediction(processed_utterance, mr_slots)
        beamObj.probability += np.log(score)
    # order again by probability
    beam.sort(key=lambda x: x.probability, reverse=True)
    best_prediction = beam[0].utterance
    #attention_plot = attention_plot[:len(best_prediction.split(' ')), :len(mr_info.split(' '))]
    #plot_attention(attention_plot, mr_info.split(' '), best_prediction.split(' '))
    return best_prediction

def generate_reference_no_beam(encoder, decoder, mr_info, training_info):
    """ Generate new reference, and postprocess it to form a complete sentence."""
    res, processed_mr_info, attention_plot = evaluate(encoder, decoder, mr_info, training_info)
    return res

def load_training_info(training_info_file):
    with open(training_info_file, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # restoring the latest checkpoint in checkpoint_dir
    if len(sys.argv) < 2:
        print('Please give path to data file as argument')
        exit()
    test_data_file = sys.argv[1]
    checkpoint_dir = './training_checkpoints' if len(sys.argv) < 3 else sys.argv[2]
    training_info_file = 'training_info.pkl' if len(sys.argv) < 4 else sys.argv[3]
    training_info = load_training_info(training_info_file)
    encoder = Encoder(len(training_info['mr_word2idx'])+1, 
                        training_info['embedding_dim'], 
                        training_info['units'])
    decoder = Decoder(len(training_info['ref_word2idx'])+1, 
                        training_info['decoder_layers'], 
                        training_info['embedding_dim'], 
                        training_info['units']*2, 
                        training=False)
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # get test data
    test_data = load_text_data(test_data_file)
    for i in range(len(test_data)):
        generated = generate_reference_no_beam(encoder, decoder, test_data['mr'].iloc[i], training_info)
        print(test_data['mr'].iloc[i])
        print(generated)
        bleu = nltk.translate.bleu_score.sentence_bleu([test_data['mr'].iloc[i]], generated)
        print(bleu)
        print('------------')
