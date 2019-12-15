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

BATCH_SIZE = 64

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
    result = ''
    hidden = [tf.zeros((1, training_info['units']))]*4
    enc_out, forward_hidden, forward_mem, backward_hidden, backward_mem = encoder(inputs, hidden)
    state_h = tf.keras.layers.Concatenate()([forward_hidden, backward_hidden])
    state_c = tf.keras.layers.Concatenate()([forward_mem, backward_mem])
    dec_hidden = [state_h, state_c]
    dec_input = tf.expand_dims([training_info['ref_word2idx']['sssss']], 0)
    for t in range(training_info['max_length_targ']):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)
        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += training_info['ref_idx2word'][predicted_id] + ' '
        if training_info['ref_idx2word'][predicted_id] == '>':
            return result, mr_info, attention_plot
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)
    return result, mr_info, attention_plot

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
