import pandas as pd
import sys
import numpy as np
from data_loader import create_dataset, load_data_tensors, max_length
import tensorflow as tf
from models import Encoder, BahdanauAttention, Decoder
import time
import os
import pickle
import json
import ipdb

BATCH_SIZE = 64
embedding_dim = 128
units = 512
TRAINING_INFO_FILE = 'training_info.pkl'
DECODER_NUM_LAYERS = 4

def save_training_info(ref_word2idx, ref_idx2word, mr_word2idx, mr_idx2word, max_length_targ, max_length_inp, embedding_dim, units, decoder_layers):
    training_info = {}
    training_info['ref_word2idx'] = ref_word2idx
    training_info['ref_idx2word'] = ref_idx2word
    training_info['mr_word2idx'] = mr_word2idx
    training_info['mr_idx2word'] = mr_idx2word
    training_info['max_length_targ'] = max_length_targ
    training_info['max_length_inp'] = max_length_inp
    training_info['embedding_dim'] = embedding_dim
    training_info['decoder_layers'] = decoder_layers
    training_info['units'] = units
    with open(TRAINING_INFO_FILE, 'wb') as f:
        pickle.dump(training_info, f)

def loss_function(real, pred):
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')      
  loss_ = loss_object(real, pred)
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden, ref_word2idx):
  loss = 0
  with tf.GradientTape() as tape:
    enc_output, forward_hidden, forward_mem, backward_hidden, backward_mem = encoder(inp, enc_hidden)
    # initialize using the concatenated forward and backward states
    state_h = tf.keras.layers.Concatenate()([forward_hidden, backward_hidden])
    state_c = tf.keras.layers.Concatenate()([forward_mem, backward_mem])
    dec_hidden = [state_h, state_c]
    dec_input = tf.expand_dims([ref_word2idx['sssss']] * BATCH_SIZE, 1)
    for t in range(1, targ.shape[1]):
      predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_output)
      # use log cross-entropy loss
      loss += loss_function(targ[:, t], predictions)
      dec_input = tf.expand_dims(tf.argmax(predictions, axis=1), 1)
  batch_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return batch_loss

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please give path to data file as argument')
        exit()
    data_file = sys.argv[1]
    print('Loading data')
    input_tensor, target_tensor, ref_word2idx, ref_idx2word, mr_word2idx, mr_idx2word = load_data_tensors(data_file, 1000)
    print('Creating dataset')
    train_dataset, val_dataset, steps_per_epoch = create_dataset(input_tensor, 
                                                                target_tensor, 
                                                                batch_size=BATCH_SIZE, 
                                                                embedding_dim=embedding_dim, 
                                                                units=units, 
                                                                test_size=0.2)
    print('Saving training information')
    max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
    # save training info to use in analysis
    save_training_info(ref_word2idx, 
                        ref_idx2word, 
                        mr_word2idx, 
                        mr_idx2word, 
                        max_length_targ, 
                        max_length_inp,
                        embedding_dim,
                        units,
                        DECODER_NUM_LAYERS
                        )
    encoder = Encoder(len(mr_word2idx)+1, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(len(ref_word2idx)+1, DECODER_NUM_LAYERS, embedding_dim, units*2, BATCH_SIZE)
    optimizer = tf.keras.optimizers.Adam()
    # prepare to train
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)
    print('Starting training')
    #train
    EPOCHS = 15
    s = time.time()
    for epoch in range(EPOCHS):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden, ref_word2idx)
            total_loss += batch_loss
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    print('Training took', (time.time() - s)/60, 'minutes')
