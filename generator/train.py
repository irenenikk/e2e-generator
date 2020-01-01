import pandas as pd
import sys
import numpy as np
from data_loader import create_dataset, load_data_tensors, max_length
import tensorflow as tf
import tensorflow_probability as tfp
from models import Encoder, BahdanauAttention, Decoder
import time
import os
import pickle
import json
import nltk
from generate import evaluate, load_training_info

EPOCHS = 5
BATCH_SIZE = 32
embedding_dim = 128
units = 512
TRAINING_INFO_FILE = 'training_info.pkl'
DECODER_NUM_LAYERS = 1
TEACHER_FORCING = False

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
  # ignore padded parts of the input
  # this can be done because 0 is a special index
  loss_ = loss_object(real, pred)
  pad_mask = tf.math.logical_not(tf.math.equal(real, 0))
  pad_mask = tf.cast(pad_mask, dtype=loss_.dtype)
  loss_ *= pad_mask
  return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden, ref_word2idx, ref_idx2word, teacher_force_prob):
  loss = 0
  print('teacher_force_prob', teacher_force_prob)
  with tf.GradientTape() as tape:
    enc_output, forward_hidden, backward_hidden = encoder(inp, enc_hidden)
    # initialize using the concatenated forward and backward states
    dec_hidden = tf.keras.layers.Concatenate()([forward_hidden, backward_hidden])
    dec_input = tf.expand_dims([ref_word2idx['<start>']] * BATCH_SIZE, 1)
    all_preds = None
    all_targets = None
    all_inputs = None
    for t in range(1, targ.shape[1]):
      predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_output)
      # use log cross-entropy loss
      loss += loss_function(targ[:, t], predictions)
      predicted_tokens = tf.argmax(predictions, axis=1)
      if np.random.uniform() < teacher_force_prob:
        dec_input = tf.expand_dims(targ[:, t], 1)
      else:
        dec_input = tf.expand_dims(predicted_tokens, 1)
      if all_preds is None:
            all_preds = tf.expand_dims(predicted_tokens, 1)
            all_targets = tf.expand_dims(targ[:, t], 1)
            all_inputs = dec_input
      else:
          all_preds = tf.concat([all_preds, tf.expand_dims(predicted_tokens, 1)], axis=1)
          all_targets = tf.concat([all_targets, tf.expand_dims(targ[:, t], 1)], axis=1)
          all_inputs = tf.concat([all_inputs, tf.cast(dec_input, all_inputs.dtype)], axis=1)
  batch_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return batch_loss, all_preds, all_targets, all_inputs, gradients

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please give path to data file as argument')
        exit()
    data_file = sys.argv[1]
    checkpoint_dir = './training_checkpoints' if len(sys.argv) < 3 else sys.argv[2]
    print('Loading data')
    input_tensor, target_tensor, ref_word2idx, ref_idx2word, mr_word2idx, mr_idx2word = load_data_tensors(data_file)
    print('Found input data of shape', input_tensor.shape)
    print('Found target data of shape', target_tensor.shape)
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
                        DECODER_NUM_LAYERS)
    training_info = load_training_info(TRAINING_INFO_FILE)
    encoder = Encoder(len(mr_word2idx)+1, embedding_dim, units)
    decoder = Decoder(len(ref_word2idx)+1, DECODER_NUM_LAYERS, embedding_dim, units*2, training=True)
    optimizer = tf.keras.optimizers.Adam()
    # prepare to train
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    if tf.train.latest_checkpoint(checkpoint_dir):
          print('Restoring checkpoint')
    print('Starting training')
    #train
    s = time.time()
    end_id = ref_word2idx['<end>']
    teacher_force_prob = 1
    for epoch in range(EPOCHS):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state(BATCH_SIZE)
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss, all_preds, all_targets, all_inputs, grads = train_step(inp, targ, enc_hidden, ref_word2idx, ref_idx2word, teacher_force_prob)
            #ipdb.set_trace()
            preds = all_preds.numpy()
            targets = all_targets.numpy()
            inputs = all_inputs.numpy()
            total_loss += batch_loss
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss))
                print('----------')
                # show bleu score for a random sentence in batch
                b = np.random.choice(len(all_preds))
                mr_info = ' '.join([mr_idx2word[t] for t in inp.numpy()[b] if t != 0])
                training_pred = [ref_idx2word[p] for p in preds[b] if p > 0 and p != end_id]
                print('prediction made in training: ', training_pred)
                pred_sentence, _, _ = evaluate(encoder, decoder, mr_info, training_info)
                print('prediction in evaluation: ', pred_sentence)
                input_sentence = [ref_idx2word[t] for t in inputs[b] if t > 0]
                print('input: ', input_sentence)
                target_sentence = [ref_idx2word[t] for t in targets[b] if t > 0 and t != end_id]
                print('target: ', target_sentence)
                bleu = nltk.translate.bleu_score.sentence_bleu([target_sentence], pred_sentence)
                print('Bleu score', bleu)
                print('----------')
        if epoch % 2 == 0 and TEACHER_FORCING:
              teacher_force_prob *= 0.9
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    print('Training took', (time.time() - s)/60, 'minutes')
