import pandas as pd
import sys
import numpy as np
from data_manager import create_dataset, load_data_tensors, max_length, load_text_data
import tensorflow as tf
import tensorflow_probability as tfp
from models import Encoder, BahdanauAttention, Decoder
import time
import os
import json
import nltk
from generate import calculate_mean_bleu_score, load_training_info
import argparse
import helpers

BATCH_SIZE = 32
embedding_dim = 128
units = 512

parser = argparse.ArgumentParser(description='Train the model for E2E restaurant description generation')
parser.add_argument("training_data", type=str,
                    help="The path to the training data file")
parser.add_argument("val_data", type=str,
                    help="The path to the validation data file")
parser.add_argument("-id", "--identifier", default='',
                    help="The identifier used to define checkpoint and training info directories")
parser.add_argument("-r", "--restore-checkpoint", default=False, action='store_true',
                    help="Whether to start training from a previous checkpoint")
parser.add_argument("-num", "--num-examples", type=int,
                    help="Test using only a subsample of training data (for development purposes)")
parser.add_argument("-e", "--epochs", type=int, default=5,
                    help="Amount of epochs to train for")
parser.add_argument("-f", "--metrics-file", default='val_metrics',
                    help="File to save validation bleus and batch losses to")
parser.add_argument("-tf", "--teacher-forcing", default=True, action='store_false',
                    help="Disable teacher forcing in training or not")

def save_training_info(training_info_file, ref_word2idx, ref_idx2word, mr_word2idx, mr_idx2word, max_length_targ, max_length_inp, embedding_dim, units):
    print('Saving training information to', training_info_file)
    training_info = {}
    training_info['ref_word2idx'] = ref_word2idx
    training_info['ref_idx2word'] = ref_idx2word
    training_info['mr_word2idx'] = mr_word2idx
    training_info['mr_idx2word'] = mr_idx2word
    training_info['max_length_targ'] = max_length_targ
    training_info['max_length_inp'] = max_length_inp
    training_info['embedding_dim'] = embedding_dim
    training_info['units'] = units
    helpers.save_to_pickle(training_info, training_info_file)

def loss_function(real, pred):
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  loss_ = loss_object(real, pred)
  # ignore padded parts of the input
  # this can be done because 0 is a special token
  pad_mask = tf.math.logical_not(tf.math.equal(real, 0))
  pad_mask = tf.cast(pad_mask, dtype=loss_.dtype)
  loss_ *= pad_mask
  return tf.reduce_mean(loss_)

@tf.function
def train_step(encoder, decoder, optimizer, inp, targ, enc_hidden, ref_word2idx, ref_idx2word, teacher_force_prob, sample_prediction=True):
  loss = 0
  print('Teacher forcing probability', teacher_force_prob)
  with tf.GradientTape() as tape:
    enc_output, forward_hidden, backward_hidden = encoder(inp, enc_hidden)
    # initialize using the concatenated forward and backward states
    dec_hidden = tf.keras.layers.Concatenate()([forward_hidden, backward_hidden])
    dec_input = tf.expand_dims([ref_word2idx['<start>']] * BATCH_SIZE, 1)
    for t in range(1, targ.shape[1]):
      predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_output)
      loss += loss_function(targ[:, t], predictions)
      predicted_tokens = None
      if np.random.uniform() < teacher_force_prob:
        dec_input = tf.expand_dims(targ[:, t], 1)
      else:
        # either sample the prediction made by the network or use the best one
        if sample_prediction:
          pred_dist = tfp.distributions.Multinomial(total_count=1, logits=predictions)
          predicted_tokens = tf.transpose(tf.argmax(pred_dist.sample(1), axis=2))
        else:
          predicted_tokens = tf.argmax(predictions, axis=1)
        dec_input = predicted_tokens
  batch_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return batch_loss

def save_training_metrics(losses, val_bleus, metrics_file):
    print('Saving metrics from', len(losses), 'batches to', metrics_file)
    df = pd.DataFrame(list(zip(val_bleus, losses)), columns=['bleu', 'batch_loss'])
    df.to_csv(metrics_file)

def train(data_file, dev_data_file, checkpoint_dir, training_info_file, restore_checkpoint, num_training_examples, teacher_forcing, epochs):
    print('Loading data')
    input_tensor, target_tensor, ref_word2idx, ref_idx2word, mr_word2idx, mr_idx2word = load_data_tensors(data_file, num_training_examples)
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
    save_training_info(training_info_file,
                        ref_word2idx, 
                        ref_idx2word, 
                        mr_word2idx, 
                        mr_idx2word, 
                        max_length_targ, 
                        max_length_inp,
                        embedding_dim,
                        units)
    training_info = load_training_info(training_info_file)
    encoder = Encoder(len(mr_word2idx)+1, embedding_dim, units)
    decoder = Decoder(len(ref_word2idx)+1, embedding_dim, units*2, training=True)
    optimizer = tf.keras.optimizers.Adam()
    # prepare to train
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)
    if restore_checkpoint:
      print('Restoring checkpoint from', checkpoint_dir)
      if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
      else:
        print('Could not restore checkpoint, folder', checkpoint_dir, 'not found')
    print('Starting training')
    s = time.time()
    end_id = ref_word2idx['<end>']
    teacher_force_prob = 1
    dev_data = load_text_data(dev_data_file, 200)
    val_bleus = []
    losses = []
    for epoch in range(epochs):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state(BATCH_SIZE)
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss = train_step(encoder, decoder, optimizer, inp, targ, enc_hidden, ref_word2idx, ref_idx2word, teacher_force_prob)
            total_loss += batch_loss
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss))
            # run through validation set every 500 batch  
            if (epoch*len(input_tensor)//BATCH_SIZE + batch) % 500 == 0:
                val_bleu = calculate_mean_bleu_score(dev_data, encoder, decoder, training_info, sampling=True)
                print('Validation bleu score', val_bleu)
                val_bleus.append(val_bleu)
                losses.append(batch_loss)
        if not teacher_forcing and (epoch + 1) % 2 == 0:
            teacher_force_prob *= 0.9
        # save the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            print('Saving the model')
            checkpoint.save(file_prefix = checkpoint_prefix)
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    print('Training took', (time.time() - s)/60, 'minutes')
    return losses, val_bleus

if __name__ == '__main__':
    args = parser.parse_args()
    data_file = args.training_data
    dev_data_file = args.val_data
    identifier = args.identifier
    restore_checkpoint = args.restore_checkpoint
    num_examples = args.num_examples
    metrics_file = args.metrics_file
    teacher_forcing = args.teacher_forcing
    epochs = args.epochs
    print('Using teacher forcing', teacher_forcing)
    checkpoint_dir = 'training_checkpoints' + identifier
    training_info_file = 'training_info' + identifier + '.pkl'

    losses, val_bleus = train(data_file,
                              dev_data_file,
                              checkpoint_dir,
                              training_info_file,
                              restore_checkpoint,
                              num_examples,
                              teacher_forcing,
                              epochs)
    save_training_metrics(losses, val_bleus, metrics_file)