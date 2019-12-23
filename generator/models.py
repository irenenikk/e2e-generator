import tensorflow as tf
from tensorflow.keras import layers
from IPython.core.debugger import set_trace

class Encoder(tf.keras.Model):
  """ The encoder is a bidirectional LSTM as in the winning system by Jurafsky et al. """
  def __init__(self, vocab_size, embedding_dim, hidden_size, batch_size):
    super(Encoder, self).__init__()
    self.batch_size = batch_size
    self.hidden_size = hidden_size
    self.embedding = layers.Embedding(vocab_size, embedding_dim)
    # use a bidirectional lstm
    lstm_layer = layers.LSTM(hidden_size, 
                            return_sequences=True, 
                            return_state=True)
    self.bidirectional_lstm = layers.Bidirectional(lstm_layer)

  def call(self, x, hidden):
    x = self.embedding(x)
    output, forward_hidden, forward_mem, backward_hidden, backward_mem = self.bidirectional_lstm(x, initial_state=hidden)
    return output, forward_hidden, forward_mem, backward_hidden, backward_mem

  def initialize_hidden_state(self):
    forward_state = [tf.zeros([self.batch_size, self.hidden_size])] * 2
    backward_state = [tf.zeros([self.batch_size, self.hidden_size])] * 2
    return forward_state + backward_state

class BahdanauAttention(layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = layers.Dense(units)
    self.W2 = layers.Dense(units)
    self.V = layers.Dense(1)

  def call(self, hidden, output):
    # hidden is two dimensional, storing both hidden states and memory
    # only use the hidden state when calculating attention
    # it has to be expanded because the output contains time axis information
    hidden_with_time_axis = tf.expand_dims(hidden[-1], 1)
    score = self.V(tf.nn.tanh(self.W1(output) + self.W2(hidden_with_time_axis)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * output
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  """ Jurafsky et al's decoder is a 4-layer RNN with 512 LSTM cells. 
      In order to use the concatented forward and backward states from the encoder, 
      we've doubled the amount of cells in the decoder. """
  def __init__(self, vocab_size, num_layers, embedding_dim, hidden_size, batch_sz, training):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.embedding_dim = embedding_dim
    self.embedding = layers.Embedding(vocab_size, embedding_dim)
    self.lstm_cells = [tf.keras.layers.LSTMCell(self.hidden_size) for _ in range(num_layers)]
    self.rnn = tf.keras.layers.RNN(self.lstm_cells, return_sequences=True, return_state=True)
    self.fc = layers.Dense(vocab_size, activation='softmax')
    self.dropout = tf.keras.layers.Dropout(0.3)
    # used for attention
    self.attention = BahdanauAttention(hidden_size)
    # dropout needs to know if we are training
    self.training = training

  def call(self, x, hidden, enc_output):
    #print('x', x.shape)
    context_vector, attention_weights = self.attention(hidden[0], enc_output)
    x = self.embedding(x)
    #print('x embedding', x.shape)
    expanded = tf.expand_dims(context_vector, 1)
    #print('expanded shape', expanded.shape)
    x = tf.concat(expanded, axis=-1)
    # initialize decoder with the encoder hidden state
    # and give encoded output as input
    output, *states = self.rnn(x, initial_state=hidden)
    output = tf.reshape(output, (-1, output.shape[2]))
    x = self.dropout(x, training=self.training)
    x = self.fc(output)
    # return the last state
    return x, states, attention_weights
