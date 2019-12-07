import tensorflow as tf
from tensorflow.keras import layers

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
    hidden_with_time_axis = tf.expand_dims(hidden[0], 1)
    score = self.V(tf.nn.tanh(self.W1(output) + self.W2(hidden_with_time_axis)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * output
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  """ Jurafsky et al's decoder is a 4-layer RNN with 512 LSTM cells. 
      In order to use the concatented forward and backward states from the encoder, I've doubled the amount of cells in the decoder. """
  def __init__(self, vocab_size, num_layers, embedding_dim, hidden_size, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.hidden_size = hidden_size
    self.embedding_dim = embedding_dim
    self.embedding = layers.Embedding(vocab_size, embedding_dim)
    lstm_cell = layers.LSTMCell(hidden_size)
    self.num_layers = num_layers
    self.lstm_cells = layers.StackedRNNCells([lstm_cell] * num_layers)
    self.rnn = tf.keras.layers.RNN(self.lstm_cells, return_state=True)
    self.lstm = tf.keras.layers.LSTM(self.hidden_size, return_sequences=True, return_state=True)
    self.fc = layers.Dense(vocab_size, activation='softmax')
    # used for attention
    self.attention = BahdanauAttention(hidden_size)

  def call(self, x, hidden, enc_output):
    context_vector, attention_weights = self.attention(hidden, enc_output)
    #print('x as input ', x.shape)
    #print('context', context_vector.shape)
    x = self.embedding(x)
    #print('x as embedding', x.shape)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    # initialize decoder with the encoder hidden state
    # and give encoded output as input
    #output, state = self.rnn(x, initial_state=[hidden]*self.num_layers)
    output, state_h, state_c = self.lstm(x, initial_state=hidden)
    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))
    # output shape == (batch_size, vocab)
    x = self.fc(output)
    #print('output', x.shape)
    return x, [state_h, state_c], attention_weights
