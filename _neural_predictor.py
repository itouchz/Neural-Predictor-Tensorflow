import tensorflow as tf
from tensorflow.keras import layers

class DirectedGCN(tf.Module):
  def __init__(self, input_dim, output_dim, name=None):
    super(Dense, self).__init__(name=name)
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.weight1 = tf.Variable(tf.zeros((input_dim, output_dim)))
    self.weight2 = tf.Variable(tf.zeros((input_dim, output_dim)))
    self.dropout = layers.Dropout(0.1)
    self.reset()

  @staticmethod
  def normalize(adj):
    return tf.keras.utils.normalize(x)

  def reset(self):
    tf.keras.initializers.GlorotUniform(self.weight1.value)
    tf.keras.initializers.GlorotUniform(self.weight2.value)
  
  def __call__(self, x, adj):
    norm_adj = self.normalize(adj)
    x1 = tf.keras.activations.relu(tf.matmul(norm_adj, tf.matmul(x, self.weight1)))
    inv_norm_adj = self.normalize(adj.transpose(1, 2))
    x2 = tf.keras.activations.relu(tf.matmul(inv_norm_adj, tf.matmul(x, self.weight2)))
    x = (output1 + output2) / 2
    x = self.dropout(x)
    return x

class NeuralPredictor(tf.Module):
  def __init__(self, initial_hidden=5, gcn_hidden=144, gcn_layers=3, linear_hidden=128, name=None):
    super().__init__(name=name)
    self.gcn = [DirectedGCN(initial_hidden if i == 0 else gcn_hidden, gcn_hidden) for i in range(gcn_layers)]
    self.gcn = tf.keras.Sequential(self.gcn)
    self.dropout = tf.nn.Dropout(0.1)
    self.dense1 = layers.Dense(linear_hidden, use_bias=False)
    self.dense2 = layers.Dense(1, use_bias=False)

  @staticmethod
  def graph_pooling(x, num_vertices):
    x = tf.reduce_sum(x, 1)
    return tf.divide(x, tf.broadcast_to(tf.expand_dims(num_vertices, -1), x))

  def __call__(self, x):
    numv, adj, out = x["num_vertices"], x["adjacency"], x["operations"]
    gs = adj.shape[1]  # graph node number
    adj_with_diag = normalize(adj + tf.eye(gs, device=adj.device))  # assuming diagonal is not 1
    for layer in self.gcn:
        x = layer(x, adj_with_diag)
    x = graph_pooling(x, numv)
    x = self.dense1(x)
    x = self.dropout(x)
    x = self.dense2(x).reshape(-1)
    return x