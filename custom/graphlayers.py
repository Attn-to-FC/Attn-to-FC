from keras.models import Model
from keras.layers import Input, Maximum, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, MaxPooling2D, Conv1D, Conv2D, Flatten, Bidirectional, CuDNNGRU, RepeatVector, Permute, TimeDistributed, dot
from keras.optimizers import RMSprop, Adamax
from keras.engine.topology import Layer
import keras
import keras.utils
import keras.backend as K
import tensorflow as tf

class OurCustomGraphLayer(Layer):
    def __init__(self, **kwargs):
        super(OurCustomGraphLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(OurCustomGraphLayer, self).build(input_shape)
    
    def call(self, x):
        assert isinstance(x, list)
        nodes, edges = x
        nodes_shape = tf.shape(nodes)
        edges_shape = tf.shape(edges)

        # make a self-edge for every node
        # if we don't have this, then the aggregation will fail later because
        # many nodes have only one edge, and so the average of all neighbors will
        # just have one node... this is confusing because when the neighbors are
        # concatenated with the node itself, a majority of nodes will have only
        # other node... i.e. it will just look like two nodes instead of an
        # average of several nodes... the solution is to average the neighbors
        # with the node itself, to prevent the node from looking too much like
        # the neighbor 
        edges = tf.eye(nodes_shape[1]) + edges

        # edges is (batch_size, 100, 100)
        kedges = tf.reshape(edges, (edges_shape[0], edges_shape[1], edges_shape[2], 1))
        # kedges should now be (batch_size, 100, 100, 1)
        kedges = tf.tile(kedges, (1, 1, 1, nodes_shape[2]))
        # kedges should now be (batch_size, 100, 100, 256)
        
        # nodes is (batch_size, 100, 256)
        knodes = tf.reshape(nodes, (nodes_shape[0], 1, nodes_shape[1], nodes_shape[2]))
        # knodes should now be (batch_size, 1, 100, 256)
        knodes = tf.tile(knodes, (1, nodes_shape[1], 1, 1))
        # knodes should now be (batch_size, 100, 100, 256)
        
        # element-wise multiplication
        knodes = knodes * kedges

        # now we need to aggregate, which normally would just be a call to reduce_mean
        # but reduce_mean will average in a bunch of zeros, so we need to divide only
        # by the number of nodes that are actually connected i.e. not zeros in the adjacency
        # matrix... but using count_nonzero is really slow, so we kinda hack it by just
        # adding together the adjacency matrix... this is fast, but it assumes that edge
        # values are 1, so it may not work for non-binary edge weights
        knodes = tf.math.reduce_sum(knodes, axis=2, keepdims=True)
        fixmean = tf.math.reduce_sum(kedges, axis=2, keepdims=True)
        knodes = knodes / fixmean

        #knodes = tf.math.reduce_mean(knodes, axis=2, keepdims=True)

        # knodes should now be (batch_size, 100, 1, 256), so reshape to (batch_size, 100, 256)
        knodes = tf.reshape(knodes, (nodes_shape[0], nodes_shape[1], nodes_shape[2]))

        # remove all the nans that will result from the fixmean calculation
        knodes = tf.where(tf.is_nan(knodes), tf.zeros_like(knodes), knodes)
        
        # knodes is now the updated nodes matrix for one iteration of the graph
        return knodes
        
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        nodes_shape, edges_shape = input_shape
        return (nodes_shape)
