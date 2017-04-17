'''
Created on Mar 6, 2017

@author: Tuan
'''
from collections import deque
from copy import deepcopy
import copy

import numpy as np
import tensorflow as tf

try:
    from tensorflow.nn.rnn_cell import BasicLSTMCell, DropoutWrapper, MultiRNNCell
except:
    from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell

class LSTM_TREE_CRF(object):
    '''
    '''


    def __init__(self, is_training, config):
        '''
        Parameters:
        ----------
        
        config shoule have:
            config.tree = Tree
        '''

        self.tree = config.tree
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.n_input = n_input = config.n_input
        self.max_grad_norm = config.max_grad_norm
        self.size = size = config.hidden_size
        self.crf_weight = crf_weight = config.crf_weight
        self.node_types = config.tree.node_types

        # This is actually just the same
        # self.label_classes is list of dict
        self.label_classes = label_classes = config.label_classes
        # self.dictionaries is dict of dict
        self.dictionaries = config.tree.dictionaries

        self.n_labels = len(self.node_types)
        
        # Input data and labels should be set as placeholders
        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps, n_input])
        self._targets = tf.placeholder(tf.int32, [batch_size, self.n_labels])

        self._debug = []
        
        # self.n_labels cells for self.n_labels outputs
        lstm_cells = [BasicLSTMCell(size, forget_bias = 0.0, state_is_tuple=True)\
                      for _ in xrange(self.n_labels)]

        # DropoutWrapper is a decorator that adds Dropout functionality
        if is_training and config.keep_prob < 1:
            lstm_cells = [DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)\
                              for lstm_cell in lstm_cells]
        cells = [MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)\
                 for lstm_cell in lstm_cells]
        
        # Initial states of the cells
        # cell.state_size = config.num_layers * 2 * size
        # Size = self.n_labels x ( batch_size x cell.state_size )
        self._initial_state = [cell.zero_state(batch_size, tf.float32) for cell in cells]
        
        # Transformation of input to a list of num_steps data points
        inputs = tf.transpose(self._input_data, [1, 0, 2]) #(num_steps, batch_size, n_input)
        inputs = tf.reshape(inputs, [-1, n_input]) # (num_steps * batch_size, n_input)
        
        with tf.variable_scope("hidden"):
            weight = tf.get_variable("weight", [n_input, size])
            bias = tf.get_variable("bias", [size])

            inputs = tf.matmul(inputs, weight) + bias

        inputs = tf.split( int(0), num_steps, inputs) # num_steps * ( batch_size, size )
        
        outputs_and_states = []
        
        # A list of n_labels values
        # Each value is (output, state)
        # output is of size:   num_steps * ( batch_size, size )
        # state is of size:   ( batch_size, cell.state_size )
        for i in xrange(self.n_labels):
            with tf.variable_scope("lstm" + str(i)):
                output_and_state = tf.nn.rnn(cells[i], inputs, initial_state = self._initial_state[i])
                outputs_and_states.append(output_and_state)
                
        
        
        # n_labels x ( batch_size, size )
        outputs = [output_and_state[0][-1]\
                   for output_and_state in outputs_and_states]
        
        # n_labels x ( batch_size, cell.state_size )
        self._final_state = [output_and_state[1]\
                   for output_and_state in outputs_and_states]
        
        # self.n_labels x ( batch_size, n_classes )
        self.logits = logits = {}
        
        for slot in self.node_types:
            n_classes = len(self.dictionaries[slot])
            with tf.variable_scope("output_" + slot):
                weight = tf.get_variable("weight", [size, n_classes])
                bias = tf.get_variable("bias", [n_classes])

                # ( batch_size, n_classes )
                logit = tf.matmul(outputs[i], weight) + bias
            
            # logits
            logits[slot] = logit
        
        log_sum = self.tree.sum_over(crf_weight, batch_size, logits)
        
        logit_correct = self.tree.calculate_logit_correct(crf_weight, batch_size, logits, self._targets)
        
        self._cost =  tf.reduce_mean(log_sum - logit_correct)
            
        if is_training:
            self.make_train_op( )
        else:
            self.make_test_op( )
    
        self._saver =  tf.train.Saver()
        
        
    def make_train_op(self):
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        self._train_op = []
            
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          self.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        
    def make_test_op(self):
        # (batch_size, self.n_labels)
        out = self.tree.predict( self.crf_weight, self.batch_size, self.logits )
        
        # (self.n_labels, batch_size)
        correct_preds = [tf.equal(out[:,i], self._targets[:,i]) \
                for i in xrange(self.n_labels)]

        # Return number of correct predictions as well as predictions
        self._test_op = ([out[:,i] for i in xrange(self.n_labels)], 
                         [tf.reduce_mean(tf.cast(correct_pred, np.float32)) \
                         for correct_pred in correct_preds])
    
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))
    
    @property
    def debug(self):
        return self._debug
        
    @property
    def saver(self):
        return self._saver
    
    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
    
    @property
    def test_op(self):
        return self._test_op