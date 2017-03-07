'''
Created on Mar 6, 2017

@author: Tuan
'''
from collections import deque
from copy import deepcopy
import copy

import numpy as np
import tensorflow as tf

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
        self.n_input = n_input = config.n_input
        size = config.hidden_size
        self.crf_weight = crf_weight = config.crf_weight
        self.node_types = config.node_types
        self.dictionaries = config.dictionaries
        self.n_labels = len(self.node_types)
        
        # Input data and labels should be set as placeholders
        self._input_data = tf.placeholder(tf.float32, [batch_size, None, n_input])
        self._targets = tf.placeholder(tf.int32, [batch_size, self.n_labels])
        
        # self.n_labels cells for self.n_labels outputs
        lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias = 0.0, state_is_tuple=True)\
                      for _ in xrange(self.n_labels)]

        # DropoutWrapper is a decorator that adds Dropout functionality
        if is_training and config.keep_prob < 1:
            lstm_cells = [tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)\
                              for lstm_cell in lstm_cells]
        cells = [tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)\
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

        inputs = tf.split(0, self._input_data.shape[1], inputs) # num_steps * ( batch_size, size )
        
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
        logits = {}
        
        for slot in xrange(self.node_types):
            n_classes = len(self.dictionaries[slot])
            with tf.variable_scope("output" + str(i)):
                weight = tf.get_variable("weight", [size, n_classes])
                bias = tf.get_variable("bias", [n_classes])

                # ( batch_size, n_classes )
                logit = tf.matmul(outputs[i], weight) + bias
            
            # logits
            logits[slot] = logit
        
        log_sum = self.tree.sum_over(crf_weight, batch_size, logits)
        
        logit_correct = self.tree.calculate_logit_correct(crf_weight, batch_size, logits, self._targets)
        
        self._cost = cost = tf.reduce_mean(log_sum - logit_correct)
            
        if is_training:
            self.make_train_op( cost )
        else:
            self.make_test_op( logits )
    
        self._saver =  tf.train.Saver()
        
        
    def make_train_op(self, cost):
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        self._train_op = []
            
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          self.config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        
    def make_test_op(self, logits):
        self._test_op = ( logits, self.tree.crf )
    
    def calculate_best(self, targets, logits, crf):
        '''---------------------------------------------------------------'''
        '''Message passing algorithm to max over terms of all combinations'''
        '''---------------------------------------------------------------'''
        # For theme
        best_theme_values = np.zeros((no_of_theme, self.batch_size))
        best_combination_theme = np.zeros((no_of_theme, self.batch_size, self.n_labels), dtype=np.int32)

        # For subject
        best_subject_values = np.zeros((no_of_subject, self.batch_size))
        best_combination_subject = np.zeros((no_of_subject, self.batch_size, self.n_labels))

        for t in xrange(no_of_theme):
            best_theme_values[t] = logit_t[:, t] + self.crf_weight * A_start_t[t]
            best_combination_theme[t,:,2] = t

        for t in xrange(no_of_theme):
            o_values = [logit_o[:, o] + self.crf_weight * A_to[t,o] for o in xrange(no_of_object)]
            best_theme_values[t] += np.max(o_values, 0)
            best_combination_theme[t,:,1] = np.argmax(o_values, 0)

        for t in xrange(no_of_theme):
            p_values = [logit_p[:, p] + self.crf_weight * A_tp[t,p] for p in xrange(no_of_prep)]
            best_theme_values[t] += np.max(p_values, 0)
            best_combination_theme[t,:,4] = np.argmax(p_values, 0)

        # Message passing between Theme and Subject
        for s in xrange(no_of_subject):
            best_subject_values[s] += logit_s[:, s]
            t_values = [best_theme_values[t] + self.crf_weight * A_ts[t,s] for t in xrange(no_of_theme)]
            best_subject_values[s] += np.max(t_values, 0)
            best_t = np.argmax(t_values, 0)
            # This could be improve when multidimensional array indexing is supported  
            for index in xrange(self.n_labels):
                for i in xrange(self.batch_size):
                    best_combination_subject[s,i,index] = best_combination_theme[best_t[i],i,index]
            
            best_combination_subject[s,:,0] = s

        # Message passing between Subject and Verb
        for s in xrange(no_of_subject):
            e_values = [self.crf_weight * A_se[s,e] + logit_e[:, e] for e in xrange(no_of_event)]
            best_subject_values[s] += np.max(e_values, 0)
            best_combination_subject[s,:,3] = np.argmax(e_values, 0)

        # Take the best out of all subject values
        # batch_size
        best_best_subject_values = np.argmax(best_subject_values, 0)

        out = np.zeros((self.batch_size, self.n_labels))
        for i in xrange(self.batch_size):
            out[i] = best_combination_subject[best_best_subject_values[i], i, :]
            
        correct_preds = [np.equal(out[:,i], targets[:,i]) \
                for i in xrange(self.n_labels)]


        # Return number of correct predictions as well as predictions
        return ([out[:,i] for i in xrange(self.n_labels)], 
                         [np.sum(correct_pred.astype(np.float32)) / self.batch_size \
                         for correct_pred in correct_preds])
    
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))
    
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