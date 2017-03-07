'''
Created on Mar 4, 2017

@author: Tuan
'''
'''
A work around for gathering (correspond to indexing on numpy array with another numpy array)
- Tensorflow couldn't run gradient for this 

Issue: https://github.com/tensorflow/tensorflow/issues/206

Workaround: Turn the original params and indices to one dimension, then turn back to 2 dimensions
'''

import numpy as np
import tensorflow as tf
from utils import role_to_id, prep_to_id, event_to_id, DEVICE


def gather_2d(params, indices):
    # only for two dim now
    shape = params.get_shape().as_list()
    assert len(shape) == 2, 'only support 2d matrix'
    indices_shape = indices.get_shape().as_list()
    assert indices_shape[1] == 2, 'only support indexing on both dimensions'
    
    flat = tf.reshape(params, [shape[0] * shape[1]])
#     flat_idx = tf.slice(indices, [0,0], [shape[0],1]) * shape[1] + tf.slice(indices, [0,1], [shape[0],1])
#     flat_idx = tf.reshape(flat_idx, [flat_idx.get_shape().as_list()[0]])
    
    flat_idx = indices[:,0] * shape[1] + indices[:,1]
    return tf.gather(flat, flat_idx)
            
class LSTM_CRF_Exp(object):
    '''
    A model to recognize event recorded in 3d motions
    This version is the explicit version that do the calculation explicitly on a specific graph
    '''
    
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.n_input = n_input = config.n_input
        self.label_classes = label_classes = config.label_classes
        self.n_labels = len(self.label_classes)
        size = config.hidden_size
        self.crf_weight = crf_weight = config.crf_weight
        
        '''
                                       Start
                                         |
                                         |
                                         |            
        Verb ------  Subject  -------  Theme  --------- Object
                                         |
                                         |
                                         |
                                    Preposition
        '''
        no_of_theme = no_of_subject = no_of_object =  len(role_to_id)
        no_of_prep = len(prep_to_id)
        no_of_event = len(event_to_id)
        
        with tf.variable_scope("crf"):
            '''Start -- Theme '''
            A_start_t = tf.get_variable("A_start_t", [no_of_theme])
            '''Theme -- Object '''
            A_to = tf.get_variable("A_to", [no_of_theme, no_of_object])
            '''Theme -- Subject '''
            A_ts = tf.get_variable("A_ts", [no_of_theme, no_of_subject])
            '''Theme -- Preposition '''
            A_tp = tf.get_variable("A_tp", [no_of_theme, no_of_prep])
            '''Subject -- Verb '''
            A_se = tf.get_variable("A_se", [no_of_subject, no_of_event])
        
        # Input data and labels should be set as placeholders
        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps, n_input])
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

        inputs = tf.split(0, num_steps, inputs) # num_steps * ( batch_size, size )
        
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
        logits = []
        
        for i in xrange(self.n_labels):
            label_class = label_classes[i]
            n_classes = len(label_class)
            with tf.variable_scope("output" + str(i)):
                weight = tf.get_variable("weight", [size, n_classes])
                bias = tf.get_variable("bias", [n_classes])

                # ( batch_size, n_classes )
                logit = tf.matmul(outputs[i], weight) + bias
            
            # logits
            logits.append(logit)
        
            
        
        if is_training:
            '''----------------------------------------------------------------------------'''
            '''Message passing algorithm to sum over exponentinal terms of all combinations'''
            '''----------------------------------------------------------------------------'''
            logit_s = logits[0]
            logit_o = logits[1]
            logit_t = logits[2]
            logit_e = logits[3]
            logit_p = logits[4]
            
            # Calculate log values for Node Theme and Subject
            # Which is 2 inner nodes (we don't need to store log values for leaf nodes)
            # Message passing between Start and Theme; Theme and Object ; Theme and Preposition
    
            def expand( logit, size ):
                return tf.matmul(tf.expand_dims(logit, axis = 1), tf.ones((1, size)) )
            
            '''
            
            theme_values will store sums of values that has been passed through Start, Object, Preposition
                                            Start
                                             |
                                             |
                                             |
                                             v            
            Verb ------  Subject  -------  Theme  <--------- Object
                                             ^
                                             |
                                             |
                                             |
                                        Preposition
                                        
                                        
            Verb ------  Subject  -------  Theme*
            '''
            '''Start -- Theme '''
            # (batch_size, #Theme)
            log_start_t = logit_t + crf_weight * A_start_t
            
            '''Theme -- Object '''
            # (batch_size, #Theme)
            log_t_o = tf.reduce_min([(crf_weight * A_to[:,o] + expand(logit_o[:, o], no_of_theme)) 
                                    for o in xrange(no_of_object)], 0)
            
            log_t_o += tf.log(tf.reduce_sum([tf.exp(crf_weight * A_to[:,o] +\
                                                     expand(logit_o[:, o], no_of_theme) -\
                                                      log_t_o) 
                                    for o in xrange(no_of_object)], 0))
            
            '''Theme -- Preposition'''
            # (batch_size, #Theme)
            log_t_p = tf.reduce_min([(crf_weight * A_tp[:,p] + expand(logit_p[:, p], no_of_theme)) 
                                    for p in xrange(no_of_prep)], 0)
            
            log_t_p += tf.log(tf.reduce_sum([tf.exp(crf_weight * A_tp[:,p] +\
                                                     expand(logit_p[:, p], no_of_theme) -\
                                                      log_t_p) 
                                    for p in xrange(no_of_prep)], 0))
            
            # (batch_size, #Theme)
            theme_values = log_start_t + log_t_o + log_t_p
            
            '''
            
            subject_values will store sums of values that has been passed on edges (Subject, Theme*) and (Subject, Verb)
            
            Verb ------>  Subject  <-------  Theme*
            
            Subject *
            '''
            
            # (batch_size, #Subject)
            log_s_t = tf.reduce_min([(crf_weight * A_ts[t,:] + expand(theme_values[:, t], no_of_subject)) 
                                    for t in xrange(no_of_theme)], 0)
            
            log_s_t += tf.log(tf.reduce_sum([tf.exp(crf_weight * A_ts[t,:] +\
                                                     expand(theme_values[:, t], no_of_subject) -\
                                                      log_s_t) 
                                    for t in xrange(no_of_theme)], 0))
            
            # (batch_size, #Subject)
            log_s_e = tf.reduce_min([(crf_weight * A_se[:,e] + expand(logit_e[:, e], no_of_subject)) 
                                    for e in xrange(no_of_event)], 0)
            
            log_s_e += tf.log(tf.reduce_sum([tf.exp(crf_weight * A_se[:,e] +\
                                                     expand(logit_e[:, e], no_of_subject) -\
                                                      log_s_e) 
                                    for e in xrange(no_of_event)], 0))
            
            subject_values = tf.transpose(logit_s + log_s_t + log_s_e)
    
            # Sum over all possible values of subject
            # batch_size
            log_sum = tf.reduce_min(subject_values, 0)
            log_sum += tf.log(tf.reduce_sum(tf.exp(subject_values - log_sum), 0))
    
            # This could be improve when multidimensional array indexing is supported
            # Known issue
            # https://github.com/tensorflow/tensorflow/issues/206
            # Currently formularizing is ok, but gpu couldn't learn gradient 
    
            # batch_size
            correct_s = self._targets[:,0]
            correct_o = self._targets[:,1]
            correct_t = self._targets[:,2]
            correct_e = self._targets[:,3]
            correct_p = self._targets[:,4]
    
            logit_correct = \
                crf_weight * tf.gather(A_start_t, correct_t) +\
                crf_weight * gather_2d(A_to, tf.transpose(tf.pack([correct_t, correct_o]))) +\
                crf_weight * gather_2d(A_tp, tf.transpose(tf.pack([correct_t, correct_p]))) +\
                crf_weight * gather_2d(A_ts, tf.transpose(tf.pack([correct_t, correct_s]))) +\
                crf_weight * gather_2d(A_se, tf.transpose(tf.pack([correct_s, correct_e]))) +\
                gather_2d(logit_t, tf.transpose(tf.pack([tf.range(batch_size), correct_t]))) +\
                gather_2d(logit_o, tf.transpose(tf.pack([tf.range(batch_size), correct_o]))) +\
                gather_2d(logit_p, tf.transpose(tf.pack([tf.range(batch_size), correct_p]))) +\
                gather_2d(logit_e, tf.transpose(tf.pack([tf.range(batch_size), correct_e]))) +\
                gather_2d(logit_s, tf.transpose(tf.pack([tf.range(batch_size), correct_s])))
                
            self._cost = cost = tf.reduce_mean(log_sum - logit_correct)
            
            self._lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            self._train_op = []
                
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                              config.max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        else:
            self._test_op = ( logits, A_start_t, A_to, A_ts, A_tp, A_se )
    
        self._saver =  tf.train.Saver()
    
    def calculate_best(self, targets, logits, A_start_t, A_to, A_ts, A_tp, A_se):
        no_of_theme = no_of_subject = no_of_object =  len(role_to_id)
        no_of_prep = len(prep_to_id)
        no_of_event = len(event_to_id)
            
        logit_s = logits[0]
        logit_o = logits[1]
        logit_t = logits[2]
        logit_e = logits[3]
        logit_p = logits[4]
        
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