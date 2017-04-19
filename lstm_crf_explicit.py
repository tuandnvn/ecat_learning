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
from utils import role_to_id, prep_to_id, event_to_id, DEVICE, TEST_DEVICE

try:
    from tensorflow.nn.rnn_cell import BasicLSTMCell, DropoutWrapper, MultiRNNCell
except:
    from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell

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

def gather_2d_to_shape(params, indices, output_shape):
    flat = gather_2d(params, indices)
    return tf.reshape(flat, output_shape)

# x -> (x, size)
def expand( tensor, size, axis = 1 ):
    return tf.stack([tensor for _ in xrange(size)], axis = axis)

# x -> (size, x)
def expand_first( tensor, size ):
    return tf.stack( [tensor for _ in xrange(size)] )
          
class LSTM_CRF_Exp(object):
    '''
    A model to recognize event recorded in 3d motions
    This version is the explicit version that do the calculation explicitly on a specific graph
    '''
    
    def __init__(self, is_training, config):
        self.config = config
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
            self.A_start_t = A_start_t = tf.get_variable("A_start_t", [no_of_theme])
            '''Theme -- Object '''
            self.A_to = A_to = tf.get_variable("A_to", [no_of_theme, no_of_object])
            '''Theme -- Subject '''
            self.A_ts = A_ts = tf.get_variable("A_ts", [no_of_theme, no_of_subject])
            '''Theme -- Preposition '''
            self.A_tp = A_tp = tf.get_variable("A_tp", [no_of_theme, no_of_prep])
            '''Subject -- Verb '''
            self.A_se = A_se = tf.get_variable("A_se", [no_of_subject, no_of_event])
        
        # Input data and labels should be set as placeholders
        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps, n_input])
        self._targets = tf.placeholder(tf.int32, [batch_size, self.n_labels])
        
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

        # inputs = tf.reshape(inputs, (-1, num_steps, size)) # (batch_size, num_steps, size)
        # For tf.nn.rnn
        inputs = tf.split(inputs, num_steps, axis = 0) # num_steps * ( batch_size, size )
        
        outputs_and_states = []
        
        # A list of n_labels values
        # Each value is (output, state)
        # output is of size:   num_steps * ( batch_size, size )
        # state is of size:   ( batch_size, cell.state_size )
        for i in xrange(self.n_labels):
            with tf.variable_scope("lstm" + str(i)):
                # output_and_state = tf.nn.rnn(cells[i], inputs, initial_state = self._initial_state[i])
                output_and_state =  tf.contrib.rnn.static_rnn (cells[i], inputs, initial_state = self._initial_state[i])
                # output_and_state = tf.nn.dynamic_rnn(cells[i], inputs, dtype=tf.float32, initial_state = self._initial_state[i])
                outputs_and_states.append(output_and_state)
        
        
        # n_labels x ( batch_size, size )
        outputs = [output_and_state[0][-1]\
                   for output_and_state in outputs_and_states]

        # n_labels x ( num_steps, batch_size, size )
        # outputs = [tf.transpose(output_and_state[0], [1, 0, 2])  
        #           for output_and_state in outputs_and_states]
        
        # Last step
        # n_labels x ( batch_size, size )
        #outputs = [tf.gather(output, int(output.get_shape()[0]) - 1) 
        #           for output in outputs]
        
        # n_labels x ( batch_size, cell.state_size )
        self._final_state = [output_and_state[1]\
                   for output_and_state in outputs_and_states]
        
        # self.n_labels x ( batch_size, n_classes )
        self.logits = logits = []
        
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
        
        self._debug = []
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
        log_t_o = tf.reduce_min( crf_weight * tf.transpose(A_to) + expand(logit_o, no_of_theme, axis = 2), 1)
         
        log_t_o += tf.log(tf.reduce_sum( tf.exp(crf_weight * tf.transpose(A_to) +\
                                        expand(logit_o, no_of_theme, axis = 2) -\
                                        expand(log_t_o, no_of_object, axis = 1) ), 1))
        
        
        '''Theme -- Preposition'''
        
        log_t_p = tf.reduce_min(crf_weight * tf.transpose(A_tp) + expand(logit_p, no_of_theme, axis = 2), 1)
        
        log_t_p += tf.log(tf.reduce_sum( tf.exp(crf_weight * tf.transpose(A_tp) +\
                                        expand(logit_p, no_of_theme, axis = 2) -\
                                        expand(log_t_p, no_of_prep, axis = 1) ), 1))
        
        # (batch_size, #Theme)
        theme_values = log_start_t + log_t_o + log_t_p
        
        '''
        
        subject_values will store sums of values that has been passed on edges (Subject, Theme*) and (Subject, Verb)
        
        Verb ------>  Subject  <-------  Theme*
        
        Subject *
        '''
        
        # (batch_size, #Subject)
        log_s_t = tf.reduce_min(crf_weight * A_ts + expand(theme_values, no_of_subject, axis = 2), 1)
        
        log_s_t += tf.log(tf.reduce_sum(tf.exp(crf_weight * A_ts +\
                                        expand(theme_values, no_of_subject, axis = 2) -\
                                        expand(log_s_t, no_of_theme, axis = 1) ), 1))
        
        # (batch_size, #Subject)
        log_s_e = tf.reduce_min(crf_weight * tf.transpose(A_se) + expand(logit_e, no_of_subject, axis = 2), 1)
        
        log_s_e += tf.log(tf.reduce_sum(tf.exp(crf_weight * tf.transpose(A_se) +\
                                        expand(logit_e, no_of_subject, axis = 2) -\
                                        expand(log_s_e, no_of_event, axis = 1) ), 1))
        
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
            crf_weight * gather_2d(A_to, tf.transpose(tf.stack([correct_t, correct_o]))) +\
            crf_weight * gather_2d(A_tp, tf.transpose(tf.stack([correct_t, correct_p]))) +\
            crf_weight * gather_2d(A_ts, tf.transpose(tf.stack([correct_t, correct_s]))) +\
            crf_weight * gather_2d(A_se, tf.transpose(tf.stack([correct_s, correct_e]))) +\
            gather_2d(logit_t, tf.transpose(tf.stack([tf.range(batch_size), correct_t]))) +\
            gather_2d(logit_o, tf.transpose(tf.stack([tf.range(batch_size), correct_o]))) +\
            gather_2d(logit_p, tf.transpose(tf.stack([tf.range(batch_size), correct_p]))) +\
            gather_2d(logit_e, tf.transpose(tf.stack([tf.range(batch_size), correct_e]))) +\
            gather_2d(logit_s, tf.transpose(tf.stack([tf.range(batch_size), correct_s])))
            
        self._cost = tf.reduce_mean(log_sum - logit_correct)    
        
        if is_training:
            self.make_train_op()
        else:
            self.make_test_op()
#             self._test_op = ( logits, A_start_t, A_to, A_ts, A_tp, A_se )
    
        self._saver =  tf.train.Saver()
    
    def make_train_op(self):
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        self._train_op = []
            
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          self.config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
            
    def make_test_op(self):
        no_of_theme = no_of_subject = no_of_object =  len(role_to_id)
        no_of_prep = len(prep_to_id)
        no_of_event = len(event_to_id)
            
        logit_s = self.logits[0]
        logit_o = self.logits[1]
        logit_t = self.logits[2]
        logit_e = self.logits[3]
        logit_p = self.logits[4]
        
        '''---------------------------------------------------------------'''
        '''Message passing algorithm to max over terms of all combinations'''
        '''---------------------------------------------------------------'''
        # For theme
        # In collapsing, two nodes are collapsed into Theme : Object and Preposition
        best_combination_theme = dict( (slot, tf.zeros((self.batch_size, no_of_theme), dtype=np.int32)) for slot in ['Object', 'Preposition'] )

        # For subject
        # In collapsing, two nodes are collapsed into Subject : Theme and Event
        best_combination_subject = dict ( (slot, tf.zeros((self.batch_size, no_of_subject), dtype=np.int32)) for slot in ['Theme', 'Event']) 
        
        # (batch_size, #Theme)
        best_theme_values = logit_t + self.crf_weight * self.A_start_t
        
        # (#Object, batch_size, #Theme)
        o_values = [expand(logit_o[:, o], no_of_theme) + self.crf_weight * self.A_to[:,o]  for o in xrange(no_of_object)]
        best_theme_values += tf.reduce_max(o_values, 0)
        
        # Best value on edge ( Theme -> Object )
        best_combination_theme['Object'] = tf.cast(tf.argmax(o_values, 0), np.int32)
        
        # (#Prep, batch_size, #Theme)
        p_values = [expand(logit_p[:, p],no_of_theme)  + self.crf_weight * self.A_tp[:,p] for p in xrange(no_of_prep)]
        best_theme_values += tf.reduce_max(p_values, 0)
        
        # Best value on edge ( Theme -> Preposition )
        best_combination_theme['Preposition'] = tf.cast(tf.argmax(p_values, 0), np.int32)
        
        # (batch_size, #Subject)
        best_subject_values = logit_s
        
        # Message passing between Theme and Subject
        # (#Theme, batch_size, #Subject)
        t_values = [expand(best_theme_values[:, t], no_of_subject)  + self.crf_weight * self.A_ts[t,:] for t in xrange(no_of_theme)]
        best_subject_values += tf.reduce_max(t_values, 0)
        
        # Best value on edge ( Subject -> Theme )
        # (batch_size, #Subject)
        best_combination_subject['Theme'] = tf.cast(tf.argmax(t_values, 0), np.int32)
        
        
        # Message passing between Subject and Verb
        # (#Event, batch_size, #Subject)
        e_values = [expand(logit_e[:, e], no_of_subject) + self.crf_weight * self.A_se[:,e] for e in xrange(no_of_event)]
        # (batch_size, #Subject)
        best_subject_values += tf.reduce_max(e_values, 0)
        
        # Best value on edge ( Subject -> Event )
        best_combination_subject['Event'] = tf.cast(tf.argmax(e_values, 0), np.int32)
        
        """
        ======================================================
        Propagate the best combination through message passing
        ======================================================
        """
        best_combination = [tf.zeros((self.batch_size, no_of_subject), dtype=np.int32) for _ in xrange(self.n_labels)]
        
        best_combination[0] = expand_first(range(no_of_subject), self.batch_size)
        best_combination[2] = best_combination_subject['Theme']
        best_combination[3] = best_combination_subject['Event']
        
        """
        Propagate from Theme to [Object, Preposition]
        """
        # (batch_size, #Subject)
        q = np.array([[i for _ in xrange(no_of_subject)] for i in xrange(self.batch_size)])
        
        # (batch_size x #Subject, 2)
        indices = tf.reshape( tf.transpose( tf.stack ( [q, best_combination_subject['Theme']]), [1, 2, 0] ), [-1, 2]) 
        
        for index, slot in [(1, 'Object'), (4, 'Preposition')]:
            best_combination[index] = gather_2d_to_shape(best_combination_theme[slot], 
                                                 indices, (self.batch_size, no_of_subject))
        
        # Take the best out of all subject values
        # batch_size
        best_best_subject_values = tf.argmax(best_subject_values, 1)
        
        # (batch_size, 2)
        # Indices on best_combination[index] should have order of (self.batch_size, #Subject)
        indices = tf.transpose( tf.stack([range(self.batch_size), best_best_subject_values]))
        
        # (batch_size, self.n_labels)
        out = tf.transpose(tf.stack([gather_2d( best_combination[t], indices ) for t in xrange(self.n_labels)]))
        
        
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