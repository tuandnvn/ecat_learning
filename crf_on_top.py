import numpy as np
import tensorflow as tf
from utils import role_to_id, prep_to_id, event_to_id, DEVICE, TEST_DEVICE
from lstm_crf_explicit import gather_2d, gather_2d_to_shape, expand, expand_first 

class CRF_ON_TOP(object):
    def __init__(self, is_training, config):
        self.init_params(is_training, config)
        self.calculate_logits()
        self.train_test_crf()

        # This is just for convenience
        # These states are used for sequential learning
        self._initial_state = []
        self._final_state = []

    def init_params(self, is_training, config):
    	print 'Init params'
        self.is_training = is_training
    	self.config = config
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.n_input = n_input = config.n_input
        self.label_classes = label_classes = config.label_classes
        self.n_labels = len(self.label_classes)
        self.size = config.hidden_size
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

    def calculate_logits(self):
    	print 'Nothing'

    def train_test_crf(self):
    	self._debug = []
        '''----------------------------------------------------------------------------'''
        '''Message passing algorithm to sum over exponentinal terms of all combinations'''
        '''----------------------------------------------------------------------------'''
        no_of_theme = no_of_subject = no_of_object =  len(role_to_id)
        no_of_prep = len(prep_to_id)
        no_of_event = len(event_to_id)

        logit_s = self.logits[0]
        logit_o = self.logits[1]
        logit_t = self.logits[2]
        logit_e = self.logits[3]
        logit_p = self.logits[4]

        A_start_t = self.A_start_t
        A_to = self.A_to
        A_ts = self.A_ts
        A_tp = self.A_tp
        A_se = self.A_se

        
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
        log_start_t = logit_t + self.crf_weight * A_start_t
        
        '''Theme -- Object '''
        # (batch_size, #Theme)
        log_t_o = tf.reduce_min( self.crf_weight * tf.transpose(A_to) + expand(logit_o, no_of_theme, axis = 2), 1)
         
        log_t_o += tf.log(tf.reduce_sum( tf.exp(self.crf_weight * tf.transpose(A_to) +\
                                        expand(logit_o, no_of_theme, axis = 2) -\
                                        expand(log_t_o, no_of_object, axis = 1) ), 1))
        
        
        '''Theme -- Preposition'''
        
        log_t_p = tf.reduce_min(self.crf_weight * tf.transpose(A_tp) + expand(logit_p, no_of_theme, axis = 2), 1)
        
        log_t_p += tf.log(tf.reduce_sum( tf.exp(self.crf_weight * tf.transpose(A_tp) +\
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
        log_s_t = tf.reduce_min(self.crf_weight * A_ts + expand(theme_values, no_of_subject, axis = 2), 1)
        
        log_s_t += tf.log(tf.reduce_sum(tf.exp(self.crf_weight * A_ts +\
                                        expand(theme_values, no_of_subject, axis = 2) -\
                                        expand(log_s_t, no_of_theme, axis = 1) ), 1))
        
        # (batch_size, #Subject)
        log_s_e = tf.reduce_min(self.crf_weight * tf.transpose(A_se) + expand(logit_e, no_of_subject, axis = 2), 1)
        
        log_s_e += tf.log(tf.reduce_sum(tf.exp(self.crf_weight * tf.transpose(A_se) +\
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
            self.crf_weight * tf.gather(A_start_t, correct_t) +\
            self.crf_weight * gather_2d(A_to, tf.transpose(tf.stack([correct_t, correct_o]))) +\
            self.crf_weight * gather_2d(A_tp, tf.transpose(tf.stack([correct_t, correct_p]))) +\
            self.crf_weight * gather_2d(A_ts, tf.transpose(tf.stack([correct_t, correct_s]))) +\
            self.crf_weight * gather_2d(A_se, tf.transpose(tf.stack([correct_s, correct_e]))) +\
            gather_2d(logit_t, tf.transpose(tf.stack([tf.range(self.batch_size), correct_t]))) +\
            gather_2d(logit_o, tf.transpose(tf.stack([tf.range(self.batch_size), correct_o]))) +\
            gather_2d(logit_p, tf.transpose(tf.stack([tf.range(self.batch_size), correct_p]))) +\
            gather_2d(logit_e, tf.transpose(tf.stack([tf.range(self.batch_size), correct_e]))) +\
            gather_2d(logit_s, tf.transpose(tf.stack([tf.range(self.batch_size), correct_s])))
            
        self._cost = tf.reduce_mean(log_sum - logit_correct)    
        
        if self.is_training:
            self.make_train_op()
        else:
            self.make_test_op()
    
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

        A_start_t = self.A_start_t
        A_to = self.A_to
        A_ts = self.A_ts
        A_tp = self.A_tp
        A_se = self.A_se
        
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
        best_theme_values = logit_t + self.crf_weight * A_start_t
        
        # (#Object, batch_size, #Theme)
        o_values = [expand(logit_o[:, o], no_of_theme) + self.crf_weight * A_to[:,o]  for o in xrange(no_of_object)]
        best_theme_values += tf.reduce_max(o_values, 0)
        
        # Best value on edge ( Theme -> Object )
        best_combination_theme['Object'] = tf.cast(tf.argmax(o_values, 0), np.int32)
        
        # (#Prep, batch_size, #Theme)
        p_values = [expand(logit_p[:, p],no_of_theme)  + self.crf_weight * A_tp[:,p] for p in xrange(no_of_prep)]
        best_theme_values += tf.reduce_max(p_values, 0)
        
        # Best value on edge ( Theme -> Preposition )
        best_combination_theme['Preposition'] = tf.cast(tf.argmax(p_values, 0), np.int32)
        
        # (batch_size, #Subject)
        best_subject_values = logit_s
        
        # Message passing between Theme and Subject
        # (#Theme, batch_size, #Subject)
        t_values = [expand(best_theme_values[:, t], no_of_subject)  + self.crf_weight * A_ts[t,:] for t in xrange(no_of_theme)]
        best_subject_values += tf.reduce_max(t_values, 0)
        
        # Best value on edge ( Subject -> Theme )
        # (batch_size, #Subject)
        best_combination_subject['Theme'] = tf.cast(tf.argmax(t_values, 0), np.int32)
        
        
        # Message passing between Subject and Verb
        # (#Event, batch_size, #Subject)
        e_values = [expand(logit_e[:, e], no_of_subject) + self.crf_weight * A_se[:,e] for e in xrange(no_of_event)]
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