import numpy as np
import tensorflow as tf
from utils import role_to_id, prep_to_id, event_to_id
from crf_on_top import CRF_ON_TOP
from lstm_crf_explicit import gather_2d

class MLP_CRF(CRF_ON_TOP):
    def __init__(self, is_training, config):
        CRF_ON_TOP.__init__(self, is_training, config)

    def calculate_logits(self):
        print 'MLP_CRF'
        # Input data and labels should be set as placeholders
        self._input_data = tf.placeholder(tf.float32, [self.batch_size, self.n_input])
        self._targets = tf.placeholder(tf.int32, [self.batch_size, self.n_labels])

        outputs = []
        for i in xrange(self.n_labels):
            if self.config.num_layers == 1:
                with tf.variable_scope("mlp_%d" % i):
                    weight = tf.get_variable("weight", [self.n_input, self.size])
                    bias = tf.get_variable("bias", [self.size])
                    
                    layer_output = tf.add(tf.matmul(self._input_data, weight), bias)
                    layer_output = tf.nn.relu(layer_output)

                    if self.is_training and self.config.keep_prob < 1:
                        layer_output = tf.nn.dropout(layer_output, keep_prob = self.config.keep_prob)
            else:
                with tf.variable_scope("mlp_0_%d" % i):
                    weight = tf.get_variable("weight", [self.n_input, self.size])
                    bias = tf.get_variable("bias", [self.size])

                    layer_output = tf.add(tf.matmul(self._input_data, weight), bias)
                    layer_output = tf.nn.relu(layer_output)

                    if self.is_training and self.config.keep_prob < 1:
                        layer_output = tf.nn.dropout(layer_output, keep_prob = self.config.keep_prob)

                for j in xrange(1, self.config.num_layers):
                    with tf.variable_scope("mlp_%d_%d" % (j, i) ):
                        weight = tf.get_variable("weight", [self.size, self.size])
                        bias = tf.get_variable("bias", [self.size])

                        layer_output = tf.add(tf.matmul(layer_output, weight), bias)
                        layer_output = tf.nn.relu(layer_output)

                        if self.is_training and self.config.keep_prob < 1:
                            layer_output = tf.nn.dropout(layer_output, keep_prob = self.config.keep_prob)

            outputs.append(layer_output)


        self.logits = logits = []
        

        for i in xrange(self.n_labels):
            label_class = self.label_classes[i]
            n_classes = len(label_class)
            with tf.variable_scope("output_" + str(i)):
                weight = tf.get_variable("weight", [self.size, n_classes])
                bias = tf.get_variable("bias", [n_classes])

                # ( batch_size, n_classes )
                logit = tf.matmul(outputs[i], weight) + bias

            # logits
            logits.append(logit)

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

        # log_sum = tf.reduce_sum(logit_s, axis = 1) + tf.reduce_sum(logit_o, axis = 1)\
        #  + tf.reduce_sum(logit_t, axis = 1) + tf.reduce_sum(logit_e, axis = 1) + tf.reduce_sum(logit_p, axis = 1)

        # batch_size
        correct_s = self._targets[:,0]
        correct_o = self._targets[:,1]
        correct_t = self._targets[:,2]
        correct_e = self._targets[:,3]
        correct_p = self._targets[:,4]

        # logit_correct = \
        #     gather_2d(logit_t, tf.transpose(tf.stack([tf.range(self.batch_size), correct_t]))) +\
        #     gather_2d(logit_o, tf.transpose(tf.stack([tf.range(self.batch_size), correct_o]))) +\
        #     gather_2d(logit_p, tf.transpose(tf.stack([tf.range(self.batch_size), correct_p]))) +\
        #     gather_2d(logit_e, tf.transpose(tf.stack([tf.range(self.batch_size), correct_e]))) +\
        #     gather_2d(logit_s, tf.transpose(tf.stack([tf.range(self.batch_size), correct_s])))

        self._cost =  tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( logits = logit_s, labels = correct_s ) +\
           tf.nn.sparse_softmax_cross_entropy_with_logits( logits = logit_o, labels = correct_o ) +\
           tf.nn.sparse_softmax_cross_entropy_with_logits( logits = logit_t, labels = correct_t ) +\
           tf.nn.sparse_softmax_cross_entropy_with_logits( logits = logit_e, labels = correct_e ) +\
           tf.nn.sparse_softmax_cross_entropy_with_logits( logits = logit_p, labels = correct_p ) )

        # self._cost = tf.reduce_mean(log_sum - logit_correct)    
        
        if self.is_training:
            self.make_train_op()
        else:
            self.make_test_op()
    
        self._saver =  tf.train.Saver()
            
    def make_test_op(self):
        no_of_theme = no_of_subject = no_of_object =  len(role_to_id)
        no_of_prep = len(prep_to_id)
        no_of_event = len(event_to_id)

        logit_s = self.logits[0]
        logit_o = self.logits[1]
        logit_t = self.logits[2]
        logit_e = self.logits[3]
        logit_p = self.logits[4]

        best_s = tf.cast(tf.argmax(logit_s, axis = 1), np.int32)
        best_o = tf.cast(tf.argmax(logit_o, axis = 1), np.int32)
        best_t = tf.cast(tf.argmax(logit_t, axis = 1), np.int32)
        best_e = tf.cast(tf.argmax(logit_e, axis = 1), np.int32)
        best_p = tf.cast(tf.argmax(logit_p, axis = 1), np.int32)

        # (batch_size, self.n_labels)
        out = tf.transpose( tf.stack( [best_s, best_o, best_t, best_e, best_p ] ) )

        # (self.n_labels, batch_size)
        correct_preds = [tf.equal(out[:,i], self._targets[:,i]) \
                for i in xrange(self.n_labels)]

        # Return number of correct predictions as well as predictions
        self._test_op = ([out[:,i] for i in xrange(self.n_labels)], 
                         [tf.reduce_mean(tf.cast(correct_pred, np.float32)) \
                         for correct_pred in correct_preds])