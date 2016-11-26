# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import time
import datetime

import numpy as np
import tensorflow as tf
#from tf.nn import rnn, rnn_cell

import codecs
import collections
import random
from collections import Counter

import argparse
import xml.etree.ElementTree as ET
import glob
import nltk
from nltk.stem.porter import PorterStemmer 

import sys
import pickle
import os
import shutil
import os.path

from sklearn.metrics import confusion_matrix


def print_and_log(log_str):
    print (log_str)
    logging.info(log_str)
    
SESSION_NAME = "session_name"
SESSION_DATA = "session_data"
SESSION_EVENTS = "session_events"

ps = PorterStemmer() 

role_to_id = {'None' : 0, 'Subject': 1, 'Object': 2, 'Theme' : 3}
event_to_id = { 'None': 0, 'push' : 1, 'pull' : 2 , 'roll': 3, 'slide' : 4}
prep_to_id = {'None': 0, 'Across': 1, 'From': 2, 'To': 3}

id_to_role = {}
id_to_event = {}
id_to_prep = {}

for key, value in role_to_id.iteritems():
    id_to_role[value] = key
for key, value in event_to_id.iteritems():
    id_to_event[value] = key
for key, value in prep_to_id.iteritems():
    id_to_prep[value] = key

def from_str_labels_to_id_labels(rig_role, glyph_role_1, glyph_role_2, event, prep):
    rig_role = role_to_id[rig_role]
    glyph_role_1 = role_to_id[glyph_role_1]
    glyph_role_2 = role_to_id[glyph_role_2]
    event = event_to_id[event]
    prep = prep_to_id[prep]
    
    return (rig_role, glyph_role_1, glyph_role_2, event, prep)

def from_id_labels_to_str_labels(rig_role, glyph_role_1, glyph_role_2, event, prep):
    rig_role = id_to_role[rig_role]
    glyph_role_1 = id_to_role[glyph_role_1]
    glyph_role_2 = id_to_role[glyph_role_2]
    event = id_to_event[event]
    prep = id_to_prep[prep]
    
    return (rig_role, glyph_role_1, glyph_role_2, event, prep)

project_data = {}

data_length = None
# For each data sample, we have to learn an output of 5 values
# (role_to_id, role_to_id, role_to_id, event_to_id, prep_to_id)
label_classes = [role_to_id, role_to_id, role_to_id, event_to_id, prep_to_id]
num_labels = len(label_classes)

def read_project_data():
    global data_length
    for file_name in glob.glob('data/*.txt'):
        project_name = file_name[file_name.rfind('/') + 1:]
        project_name = project_name[:len(project_name)-4]
        project_data[project_name] = []


        tree = ET.parse(file_name)

        doc = tree.getroot()


        for session_element in doc.findall('session'):
            session_data = {}

            session_name = session_element.attrib['name']
            print(session_name)

            session_data[SESSION_NAME] = session_name
            session_data[SESSION_DATA] = []
            session_data[SESSION_EVENTS] = []

            frame_elements = session_element.findall('data/frame')

            for frame_element in frame_elements:
                object_point_elements = frame_element.findall('o')

                point_data = []
                for object_point_element in object_point_elements:
                    for s in object_point_element.text.split(','):
                        point_data.append(float(s))
                        
                if data_length == None:
                    data_length = len(point_data)

                session_data[SESSION_DATA].append(point_data)
            
#             # Calculate the difference of data points -> gradient feature
            # Move all points to the same coordinations
            session_data[SESSION_DATA] = [[(session_data[SESSION_DATA][i][t] - session_data[SESSION_DATA][0][0])\
                 for t in xrange(data_length)]\
                 for i in xrange(0, len(session_data[SESSION_DATA]))]

            event_elements = session_element.findall('events/event')

            for event_element in event_elements:
                event_str = {}
                event_str['start'] = event_element.attrib['start']
                event_str['end'] = event_element.attrib['end']

                rig_role, glyph_role_1, glyph_role_2, event, prep = event_element.text.split(',')

                event = ps.stem(event)
                
                rig_role, glyph_role_1, glyph_role_2, event, prep =\
                    from_str_labels_to_id_labels(rig_role, glyph_role_1, glyph_role_2, event, prep)

                event_str['label'] = (rig_role, glyph_role_1, glyph_role_2, event, prep)

                session_data[SESSION_EVENTS].append(event_str)
            
#             print ('session name = %s' % session_data[SESSION_NAME])
#             print ('len %d ' % len(session_data[SESSION_DATA]))
#             print ('correct %d ' % correct_no_samples)
        
            project_data[project_name].append(session_data) 



'''Generate a training set and a testing set of data'''
def generate_data(data, config) :
    training_data = []
    testing_data = []
    
    # Flatten the data (collapse the project and session hierarchy into a list of session_data)
    for v in config.train_project_names:
        session_data = random.sample(project_data[v], len(project_data[v]))
        print(len(session_data))

        training_data += session_data[int(config.session_training_percentage[0] * len(session_data)):
                                     int(config.session_training_percentage[1] * len(session_data))]

        testing_data += session_data[int(config.session_testing_percentage[0] * len(session_data)):
                                     int(config.session_testing_percentage[1] * len(session_data))]
    
    return (training_data, testing_data)

def check_validity_label(labels):
    # Event is None -> All other values are None
    if labels[3] == 0:
        for i in xrange(5):
            if labels[i] != 0:
                return False
        return True
    
    # If two objects play the same role return False
    for i in xrange(3):
        for j in xrange(3):
            if i != j and labels[i] == labels[j] and labels[i] != 0:
                return False
    
    # If there is a Theme, there needs to be a Preposition and vice versa
    has_a_theme = False
    for i in xrange(3):
        if labels[i] == 3:
            has_a_theme = True
            break
    if has_a_theme and labels[4] == 0:
        return False
    
    if not has_a_theme and labels[4] != 0:
        return False
    
    return True

'''A function to generate a pair of batch-data (x, y)
Inputs
data: a list of session_data
data_point_size: Vector feature size (63)
num_steps: A fix number of steps for each event (this should be the original num_steps - 1
because data point difference is used instead of )
hop_step: A fix number of frame offset btw two events
Outputs
Take batch_size of data samples, each is a chain of num_steps data points
x: [batch_size, num_steps, data_point_size]

y: [batch_size, num_labels]
'''
def gothrough(data, data_point_size, batch_size, num_steps, hop_step):
    samples = 0   # Number of samples of interpolating
    
    #counters = [Counter() for _ in xrange(num_labels)]

    sample_counter = 0
    for session_data in data:
        
        # This should be the correct number of sample for each session
        # But it could be different with the number of events in the session
        # There is some difference in the way events in session is created
        # For example, when create and annotate a session having frame from 0 to 79
        # I actually create events [0,20] to [60,80] so the right hand side brace should be 
        # [0,20) -> Excluding last frame 
        correct_no_samples = ( len(session_data[SESSION_DATA]) - num_steps ) // hop_step + 1
#         print ('session name = %s' % session_data[SESSION_NAME])
#         print ('len %d ' % len(session_data[SESSION_DATA]))
#         print ('correct %d ' % correct_no_samples)
        
        if correct_no_samples != len(session_data[SESSION_EVENTS]):
            # A step to find session that has problem to fix
            print (session_data[SESSION_NAME])
            print ("correct_no_samples " + str(correct_no_samples))
            print ("session_data_events " + str(len(session_data[SESSION_EVENTS])))
        
            print ("=========================PROBLEMATIC========================")
        else:
            samples += len(session_data[SESSION_EVENTS])
    
    print('Total number of samples' + str(samples))
    
    interpolated_data = np.zeros([samples * num_steps, data_point_size], dtype=np.float32)
    interpolated_lbls = np.zeros([samples, num_labels], dtype=np.int32)
    
    
    for session_data in data:
        session_data_vals = session_data[SESSION_DATA]
        session_data_events = session_data[SESSION_EVENTS]
               
        correct_no_samples = ( len(session_data_vals) - num_steps ) // hop_step + 1
        if correct_no_samples == len(session_data_events):
            for i in range(len(session_data_events)):
                for j in range(num_steps):
                    interpolated_data[( ( sample_counter + i ) * num_steps + j)] =\
                                 session_data_vals[i * hop_step + j]

                event_labels = session_data[SESSION_EVENTS][i]['label']
                
#                 for i, event_label in enumerate(event_labels):
#                     counters[i][event_label] += 1
                
                interpolated_lbls[sample_counter + i] = list(event_labels)
            
        sample_counter += len(session_data_events)
    
    # Number of epoch, each epoch has a batch_size of data 
    epoch_size = samples // batch_size
    
    # Divide the first dimension from samples * num_steps -> (samples, num_steps)
    rearranged_data = interpolated_data.reshape((samples, num_steps, data_point_size))
    # Divide first dimenstion from samples -> epoch_size * batch_size (remove remaining)
    rearranged_data = rearranged_data[:epoch_size * batch_size].\
            reshape((epoch_size, batch_size, num_steps, data_point_size))
    
    rearranged_lbls = interpolated_lbls[:epoch_size * batch_size].\
            reshape((epoch_size, batch_size, num_labels))
    
    for i in range(epoch_size):
        x = rearranged_data[i, :, :,  :]
        y = rearranged_lbls[i, :, :]
        yield (x, y)

class Recognizer(object):
    "A model to recognize event recorded in 3d motions"
    
    def __init__(self, is_training, config):
        with tf.device('/gpu:3'):
            self.batch_size = batch_size = config.batch_size
            self.num_steps = num_steps = config.num_steps
            self.n_input = n_input = config.n_input
            self.label_classes = label_classes = config.label_classes
            self.n_labels = len(self.label_classes)
            self.hop_step = config.hop_step
            size = config.hidden_size
            
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
            
#             outputs_and_states = [tf.nn.rnn(cells[i], inputs, initial_state = self._initial_state[i])\
#                                   for i in xrange(self.n_labels)]
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
            
            cost = 0
            
            # self.n_labels x batch_size
            max_logits = []
            
            # self.n_labels x ( batch_size, n_classes )
            logits = []
            
            role_scope = None
            
            for i in xrange(self.n_labels):
                label_class = label_classes[i]
                n_classes = len(label_class)
                with tf.variable_scope("output" + str(i)):
                    weight = tf.get_variable("weight", [size, n_classes])
                    bias = tf.get_variable("bias", [n_classes])

                    # ( batch_size, n_classes )
                    logit = tf.matmul(outputs[i], weight) + bias
                
                # batch_size
                max_logit = tf.argmax(logit, 1)
                
                max_logits.append(max_logit)
                
                # logits
                logits.append(logit)
                
            self._debug = logits[2]
            
            # batch_size
            loss = tf.nn.seq2seq.sequence_loss_by_example(
                    [logits[i] for i in xrange(self.n_labels)], 
                    [self._targets[:,i] for i in xrange(self.n_labels)],
                    [tf.ones([batch_size]) for i in xrange(self.n_labels)]) 
            
#             losses = [tf.nn.seq2seq.sequence_loss_by_example(
#                     [logits[i]], 
#                     [self._targets[:,i]],
#                     [tf.ones([batch_size])])  for i in xrange(self.n_labels)]
            
            # self.n_labels
#             self._cost = cost = [tf.reduce_sum(loss) / batch_size for loss in losses]
            
            self._cost = cost = tf.reduce_sum(loss) / batch_size

            
            if is_training:
                self._lr = tf.Variable(0.0, trainable=False)
                tvars = tf.trainable_variables()
                self._train_op = []
                
#                 for i in xrange(self.n_labels):
#                     grads, _ = tf.clip_by_global_norm(tf.gradients(cost[i], tvars),
#                                                   config.max_grad_norm)
#                     optimizer = tf.train.GradientDescentOptimizer(self.lr)
#                     self._train_op.append(optimizer.apply_gradients(zip(grads, tvars)))
                    
                grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                                  config.max_grad_norm)
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
                self._train_op = optimizer.apply_gradients(zip(grads, tvars))
                    
            else:
                correct_preds = [tf.equal(tf.cast(max_logits[i], tf.int32), self._targets[:,i]) \
                                for i in xrange(self.n_labels)]
                
#                 self._test_op = max_logits
                # Return number of correct predictions as well as predictions
                self._test_op = ([tf.cast(max_logits[i], tf.int32) for i in xrange(self.n_labels)], 
                                 [tf.reduce_sum (tf.cast(correct_pred, tf.float32)) / batch_size \
                                 for correct_pred in correct_preds])
    
        self._saver = saver = tf.train.Saver()
                
                
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


def run_epoch(session, m, data, eval_op, verbose=False, is_training=True):
    """Runs the model on the given data."""
    start_time = time.time()
    # costs = np.zeros(len(m.label_classes))
    costs = 0
    evals = np.zeros(len(m.label_classes))
    cost_iters = 0
    eval_iters = 0
    state = [session.run(s) for s in m.initial_state]
    # For each label class, create a confusion matrix
    confusion_matrixs = [np.zeros((len(label_classes[i]), len(label_classes[i])), dtype=np.int) 
                         for i in xrange(len(m.label_classes))]
    total_correct_pred = 0
    
    valid_labels = {True: 0, False: 0}
    
    for step, (x, y) in enumerate( gothrough(data, m.n_input, m.batch_size, m.num_steps, m.hop_step) ):
        feed_dict = {}
        feed_dict[m.input_data] = x
        feed_dict[m.targets] = y
        for i in xrange(len(m.initial_state)):
            feed_dict[m.initial_state[i]] = state[i]
    
        debug_val, cost, state, eval_val = session.run([m.debug, m.cost, m.final_state, eval_op], feed_dict)
        
        if not is_training:
            # Unpack the predictions and cost values
            y_pred, eval_val = eval_val
            
        costs += cost
        cost_iters += 1
        eval_iters += 1
        
        if not is_training:
#             print('-----------')
#             print(y)
#             print('===========')
#             print(eval_val)
            evals += eval_val
            
            correct_pred = np.sum(np.all([np.equal(y_pred[i], y[:,i]) \
                                for i in xrange(len(m.label_classes))], axis = 0))
            total_correct_pred += correct_pred
            
            # self.n_label x m.batch_size
            y_pred_array = np.array(y_pred)

            for i in xrange(m.batch_size):
                valid = check_validity_label( y_pred_array[:,i] )
                valid_labels[valid] += 1

                if verbose and not valid:
                    print(from_id_labels_to_str_labels(*y_pred_array[:,i]))
                
            epoch_confusion_matrixs = [confusion_matrix(y[:,i], y_pred[i], label_classes[i].values()) 
                                       for i in xrange(len(m.label_classes))]
        
            for i in xrange(len(m.label_classes)):
                confusion_matrixs[i] += epoch_confusion_matrixs[i]
        
#         if verbose and step % 30 == 0:
#             print('---')
#             print("cost_iters %d, eval_iters %d, Step %d" % (cost_iters, eval_iters, step))
#             print("Rig: cost %.3f, costs %.3f, perplexity: %.3f" % 
#               (cost[0], costs[0], np.exp(costs[0] / cost_iters)))
#             print("Glyph 1: cost %.3f, costs %.3f, perplexity: %.3f" % 
#               (cost[1], costs[1], np.exp(costs[1] / cost_iters)))
#             print("Glyph 2: cost %.3f, costs %.3f, perplexity: %.3f" % 
#               (cost[2], costs[2], np.exp(costs[2] / cost_iters)))
#             print("Event: cost %.3f, costs %.3f, perplexity: %.3f" % 
#               (cost[3], costs[3], np.exp(costs[3] / cost_iters)))
#             print("Preposition: cost %.3f, costs %.3f, perplexity: %.3f" % 
#               (cost[4], costs[4], np.exp(costs[4] / cost_iters)))
            
        if verbose and step % 30 == 0 and step > 0:
            print(debug_val)
            print_and_log("cost %.3f, costs %.3f, iters %d, Step %d, perplexity: %.3f" % 
              (cost, costs, cost_iters, step, np.exp(costs / cost_iters)))
            
    if not is_training:
        print_and_log("Number of valid/Number of invalid = %d/%d" % 
                      (valid_labels[True], valid_labels[False]))
        
        print_and_log("Number of correct predictions = %d, Percentage = %.3f" % 
                      (total_correct_pred, total_correct_pred/ (eval_iters * m.batch_size) ))
        
        print_and_log("Rig accuracy = %.5f" % (evals[0] / eval_iters))
        if verbose:
            print_and_log("-- Confusion matrix --")
            print_and_log(confusion_matrixs[0])
        
        print_and_log("Glyph 1 accuracy = %.5f" % (evals[1] / eval_iters))
        if verbose:
            print_and_log("-- Confusion matrix --")
            print_and_log(confusion_matrixs[1])
        
        print_and_log("Glyph 2 accuracy = %.5f" % (evals[2] / eval_iters))
        if verbose:
            print_and_log("-- Confusion matrix --")
            print_and_log(confusion_matrixs[2])
        
        print_and_log("Event accuracy = %.5f" % (evals[3] / eval_iters))
        if verbose:
            print_and_log("-- Confusion matrix --")
            print_and_log(confusion_matrixs[3])
        
        print_and_log("Preposition accuracy = %.5f" % (evals[4] / eval_iters))
        if verbose:
            print_and_log("-- Confusion matrix --")
            print_and_log(confusion_matrixs[4])
        
    return np.exp(costs / cost_iters)


# Train
# Statistics
# Counter({1: 1888, 0: 835})  72 %
# Counter({2: 1182, 3: 759, 1: 418, 0: 364})  42%
# Counter({3: 1175, 2: 693, 0: 469, 1: 386})  38%
# Counter({3: 1164, 2: 603, 1: 516, 4: 409, 0: 31})  43%
# Counter({2: 838, 0: 783, 1: 681, 3: 421})  30 %
    
'''Train on a subset of sessions for each project'''
'''Training_percentages = Percentage of training sessions/ Total # of sessions'''
class Simple_Train_Test_Config(object):
    
    def __init__(self, project_data):
        # Using all projects for training
        self.train_project_names = project_data.keys()
        self.test_project_names = project_data.keys()
        self.session_training_percentage = (0, 0.6)
        self.session_testing_percentage = (0.6, 1)

'''Only train on a subset of projects'''
'''For each training project, train on all sessions'''
'''Training_percentages = 1'''
class Partial_Train_Test_Config(object):
    # Using a subset of projects for training
    train_project_names = ['pullacross', 'pullfrom', 'pushfrom', 'pushto',
                            'rollacross', 'rollto', 'selfrollacross', 'selfrollto',
                          'selfslidefrom']
    test_project_names = ['pullto', 'pushacross', 'rollfrom', 'selfrollfrom',
                          'selfslideacross', 'selfslideto']
    session_training_percentage = (0, 1)
    session_testing_percentage = (0, 1)

# default mode is to train and test at the same time
TRAIN = 'TRAIN'
TEST = 'TEST'
mode = TRAIN

if __name__ == '__main__':
    # ========================================================================
    # ========================================================================
    # ===========================SETUP TRAIN TEST=============================
    parser = argparse.ArgumentParser(description = 'Train and test using 3d data from ECAT')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')
    
    if len(sys.argv) > 1:
        train_test_config = sys.argv[1] 
        if train_test_config == 'train':
            mode = TRAIN
        if train_test_config == 'test' :
            mode = TEST
    
    if mode == TRAIN:
        if len(sys.argv) > 2:
            log_dir = sys.argv[2] 
        else:
            current_time = datetime.datetime.now()
            time_str = '%s_%s_%s_%s_%s_%s' % (current_time.year, current_time.month, current_time.day, 
                                  current_time.hour, current_time.minute, current_time.second)

            log_dir = 'logs/run_' + time_str
            
        print('Train and output into directory ' + log_dir)
        os.makedirs(log_dir)
        logging.basicConfig(filename = log_dir + '/logs.log',level=logging.DEBUG)
        
        # Copy the current executed py file to log (To make sure we can replicate the experiment with the same code)
        shutil.copy(os.path.realpath(__file__), log_dir)
    
    if mode == TEST:
        if len(sys.argv) > 2:
            model_path = sys.argv[2] 
            print('Test using model ' + model_path)
        else:
            sys.exit("learning.py test model_path")

    
    # ========================================================================
    # ========================================================================
    # =============================READING INPUT =============================

    SIMPLE_SPLIT = 'simple_train_test.pkl'
    if os.path.isfile(SIMPLE_SPLIT) :
        # Load the file
        logging.info("Load file into training and testing data sets " + SIMPLE_SPLIT)
        with open(SIMPLE_SPLIT, 'rb') as f:
            t = pickle.load(f)
            train = t['train']
            test = t['test']

        data_length = 63
    else:
        logging.info("Read training and testing data sets from data directory ")
        read_project_data()
        print("data_length " + str(data_length))
        train, test = generate_data(project_data, Simple_Train_Test_Config(project_data))

        with open(SIMPLE_SPLIT, 'wb') as f:
            pickle.dump({'train': train,
                        'test': test}, 
                        f, pickle.HIGHEST_PROTOCOL)

        print_and_log('----Done saving training and testing data---')


    print_and_log('Train size ' + str(len(train)))
    print_and_log('Test size ' + str(len(test)))

    class SmallConfig(object):
        """Small config."""
        init_scale = 0.1
        learning_rate = 1     # Set this value higher without norm clipping
                                # might make the cost explodes
        max_grad_norm = 5       # The maximum permissible norm of the gradient
        num_layers = 1          # Number of LSTM layers
        num_steps = 20          # Divide the data into num_steps segment 
        hidden_size = 200       # the number of LSTM units
        max_epoch = 10          # The number of epochs trained with the initial learning rate
        max_max_epoch = 250     # Number of running epochs
        keep_prob = 0.6         # Drop out keep probability, = 1.0 no dropout
        lr_decay = 0.980         # Learning rate decay
        batch_size = 40         # We could actually still use batch_size for convenient
        n_input = data_length   # Number of float values for each frame
        label_classes = label_classes # Number of classes, for each output label
        hop_step = 5            # Hopping between two samples
        test_epoch = 20         # Test after these many epochs 

    config = SmallConfig()
    intermediate_config = SmallConfig()
    intermediate_config.keep_prob = 1
    eval_config = SmallConfig()
    eval_config.keep_prob = 1
    eval_config.batch_size = 1

    logging.info("Train Configuration")
    for attr in dir(config):
        # Not default properties
        if attr[:2] != '__':
            log_str = "%s = %s" % (attr, getattr(config, attr))
            logging.info(log_str)

    logging.info("Evaluation Configuration")
    for attr in dir(eval_config):
        # Not default properties
        if attr[:2] != '__':
            log_str = "%s = %s" % (attr, getattr(eval_config, attr))
            logging.info(log_str)


    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
              m = Recognizer(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
              m_intermediate_test = Recognizer(is_training=False, config=intermediate_config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
              mtest = Recognizer(is_training=False, config=eval_config)
    
        if mode == TRAIN:
            tf.initialize_all_variables().run()

            #     print('Go through train data')
            #     for step, (x, y) in enumerate( gothrough(train, 63, 10, 19, 5) ):
            #         pass
            #     print('Go through test data')
            #     for step, (x, y) in enumerate( gothrough(test, 63, 10, 19, 5) ):
            #         pass

            random.seed()
            random.shuffle(train)

            print_and_log('---------------BASELINE-------------')

            test_perplexity = run_epoch(session, m_intermediate_test, test, m_intermediate_test.test_op, 
                                        is_training=False,
                                           verbose=False)
            print_and_log("Test Perplexity on Test: %s" % str(test_perplexity))


            print_and_log('----------------TRAIN---------------')  
            for i in range(config.max_max_epoch):
                print_and_log('-------------------------------')
                start_time = time.time()
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print_and_log("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

                train_perplexity = run_epoch(session, m, train, m.train_op,
                                           verbose=True)
                print_and_log("Epoch: %d Train Perplexity: %s" % (i + 1, str(train_perplexity)))
                print_and_log("Time %.3f" % (time.time() - start_time) )
                print_and_log('-------------------------------') 

                if i % config.test_epoch == 0:
                    print_and_log('----------Intermediate test -----------')  
                    # Run test on train
                    print_and_log('Run model on train data')
                    test_perplexity = run_epoch(session, m_intermediate_test, train, m_intermediate_test.test_op, 
                                                is_training=False, verbose = False)
                    print_and_log('Run model on test data')
                    test_perplexity = run_epoch(session, m_intermediate_test, test, m_intermediate_test.test_op, 
                                                is_training=False, verbose = False)

            # Save the variables to disk.
            model_path = m.saver.save(session, log_dir + "/model.ckpt")
            print_and_log("Model saved in file: %s" % model_path)
        
        if mode == TEST:
            m.saver.restore(session, model_path)
            print_and_log("Restore model saved in file: %s" % model_path)
            
        print_and_log('--------------TEST--------------')  
        # Run test on train
        print_and_log('Run model on train data')
        test_perplexity = run_epoch(session, mtest, train, mtest.test_op, 
                                    is_training=False, verbose=True)
        print_and_log('Run model on test data')
        test_perplexity = run_epoch(session, mtest, test, mtest.test_op, 
                                    is_training=False, verbose=True)
