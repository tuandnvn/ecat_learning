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

# Data to train auto-encoder
rig_data = []
glyph_data = []

def read_project_data():
    for file_name in glob.glob('data/*.txt'):
        project_name = file_name[file_name.rfind('/') + 1:]
        project_name = project_name[:len(project_name)-4]

        print ("File name = %s, project name = %s " % (file_name, project_name))
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

                rig_data.append(point_data[:39])
                glyph_data.append(point_data[39:51])
                glyph_data.append(point_data[51:63])

def turn_to_intermediate_data(data, data_point_size, batch_size):
    rearranged_data = np.array(data)
    epoch_size = len(data) // batch_size
    
    return rearranged_data[:epoch_size * batch_size].\
            reshape((epoch_size, batch_size, data_point_size))
    
class Auto_Encoder_Config(object):
    session_training_percentage = (0, 0.8)
    session_testing_percentage = (0.8, 1.0)
    
class Autoencoder(object):
    def __init__(self, is_training, config):
        with tf.device('/gpu:2'):
            # multiple batch sizes
            self.batch_sizes = batch_sizes = config.batch_sizes
            # multiple input sizes
            self.n_inputs = n_inputs = config.n_inputs
            hidden_size_1 = config.hidden_size_layer_1
            hidden_size_2 = config.hidden_size_layer_2
            
            self._input_datas = input_datas = [tf.placeholder(tf.float32, [batch_size, n_input]) 
                                               for (batch_size, n_input) in zip(batch_sizes, n_inputs)]
            
            weights = {}
            biases = {}
            
            with tf.variable_scope("encoder"):
                for i, n_input in enumerate(n_inputs):
                    weights['encoder_h1_' + str(i)] = tf.get_variable('weight_encoder_h1_' + str(i), [n_input, hidden_size_1])
                    biases['encoder_h1_' + str(i)] = tf.get_variable('bias_encoder_h1_' + str(i), [hidden_size_1])


                    # # Shared between different input types
                    # weights['encoder_h2_' + str(i)] = tf.get_variable('weight_encoder_h2_' + str(i), [hidden_size_1, hidden_size_2])
                    # biases['encoder_h2_' + str(i)] = tf.get_variable('bias_encoder_h2_' + str(i), [hidden_size_2])

                weights['encoder_h2'] = tf.get_variable('weight_encoder_h2', [hidden_size_1, hidden_size_2])
                biases['encoder_h2'] = tf.get_variable('bias_encoder_h2', [hidden_size_2])
                
            with tf.variable_scope("decoder"):
                # Shared between different input types
                weights['decoder_b1' ] = tf.get_variable('weight_decoder_b1', [hidden_size_2, hidden_size_1])
                biases['decoder_b1' ] = tf.get_variable('bias_decoder_b1', [hidden_size_1])
                
                for i, n_input in enumerate(n_inputs):
                    # weights['decoder_b1_' + str(i)] = tf.get_variable('weight_decoder_b1' + str(i), [hidden_size_2, hidden_size_1])
                    # biases['decoder_b1_' + str(i)] = tf.get_variable('bias_decoder_b1_' + str(i), [hidden_size_1])

                    weights['decoder_b2_' + str(i)] = tf.get_variable("weight_decoder_b2_" + str(i), [hidden_size_1, n_input])
                    biases['decoder_b2_' + str(i)] = tf.get_variable("bias_decoder_b2_" + str(i), [n_input])
                    
            
            error_squared = []
            decodeds = []
            for i in xrange(len(n_inputs)):
                encoder_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_datas[i], weights['encoder_h1_'+ str(i)]),
                                       biases['encoder_h1_'+ str(i)]))
                if is_training and config.keep_prob < 1:
                    encoder_layer_1 = tf.nn.dropout(encoder_layer_1, config.keep_prob)

                # encoder_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_1, weights['encoder_h2_' + str(i)]),
                #                        biases['encoder_h2_' + str(i)]))
                
                encoder_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_1, weights['encoder_h2']),
                                       biases['encoder_h2']))

                if is_training and config.keep_prob < 1:
                    encoder_layer_2 = tf.nn.dropout(encoder_layer_2, config.keep_prob)

                # decoder_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_2, weights['decoder_b1_' + str(i)]),
                #                        biases['decoder_b1_' + str(i)]))
                decoder_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_2, weights['decoder_b1' ]),
                                       biases['decoder_b1']))

                # The output layer should be linear to predict real values
                decoder_layer_2 = tf.add(tf.matmul(decoder_layer_1, weights['decoder_b2_' + str(i)]),
                                       biases['decoder_b2_' + str(i)])
                
                # size = batch_sizes[i]
                error_squared.append(tf.reduce_mean(tf.pow(input_datas[i] - decoder_layer_2, 2)))
                decodeds.append(decoder_layer_2)
                
            # error_squared = tf.concat(0, error_squared)
            self._cost = cost = tf.reduce_mean(error_squared)
            
            if is_training:
                self._lr = tf.Variable(0.0, trainable=False)
                self._train_op = tf.train.RMSPropOptimizer(self._lr).minimize(cost)

                # tvars = tf.trainable_variables()
                # grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
                # self._train_op = tf.train.GradientDescentOptimizer(self.lr).apply_gradients(zip(grads, tvars))
            else:
                self._test_op = decodeds
                
        self._saver = saver = tf.train.Saver()
                
    @property
    def debug(self):
        return self._debug
    
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))
    
    @property
    def saver(self):
        return self._saver
    
    @property
    def input_datas(self):
        return self._input_datas

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
    
    @property
    def test_op(self):
        return self._test_op

'''Generate a training set and a testing set of data'''
def generate_data(datas, config) :
    training_data = []
    testing_data = []
    
    expectations = []
    for data in datas:
        expectation = np.mean(data, 0)
        expectations.append(expectation)
        data -= expectation

        # Flatten the data (collapse the project and session hierarchy into a list of session_data)
        shuffled_data = random.sample(data, len(data))


        training_data.append(shuffled_data[int(config.session_training_percentage[0] * len(shuffled_data)):
                                     int(config.session_training_percentage[1] * len(shuffled_data))])

        testing_data.append(shuffled_data[int(config.session_testing_percentage[0] * len(shuffled_data)):
                                     int(config.session_testing_percentage[1] * len(shuffled_data))])
    
    return (training_data, testing_data, expectations)

'''
Parameters:
rearranged_data: (epoch_size, batch_size, data_point_size)
Yields:
Take batch_size of data samples, each is a chain of num_steps data points
x: [batch_size, data_point_size]
'''
def gothrough(rearranged_data):
    for i in range(np.shape(rearranged_data)[0]):
        x = rearranged_data[i, :, :]
        yield x
        
def run_epoch(session, m, datas, eval_op, verbose=False, is_training=True):
    """Runs the model on the given data."""
    start_time = time.time()
    # costs = np.zeros(len(m.label_classes))
    costs = 0
    
    for step, x in enumerate( zip(*[gothrough(data) for data in datas]) ):
        feed_dict = {}
        for i in xrange(len(m.input_datas)):
            feed_dict[m.input_datas[i]] = x[i]
    
        cost, eval_val = session.run([m.cost, eval_op], feed_dict)
            
        costs += cost
        
        if verbose and not is_training:
            print('True values')
            print(x)
            print('Predicted values')
            print(eval_val)
        
        if step % 5 == 0 and step > 0:
            print_and_log("cost %.5f, costs %.5f, Step %d, average cost: %.5f" % 
              (cost, costs, step, costs / (step + 1)))
            
    if not is_training:
        print_and_log("End of epoch: costs %.5f, Step %d, average cost: %.5f" % 
              (costs, step, costs / (step + 1) ))
        
# default mode is to train and test at the same time
TRAIN = 'TRAIN'
TEST = 'TEST'
mode = TRAIN

if __name__ == '__main__':
    # ========================================================================
    # ========================================================================
    # ===========================SETUP TRAIN TEST=============================
    
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
            sys.exit("autoencoder.py test model_path")
            
    # ========================================================================
    # ========================================================================
    # =============================READING INPUT =============================

    SIMPLE_SPLIT = 'autoencoder_train_test.pkl'
    
    if os.path.isfile(SIMPLE_SPLIT) :
        # Load the file
        logging.info("Load file into training and testing data sets " + SIMPLE_SPLIT)
        with open(SIMPLE_SPLIT, 'rb') as f:
            t = pickle.load(f)
            train = t['train']
            test = t['test']
            expectation = t['expectation']
    else:
        logging.info("Read training and testing data sets from data directory ")
        read_project_data()
        train, test, expectation = generate_data([rig_data, glyph_data], Auto_Encoder_Config)

        with open(SIMPLE_SPLIT, 'wb') as f:
            pickle.dump({'train': train,
                        'test': test,
                        'expectation': expectation}, 
                        f, pickle.HIGHEST_PROTOCOL)

        print_and_log('----Done saving training and testing data---')
        
    print_and_log('Train size ' + str(len(train)))
    for i in xrange(len(train)):
        print_and_log('Type ' + str(i) + " = " + str(len(train[i])))
    print(train[0][5])
    print_and_log('Test size ' + str(len(test)))
    for i in xrange(len(test)):
        print_and_log('Type ' + str(i) + " = " + str(len(test[i])))
    print(test[0][5])

    print_and_log('expectation')
    print_and_log(expectation)
    
    class SmallConfig(object):
        """Small config."""
        init_scale = 0.5
        learning_rate = 0.02     # Set this value higher without norm clipping
                                # might make the cost explodes
        hidden_size_layer_1 = 50            # the number of LSTM units
        hidden_size_layer_2 = 200       # the number of LSTM units
        max_epoch = 10          # The number of epochs trained with the initial learning rate
        max_max_epoch = 500     # Number of running epochs
        keep_prob = 0.8        # Drop out keep probability, = 1.0 no dropout
        lr_decay = 0.990         # Learning rate decay
        batch_sizes = [200, 400]         # We could actually still use batch_size for convenient
        n_inputs = [39, 12]   # Number of float values for each frame
        test_epoch = 50         # Test after these many epochs
        save_epoch = 25
        max_grad_norm = 5
    
    config = SmallConfig()
    intermediate_config = SmallConfig()
    intermediate_config.keep_prob = 1
    eval_config = SmallConfig()
    eval_config.keep_prob = 1
    eval_config.batch_sizes = [200, 400]
    eval_config_2 = SmallConfig()
    eval_config_2.keep_prob = 1
    eval_config_2.batch_sizes = [1, 1]
    
    print('Turn train data to intermediate form')
    im_train_data = [turn_to_intermediate_data(train_data, config.n_inputs[i], config.batch_sizes[i])
                     for i, train_data in enumerate(train)]
    
    print('Turn test data to intermediate form')
    im_inter_test_data = [turn_to_intermediate_data(test_data, intermediate_config.n_inputs[i], 
                            intermediate_config.batch_sizes[i]) 
                        for i, test_data in enumerate(test)]
    im_final_test_data = [turn_to_intermediate_data(test_data, eval_config.n_inputs[i],
                            eval_config.batch_sizes[i]) 
                          for i, test_data in enumerate(test)]
    
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
            
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True)) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        print('-------- Setup m model ---------')
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Autoencoder(is_training=True, config=config)
        print('-------- Setup m_intermediate_test model ---------')
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            m_intermediate_test = Autoencoder(is_training=False, config=intermediate_config)
        print('-------- Setup mtest model ----------')
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = Autoencoder(is_training=False, config=eval_config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest_test = Autoencoder(is_training=False, config=eval_config_2)  
                
        if mode == TRAIN:
            tf.initialize_all_variables().run()

            print_and_log('---------------BASELINE-------------')

            run_epoch(session, m_intermediate_test, im_inter_test_data, 
                                        m_intermediate_test.test_op, 
                                        is_training=False,
                                           verbose=False)


            print_and_log('----------------TRAIN---------------')  
            for i in range(config.max_max_epoch):
                try:
                    print_and_log('-------------------------------')
                    start_time = time.time()
                    lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                    m.assign_lr(session, config.learning_rate * lr_decay)

                    print_and_log("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

                    run_epoch(session, m, im_train_data, m.train_op,
                                               verbose=False)
                    print_and_log("Time %.3f" % (time.time() - start_time) )
                    print_and_log('-------------------------------') 

                    if i % config.test_epoch == 0:
                        print_and_log('----------Intermediate test -----------')  
                        # Run test on train
                        print_and_log('Run model on train data')
                        run_epoch(session, m_intermediate_test, im_train_data, m_intermediate_test.test_op, 
                                                    is_training=False, verbose = False)
                        print_and_log('Run model on test data')
                        run_epoch(session, m_intermediate_test, im_inter_test_data, m_intermediate_test.test_op, 
                                                    is_training=False, verbose = False)
                    
                    if i % config.save_epoch == 0:
                        start_time = time.time()
                        model_path = m.saver.save(session, log_dir + "/model.ckpt")
                        print_and_log("Model saved in file: %s" % model_path)
                        print_and_log("Time %.3f" % (time.time() - start_time) )
                except ValueError:
                    print_and_log("Value error, reload the most recent saved model")
                    m.saver.restore(session, model_path)
                    break
            
            model_path = m.saver.save(session, log_dir + "/model.ckpt")
            print_and_log("Model saved in file: %s" % model_path)
        
        if mode == TEST:
            m.saver.restore(session, model_path)
            print_and_log("Restore model saved in file: %s" % model_path)
            
        print_and_log('--------------TEST--------------')  
        print_and_log('Run model on test data')
        run_epoch(session, mtest, im_final_test_data, mtest.test_op, 
                                    is_training=False, verbose=False)
        
        print_and_log('Run model on sample data')
        a = np.array([0.7345144,0.2071555,1.218503,0.6770352,0.2617454,1.340257,0.6781173,0.07657745,1.497948,0.637142,-0.06540541,1.2899,0.6534985,
             -0.1020509,1.250801,0.6258702,-0.138501,1.138003,0.6847188,-0.1192688,1.216028,0.7559613,0.1016199,1.152093,0.8359883,-0.0585327,1.101261,
             0.6747639,-0.1555298,1.110413,0.6371136,-0.1579534,1.140965,0.5651836,-0.1791936,1.140071,0.6521184,-0.1845737,1.168485])
        a -= expectation[0]
        b = np.array([-0.4483964,-0.2694907,1.222,-0.4602559,-0.1941137,1.227,-0.4279048,-0.2011898,1.445,-0.3881741,-0.2630816,1.323])
        b -= expectation[1]
        a = a.reshape((1,1,39))
        b = b.reshape((1,1,12))
        run_epoch(session, mtest_test, [a,b], mtest_test.test_op, 
                                    is_training=False, verbose=True)

        run_epoch(session, mtest_test,  [np.array(test[0][0]).reshape((1,1,39)), np.array(test[1][0]).reshape((1,1,12))], mtest_test.test_op, 
                                    is_training=False, verbose=True)