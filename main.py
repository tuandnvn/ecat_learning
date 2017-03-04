'''
Created on Mar 4, 2017

@author: Tuan

Refactoring the code from learning5.py.
Main module, just need to call this
'''
import datetime
import logging
import os
import pickle
import random
import shutil
import sys
import time

from sklearn.metrics.classification import confusion_matrix

from config import Simple_Train_Test_Config, ModelConfig
from generate_utils import generate_data, turn_to_intermediate_data, gothrough, \
    check_validity_label
from lstm_crf import LSTM_CRF
import numpy as np
from read_utils import read_project_data
import tensorflow as tf
from utils import label_classes, num_labels, from_id_labels_to_str_labels


# default mode is to train and test at the same time
TRAIN = 'TRAIN'
TEST = 'TEST'
mode = TRAIN

def print_and_log(log_str):
    print (log_str)
    logging.info(log_str)
    
    
def run_epoch(session, m, data, lbl, info, eval_op, verbose=False, is_training=True):
    """Runs the model on the given data."""
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
    
    if verbose:
        print_and_log('------PRINT OUT PREDICTED AND CORRECT LABELS--------')
            
    for step, (x, y, z) in enumerate( gothrough(data, lbl, info) ):
        feed_dict = {}
        feed_dict[m.input_data] = x
        feed_dict[m.targets] = y
        for i in xrange(len(m.initial_state)):
            feed_dict[m.initial_state[i]] = state[i]
    
        debug_val, cost, state, eval_val = session.run([m.debug, m.cost, m.final_state, eval_op], feed_dict)
        
        if np.isnan(cost) or not np.isfinite(cost):
            print_and_log("------------------------DEBUG-----------------------")
            print_and_log("logit_s")
            print_and_log(repr(debug_val[0]))
            print_and_log("logit_o")
            print_and_log(repr(debug_val[1]))
            print_and_log("logit_t")
            print_and_log(repr(debug_val[2]))
            print_and_log("logit_e")
            print_and_log(repr(debug_val[3]))
            print_and_log("logit_p")
            print_and_log(repr(debug_val[4]))
            
            print_and_log("all_log_start_t")
            print_and_log(repr(debug_val[5]))
            print_and_log("all_log_t_o")
            print_and_log(repr(debug_val[6]))
            print_and_log("all_log_t_p")
            print_and_log(repr(debug_val[7]))
            print_and_log("all_log_s")
            print_and_log(repr(debug_val[8]))
            print_and_log("all_log_s_t")
            print_and_log(repr(debug_val[9]))
            print_and_log("all_log_s_e")
            print_and_log(repr(debug_val[10]))
            raise ValueError('A nan value happens')
    
        
        if not is_training:
            logits, A_start_t, A_to, A_ts, A_tp, A_se = eval_val
            eval_val = m.calculate_best(y, logits, A_start_t, A_to, A_ts, A_tp, A_se)
            # Unpack the predictions and cost values
            y_pred, eval_val = eval_val
            
        costs += cost
        cost_iters += 1
        eval_iters += 1
        
        if not is_training:
            evals += eval_val
            
            correct_pred = np.sum(np.all([np.equal(y_pred[i], y[:,i]) \
                                for i in xrange(len(m.label_classes))], axis = 0))
            total_correct_pred += correct_pred
            
            y_pred_array = np.array(y_pred)
            
                
            for i in xrange(m.batch_size):
                valid = check_validity_label( y_pred_array[:,i] )
                valid_labels[valid] += 1

                if verbose and not np.all(np.equal(y_pred_array[:,i], y[i])):
                    logging.info('-------' + z[i] + '--------')
                    logging.info(from_id_labels_to_str_labels(*y_pred_array[:,i]))
                    logging.info(from_id_labels_to_str_labels(*y[i]))
                
            epoch_confusion_matrixs = [confusion_matrix(y[:,i], y_pred[i], range(len(label_classes[i].values())) ) 
                                       for i in xrange(len(m.label_classes))]
        
            for i in xrange(len(m.label_classes)):
                confusion_matrixs[i] += epoch_confusion_matrixs[i]
        
        if verbose and step % 30 == 0:
            print_and_log("cost %.3f, costs %.3f, iters %d, Step %d, perplexity: %.3f" % 
              (cost, costs, cost_iters, step, np.exp(costs / cost_iters)))
            
    if not is_training:
        print_and_log("Number of valid/Number of invalid = %d/%d" % 
                      (valid_labels[True], valid_labels[False]))
        
        print_and_log("Number of correct predictions = %d, Percentage = %.3f" % 
                      (total_correct_pred, total_correct_pred/ (eval_iters * m.batch_size) ))
        
        print_and_log("Subject accuracy = %.5f" % (evals[0] / eval_iters))
        if verbose:
            print_and_log("-- Confusion matrix --")
            print_and_log(confusion_matrixs[0])
        
        print_and_log("Object accuracy = %.5f" % (evals[1] / eval_iters))
        if verbose:
            print_and_log("-- Confusion matrix --")
            print_and_log(confusion_matrixs[1])
        
        print_and_log("Theme accuracy = %.5f" % (evals[2] / eval_iters))
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
            sys.exit("learning.py test model_path")

    
    # ========================================================================
    # ========================================================================
    # =============================READING INPUT =============================

    SIMPLE_SPLIT = 'train_test_split.pkl'
    
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
        data_length, project_data = read_project_data()
        print("data_length " + str(data_length))
        train, test = generate_data(project_data, Simple_Train_Test_Config())

        with open(SIMPLE_SPLIT, 'wb') as f:
            pickle.dump({'train': train,
                        'test': test}, 
                        f, pickle.HIGHEST_PROTOCOL)

        print_and_log('----Done saving training and testing data---')


    print_and_log('Train size ' + str(len(train)))
    print_and_log('Test size ' + str(len(test)))

    
    config = ModelConfig(data_length, label_classes)
    
    '''
    No dropout
    '''
    intermediate_config = ModelConfig(data_length, label_classes)
    intermediate_config.keep_prob = 1
    
    '''
    No droupout
    Single input
    Decrease CRF weight 
    '''
    eval_config = ModelConfig(data_length, label_classes)
    eval_config.keep_prob = 1
    eval_config.batch_size = 1
    eval_config.crf_weight = 0.5
    
    print('Turn train data to intermediate form')
    im_train_data, im_train_lbl, im_train_inf = turn_to_intermediate_data(train, config.n_input, config.batch_size, 
                                                            config.num_steps, config.hop_step)
    
    print('Turn test data to intermediate form')
    im_inter_test_data, im_inter_test_lbl, im_inter_test_inf = turn_to_intermediate_data(test, intermediate_config.n_input, 
                                        intermediate_config.batch_size, 
                                        intermediate_config.num_steps, 
                                        intermediate_config.hop_step, )
    im_final_test_data, im_final_test_lbl, im_final_test_inf = turn_to_intermediate_data(test, eval_config.n_input, 
                                        eval_config.batch_size, 
                                        eval_config.num_steps, 
                                        eval_config.hop_step)
            
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
        print('-------- Setup m model ---------')
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = LSTM_CRF(is_training=True, config=config)
        print('-------- Setup m_intermediate_test model ---------')
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            m_intermediate_test = LSTM_CRF(is_training=False, config=intermediate_config)
        print('-------- Setup mtest model ----------')
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = LSTM_CRF(is_training=False, config=eval_config)
    
        if mode == TRAIN:
            tf.initialize_all_variables().run()

            random.seed()
            random.shuffle(train)

            print_and_log('---------------BASELINE-------------')

            test_perplexity = run_epoch(session, m_intermediate_test, im_inter_test_data, 
                                        im_inter_test_lbl, im_inter_test_inf,
                                        m_intermediate_test.test_op, 
                                        is_training=False,
                                           verbose=False)
            print_and_log("Test Perplexity on Test: %s" % str(test_perplexity))


            print_and_log('----------------TRAIN---------------')  
            for i in range(config.max_max_epoch):
                try:
                    print_and_log('-------------------------------')
                    start_time = time.time()
                    lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                    m.assign_lr(session, config.learning_rate * lr_decay)

                    print_and_log("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

                    train_perplexity = run_epoch(session, m, im_train_data, 
                                                 im_train_lbl, im_train_inf,
                                                 m.train_op,
                                               verbose=True)
                    print_and_log("Epoch: %d Train Perplexity: %s" % (i + 1, str(train_perplexity)))
                    print_and_log("Time %.3f" % (time.time() - start_time) )
                    print_and_log('-------------------------------') 

                    if i % config.test_epoch == 0:
                        print_and_log('----------Intermediate test -----------')  
                        # Run test on train
                        print_and_log('Run model on train data')
                        test_perplexity = run_epoch(session, m_intermediate_test, im_train_data, 
                                                    im_train_lbl, im_train_inf,
                                                    m_intermediate_test.test_op, 
                                                    is_training=False, verbose = False)
                        print_and_log('Run model on test data')
                        test_perplexity = run_epoch(session, m_intermediate_test, im_inter_test_data, 
                                                    im_inter_test_lbl, im_inter_test_inf,
                                                    m_intermediate_test.test_op, 
                                                    is_training=False, verbose = True)
                    
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
        test_perplexity = run_epoch(session, mtest, im_final_test_data, 
                                    im_final_test_lbl, im_final_test_inf,
                                    mtest.test_op, 
                                    is_training=False, verbose=True)