'''
Created on Mar 4, 2017

@author: Tuan

Refactoring the code from learning5.py.
Main module, just need to call this
'''
from __future__ import division
from __future__ import print_function

import datetime
import logging
import os
import pickle
import random
import shutil
import sys
import time
import glob

import argparse
from sklearn.metrics.classification import confusion_matrix

from config import Simple_Train_Test_Config, ExplicitConfig, TreeConfig
from generate_utils import generate_data, turn_to_intermediate_data, gothrough, \
    check_validity_label
from lstm_crf_explicit import LSTM_CRF_Exp
from lstm_treecrf import LSTM_TREE_CRF
import numpy as np
from read_utils import read_project_data, read_pca_features, read_qsr_features
import tensorflow as tf
from utils import label_classes, num_labels, from_id_labels_to_str_labels, RAW, PCAS, QSR


# default mode is to train and test at the same time
TRAIN = 'TRAIN'
TEST = 'TEST'


def print_and_log(log_str):
    print (log_str)
    logging.info(log_str)
    
    
def run_epoch(session, m, data, lbl, info, eval_op, verbose=False, is_training=True, merged_summary = None, summary_writer = None):
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
        
        if merged_summary != None:
            summary, cost, state, eval_val = session.run([merged_summary, m.cost, m.final_state, eval_op], feed_dict)
            
            if summary_writer != None:
                summary_writer.add_summary(summary, step)
        else:
            debug, cost, state, eval_val = session.run([m.debug, m.cost, m.final_state, eval_op], feed_dict)
            
#             print(debug)
        
        if not is_training:
#             logits, A_start_t, A_to, A_ts, A_tp, A_se = eval_val
#             eval_val = m.calculate_best(y, logits, A_start_t, A_to, A_ts, A_tp, A_se)
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

                # if verbose and not np.all(np.equal(y_pred_array[:,i], y[i])):
                #     logging.info('-------' + z[i] + '--------')
                #     logging.info(from_id_labels_to_str_labels(*y_pred_array[:,i]))
                #     logging.info(from_id_labels_to_str_labels(*y[i]))
                # if verbose:
                #     logging.info('-------' + z[i] + '--------')
                #     logging.info(from_id_labels_to_str_labels(*y_pred_array[:,i]))
                #     logging.info(from_id_labels_to_str_labels(*y[i]))
                
            epoch_confusion_matrixs = [confusion_matrix(y[:,i], y_pred[i], range(len(label_classes[i].values())) ) 
                                       for i in xrange(len(m.label_classes))]
        
            for i in xrange(len(m.label_classes)):
                confusion_matrixs[i] += epoch_confusion_matrixs[i]
        
        if verbose and step % 30 == 0:
            print_and_log("cost %.3f, costs %.3f, iters %d, Step %d, perplexity: %.3f" % 
              (cost, costs, cost_iters, step, np.exp(costs / cost_iters)))
            
    if not is_training:
        print (eval_iters)
        print (m.batch_size)
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
    parser = argparse.ArgumentParser(description='A script to train and test using LSTM-CRF')
    
    parser.add_argument('-t', '--test', action = 'store_true', 
                                help = "If set, it is TEST, by default it is TRAIN" )
    
    parser.add_argument('-m', '--model',  action='store',
                                help = "Where to save the model or to load the model. By default, it is saved to the log dir" )
    
    parser.add_argument('-r', '--tree',  action='store_true',
                                help = "Whether to use the general tree LSTM-CRF model. By default, the explicit version is used." )

    parser.add_argument('-f', '--feature',  action='store',
                                help = "Choose which feature to extract. Pick between RAW, PCAS and QSR. Default is RAW." )

    parser.add_argument('-o', '--others',  nargs='+',
                                help = "Other options to be put into configuration. Give a list of key and value. \
                                Possible configuration values are [learning_rate, num_layers, hidden_size, keep_prob, lr_decay, test_crf_weight, test_batch_size]" )

    parser.add_argument('-k', '--kfold',  action='store',
                                help = "Number of fold in k-fold cross validation" )


    args = parser.parse_args()
    
    mode = TRAIN
    if args.test:
        mode = TEST
        
    model_path = args.model
    
    use_tree = args.tree

    feature_type = args.feature

    others = args.others

    kfold = args.kfold
    
    if mode == TRAIN:
        current_time = datetime.datetime.now()
        time_str = '%s_%s_%s_%s_%s_%s' % (current_time.year, current_time.month, current_time.day, 
                              current_time.hour, current_time.minute, current_time.second)

        log_dir = os.path.join('logs', 'run_' + time_str)
            
        print('Train and output into directory ' + log_dir)
        os.makedirs(log_dir)

        copy_code_dir = os.path.join( log_dir, 'code') 
        os.makedirs(copy_code_dir)

        logging.basicConfig(filename = os.path.join(log_dir, 'logs.log'),level=logging.DEBUG)
        
        # Copy the current executed py file to log (To make sure we can replicate the experiment with the same code)
        # shutil.copy(os.path.realpath(__file__), log_dir)

        code_dir = os.path.dirname(os.path.realpath(__file__))

        for f in glob.glob(os.path.join(code_dir, '*.py')):
            # Copy the current executed py file to log (To make sure we can replicate the experiment with the same code)
            shutil.copy(f, copy_code_dir)
        
        if not model_path:
            model_path = os.path.join(log_dir, "model.ckpt")
    
    if mode == TEST:
        current_time = datetime.datetime.now()
        time_str = '%s_%s_%s_%s_%s_%s' % (current_time.year, current_time.month, current_time.day, 
                              current_time.hour, current_time.minute, current_time.second)

        log_dir = os.path.join('logs', 'test_' + time_str)

        print('Test into directory ' + log_dir)
        os.makedirs(log_dir)

        logging.basicConfig(filename = os.path.join(log_dir, 'logs.log'),level=logging.DEBUG)
        if model_path:
            print('Test using model ' + model_path)
        else:
            sys.exit("learning.py -t -m model_path")

    if not feature_type:
        feature_type = RAW
    else:
        feature_type = feature_type.lower() 

    if feature_type not in [RAW, PCAS, QSR]:
        sys.exit("Feature type need to be in the set (raw, pcas, qsr)")

    
    # ========================================================================
    # ========================================================================
    # =============================READING INPUT =============================

    # if kfold:
    #     RAW_K_FOLD = 'fold_%d.pkl'  % kfold
    #     PCAS_K_FOLD = 'fold_pca_%d.pkl' % kfold
    #     QSR_K_FOLD = 'fold_qsr_%d.pkl'  % kfold
    # else:
        
    RAW_SPLIT = 'train_test_split.pkl'
    PCAS_SPLIT = 'train_test_split_pcas.pkl'
    QSR_SPLIT = 'train_test_split_qsr.pkl'
    
    SPLIT = None
    read_method = None

    if feature_type == RAW:
        SPLIT = RAW_SPLIT
        read_method = read_project_data
        data_length = 63
    elif feature_type == PCAS:
        SPLIT = PCAS_SPLIT
        read_method = read_pca_features
        data_length = 18
    elif feature_type == QSR:
        SPLIT = QSR_SPLIT
        read_method = read_qsr_features
        data_length = 25

    if os.path.isfile(SPLIT) :
        # Load the file
        logging.info("Load file into training and testing data sets " + SPLIT)
        with open(SPLIT, 'rb') as f:
            t = pickle.load(f)
            train = t['train']
            test = t['test']
    else:
        logging.info("Read training and testing data sets from data directory ")
        data_length, project_data = read_method()
        print("data_length " + str(data_length))
        train, test = generate_data(project_data, Simple_Train_Test_Config(), feature_type)

        with open(SPLIT, 'wb') as f:
            pickle.dump({'train': train,
                        'test': test}, 
                        f, pickle.HIGHEST_PROTOCOL)

        print_and_log('----Done saving training and testing data---')


    print_and_log('Train size ' + str(len(train)))
    print_and_log('Test size ' + str(len(test)))

    if use_tree:
        config = TreeConfig(data_length, label_classes)
        intermediate_config = TreeConfig(data_length, label_classes)
        eval_config = TreeConfig(data_length, label_classes)
    else:
        config = ExplicitConfig(data_length, label_classes)
        intermediate_config = ExplicitConfig(data_length, label_classes)
        eval_config = ExplicitConfig(data_length, label_classes)
    
    test_crf_weight = 0.5
    test_batch_size = 10

    if others:
        for i in xrange(len(others) // 2):
            key = others[2 * i]
            if key not in ['learning_rate', 'num_layers', 'hidden_size', 'keep_prob', 'lr_decay', 'test_crf_weight', 'test_batch_size']:
                sys.exit("Other configuration value must be in [learning_rate, num_layers, hidden_size, keep_prob, lr_decay, max_max_epoch, test_crf_weight, test_batch_size]") 

            value = None
            if key in ['learning_rate', 'keep_prob', 'lr_decay'] :
                value = float(others[2 * i + 1])
            elif key in ['num_layers', 'hidden_size', 'max_max_epoch']:
                value = int(others[2 * i + 1])

            if value:
                setattr(config, key, value)
                setattr(intermediate_config, key, value)
                setattr(eval_config, key, value)

            if key == 'test_crf_weight':
                test_crf_weight = float(others[2 * i + 1])

            if key == 'test_batch_size':
                test_batch_size = int(others[2 * i + 1])

    '''
    No dropout
    '''
    intermediate_config.keep_prob = 1

    '''
    No droupout
    Single input
    Decrease CRF weight 
    '''
    eval_config.keep_prob = 1
    eval_config.batch_size = test_batch_size
    eval_config.crf_weight = test_crf_weight
    
    
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
        
        if use_tree:
            print('-------- Setup m model ---------')
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                config.tree.initiate_crf()
                m = LSTM_TREE_CRF(is_training=True, config=config)
                
            print('-------- Setup m_intermediate_test model ---------')
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                intermediate_config.tree.initiate_crf()
                m_intermediate_test = LSTM_TREE_CRF(is_training=False, config=intermediate_config)
                
            print('-------- Setup mtest model ----------')
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                eval_config.tree.initiate_crf()
                mtest = LSTM_TREE_CRF(is_training=False, config=eval_config)
        else:
            print('-------- Setup m model ---------')
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = LSTM_CRF_Exp(is_training=True, config=config)
                
            print('-------- Setup m_intermediate_test model ---------')
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                m_intermediate_test = LSTM_CRF_Exp(is_training=False, config=intermediate_config)
                
            print('-------- Setup mtest model ----------')
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mtest = LSTM_CRF_Exp(is_training=False, config=eval_config)
        
        if mode == TRAIN:
            tf.global_variables_initializer().run()

            random.seed()
            random.shuffle(train)

            print_and_log('----------------TRAIN---------------')
            
#             merged_summary = tf.summary.merge_all()
#             train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), session.graph)
             
            for i in range(config.max_max_epoch):
                try:
                    print_and_log('-------------------------------')
                    start_time = time.time()
                    lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                    m.assign_lr(session, config.learning_rate * lr_decay)

                    print_and_log("Epoch: %d Learning rate: %.6f" % (i + 1, session.run(m.lr)))

#                     train_perplexity = run_epoch(session, m, im_train_data, 
#                                                  im_train_lbl, im_train_inf,
#                                                  m.train_op,
#                                                verbose=True, merged_summary = merged_summary, summary_writer = train_writer)
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
                        _model_path = m.saver.save(session, model_path)
                        print_and_log("Model saved in file: %s" % _model_path)
                        print_and_log("Time %.3f" % (time.time() - start_time) )
                except ValueError, e: 
                    print_and_log(e)
                    print_and_log("Value error, reload the most recent saved model")
                    # m.saver.restore(session, model_path)
                    break
            
#             train_writer.close()
#             print_and_log("Train writer is closed")
            
            _model_path = m.saver.save(session, model_path)
            print_and_log("Model saved in file: %s" % _model_path)
        
        if mode == TEST:
            m.saver.restore(session, model_path)
            print_and_log("Restore model saved in file: %s" % model_path)
        
#         test_writer = tf.summary.FileWriter(os.path.join(log_dir, 'test'))
        
        print_and_log('--------------TEST--------------')  
        print_and_log('Run model on test data')
        test_perplexity = run_epoch(session, mtest, im_final_test_data, 
                                    im_final_test_lbl, im_final_test_inf,
                                    mtest.test_op, 
                                    is_training=False, verbose=True)
#         test_perplexity = run_epoch(session, mtest, im_final_test_data, 
#                                     im_final_test_lbl, im_final_test_inf,
#                                     mtest.test_op, 
#                                     is_training=False, verbose=True, merged_summary= merged_summary, summary_writer = test_writer)
        
#         test_writer.close()
        print_and_log("Test writer is closed")
