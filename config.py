'''
Created on Mar 4, 2017

@author: Tuan
'''

'''
Train on a subset of sessions for each project
Training_percentages = Percentage of training sessions/ Total # of sessions
'''
class Simple_Train_Test_Config(object):
    # Using all projects for training
    train_project_names = ['pullacross', 'pullfrom', 'pushfrom', 'pushto',
                        'rollacross', 'rollto', 'selfrollacross', 'selfrollto',
                          'selfslidefrom','pullto', 'pushacross', 
                          'rollfrom', 'selfrollfrom',
                          'selfslideacross', 'selfslideto']
    test_project_names = ['pullacross', 'pullfrom', 'pushfrom', 'pushto',
                        'rollacross', 'rollto', 'selfrollacross', 'selfrollto',
                          'selfslidefrom','pullto', 'pushacross', 
                          'rollfrom', 'selfrollfrom',
                          'selfslideacross', 'selfslideto']
    session_training_percentage = (0, 0.6)
    session_testing_percentage = (0.6, 1)
    double_training = True


'''Only train on a subset of projects
For each training project, train on all sessions
Training_percentages = 1
'''
class Partial_Train_Test_Config(object):
    # Using a subset of projects for training
    train_project_names = ['pullacross', 'pullfrom', 'pushfrom', 'pushto',
                          'selfslidefrom']
    test_project_names = ['pullto', 'pushacross', 
                          'selfslideacross', 'selfslideto']
    session_training_percentage = (0, 1)
    session_testing_percentage = (0, 1)
    double_training = False
    
    
class ModelConfig(object):
    init_scale = 0.1
    learning_rate = 0.5     # Set this value higher without norm clipping
                            # might make the cost explodes
    max_grad_norm = 5       # The maximum permissible norm of the gradient
    num_layers = 1          # Number of LSTM layers
    num_steps = 20          # Divide the data into num_steps segment 
    hidden_size = 200       # the number of LSTM units
    max_epoch = 10          # The number of epochs trained with the initial learning rate
    max_max_epoch = 250     # Number of running epochs
    keep_prob = 0.8         # Drop out keep probability, = 1.0 no dropout
    lr_decay = 0.980         # Learning rate decay
    batch_size = 40         # We could actually still use batch_size for convenient
    hop_step = 5            # Hopping between two samples
    test_epoch = 20         # Test after these many epochs
    save_epoch = 10
    crf_weight = 1
    
    def __init__(self, data_length, label_classes):
        self.n_input = data_length  # Number of float values for each frame
        self.label_classes = label_classes # Number of classes, for each output label