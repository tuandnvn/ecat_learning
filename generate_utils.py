'''
Created on Mar 4, 2017

@author: Tuan
'''

'''
Generate a training set and a testing set of data

Parameters:
-----------
project_data: dictionary from project names to a list of session data
config:  specify training vs testing percentage

Return:
-------
training_data: similar as project_data
testing_data: similar as project_data


'''

import random

import numpy as np
from utils import SESSION_NAME, SESSION_DATA, SESSION_EVENTS, num_labels


def generate_data(project_data, config) :
    training_data = []
    testing_data = []
    
    # Flatten the data (collapse the project and session hierarchy into a list of session_data)
    for v in config.train_project_names:
        # Data in all sessions of one project
        project_session_data = random.sample(project_data[v], len(project_data[v]))

        training_data += project_session_data[int(config.session_training_percentage[0] * len(project_session_data)):
                                     int(config.session_training_percentage[1] * len(project_session_data))]

        if config.double_training:
            for i in xrange(int(config.session_training_percentage[0] * len(project_session_data)),
                                     int(config.session_training_percentage[1] * len(project_session_data))):
                session_data = project_session_data[i]

                reversed_session_data = {}
                reversed_session_data[SESSION_NAME] = session_data[SESSION_NAME] + "_reversed"
                reversed_session_data[SESSION_DATA] = []
                reversed_session_data[SESSION_EVENTS] = []

                for point_data in session_data[SESSION_DATA]:
                    reversed_point_data = point_data[:39]
                    reversed_point_data += point_data[51:63]
                    reversed_point_data += point_data[39:51]

                    reversed_session_data[SESSION_DATA].append(reversed_point_data)

                for event_str in session_data[SESSION_EVENTS]:
                    reversed_event_str = {}
                    for key in event_str:
                        reversed_event_str[key] = event_str[key]

                    subj, obj, theme, event, prep = event_str['label']
                    def swap_objects(value):
                        if value == 2:
                            return 3
                        if value == 3:
                            return 2
                        return value

                    reversed_event_str['label'] = (swap_objects(subj), swap_objects(obj), swap_objects(theme), event, prep)

                    reversed_session_data[SESSION_EVENTS].append(reversed_event_str)

                training_data.append(reversed_session_data)


        testing_data += project_session_data[int(config.session_testing_percentage[0] * len(project_session_data)):
                                     int(config.session_testing_percentage[1] * len(project_session_data))]
    
    return (training_data, testing_data)

'''
Check to see whether it makes a valid tuple

Parameters:
-----------
labels:  A tuple of labels (Object_1, Object_2, Object_3, 


Return:
-------

'''
def check_validity_label(labels):
    # Event is None -> All other values are None
    if labels[3] == 0:
        for i in xrange(5):
            if labels[i] != 0:
                return False
        return True
    
    # If two roles have the same object return False
    for i in xrange(3):
        for j in xrange(3):
            if i != j and labels[i] == labels[j] and labels[i] != 0:
                return False
    
    # If there is a Theme, there needs to be a Preposition and vice versa
    if labels[2] != 0 and labels[4] == 0:
        return False
    
    if labels[2] == 0 and labels[4] != 0:
        return False
    
    return True

'''
A function to generate a pair of batch-data (x, y)

Parameters:
-----------
data: a list of session_data
data_point_size: Vector feature size (63)
num_steps: A fix number of steps for each event (this should be the original num_steps - 1
because data point difference is used instead of )
hop_step: A fix number of frame offset btw two events
num_labels: Number of labels in output (5)

Return:
-------
rearranged_data: (epoch_size, batch_size, num_steps, data_point_size)
rearranged_lbls: (epoch_size, batch_size, num_labels)
'''
def turn_to_intermediate_data(data, data_point_size, batch_size, num_steps, hop_step):
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
    # Use a string of maximum 16 characters to store some info about a data sample 
    interpolated_info = np.zeros([samples], dtype='|S16')
    
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
                
                interpolated_lbls[sample_counter + i] = list(event_labels)

                interpolated_info[sample_counter + i] = session_data[SESSION_NAME] + '_' + str(i)
            
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

    rearranged_info = interpolated_info[:epoch_size * batch_size].\
            reshape((epoch_size, batch_size))
    
    return (rearranged_data, rearranged_lbls, rearranged_info)
        
        
'''
Iterate through data in the rearranged format

Parameters:
-----------
rearranged_data: (epoch_size, batch_size, num_steps, data_point_size)
rearranged_lbls: (epoch_size, batch_size, num_labels)
rearranged_info: (epoch_size, batch_size)

Yields:
-------
Take batch_size of data samples, each is a chain of num_steps data points
x: [batch_size, num_steps, data_point_size]
y: [batch_size, num_labels]
z: [batch_size]
'''
def gothrough(rearranged_data, rearranged_lbls, rearranged_info):
    for i in range(np.shape(rearranged_data)[0]):
        x = rearranged_data[i, :, :,  :]
        y = rearranged_lbls[i, :, :]
        z = rearranged_info[i, :]
        yield (x, y, z)