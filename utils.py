'''
Created on Mar 4, 2017

@author: Tuan
'''
import os


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join( ROOT_DIR, 'data') 

SESSION_NAME = "session_name"
SESSION_DATA = "session_data"
SESSION_EVENTS = "session_events"

SUBJECT = 'Subject'
OBJECT = 'Object'
THEME = 'Theme'
EVENT = 'Event'
PREP = 'Preposition'
ALL_SLOTS = [SUBJECT, OBJECT, THEME, EVENT, PREP]

RAW = 'raw'
PCAS = 'pcas'
QSR = 'qsr'
EVENT = 'event'
SPARSE_QSR = 'sparse_qsr'

DEVICE = '/gpu:0'
TEST_DEVICE = '/cpu:0'

role_to_id = {'None' : 0, 'Performer': 1, 'Object_1': 2, 'Object_2' : 3}
event_to_id = { 'None': 0, 'push' : 1, 'pull' : 2 , 'roll' : 3, 'slide': 4}
prep_to_id = {'None': 0, 'Past': 1, 'From': 2, 'To': 3}

id_to_role = {}
id_to_event = {}
id_to_prep = {}

for key, value in role_to_id.iteritems():
    id_to_role[value] = key
for key, value in event_to_id.iteritems():
    id_to_event[value] = key
for key, value in prep_to_id.iteritems():
    id_to_prep[value] = key
    
label_classes = [role_to_id, role_to_id, role_to_id, event_to_id, prep_to_id]
num_labels = len(label_classes)

def from_str_labels_to_id_labels(subj, obj, theme, event, prep):
    subj = role_to_id[subj]
    obj = role_to_id[obj]
    theme = role_to_id[theme]
    event = event_to_id[event]
    prep = prep_to_id[prep]
    
    return (subj, obj, theme, event, prep)

def from_id_labels_to_str_labels(subj, obj, theme, event, prep):
    subj = id_to_role[subj]
    obj = id_to_role[obj]
    theme = id_to_role[theme]
    event = id_to_event[event]
    prep = id_to_prep[prep]
    
    return (subj, obj, theme, event, prep)