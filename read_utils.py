'''
Created on Mar 4, 2017

@author: Tuan
'''

import glob
import os
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import PCA

from utils import DATA_DIR, SESSION_NAME, SESSION_DATA, SESSION_EVENTS, \
    from_str_labels_to_id_labels
import xml.etree.ElementTree as ET

from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import Object_State, World_Trace


def read_project_data():
    ps = PorterStemmer() 
    
    project_data = {}
    
    data_length = None
    
    for file_name in glob.glob(os.path.join(DATA_DIR, '*.txt')):
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
            
            '''
            Calculate the difference of data points -> gradient feature
            Move all points to the same coordinations
            '''
            #session_data[SESSION_DATA] = [[(session_data[SESSION_DATA][i][t] - session_data[SESSION_DATA][0][0])\
            #     for t in xrange(data_length)]\
            #     for i in xrange(0, len(session_data[SESSION_DATA]))]

            event_elements = session_element.findall('events/event')

            for event_element in event_elements:
                event_str = {}
                event_str['start'] = event_element.attrib['start']
                event_str['end'] = event_element.attrib['end']

                rig_role, glyph_role_1, glyph_role_2, event, prep = event_element.text.split(',')
                
                # Mapping from old structure to new structure
                mappings = {0: "Performer", 1 : "Object_1", 2 : "Object_2"}
                subj = obj = theme = "None"
                for i, role in enumerate([rig_role, glyph_role_1, glyph_role_2]):
                    if role == "Subject":
                        subj = mappings[i]
                    if role == "Object":
                        obj = mappings[i]
                    if role == "Theme":
                        theme = mappings[i]
                    
                event = ps.stem(event)
                
                subj, obj, theme, event, prep =\
                    from_str_labels_to_id_labels(subj, obj, theme, event, prep)

                event_str['label'] = (subj, obj, theme, event, prep)

                session_data[SESSION_EVENTS].append(event_str)
            
            project_data[project_name].append(session_data) 
            
    return data_length, project_data

'''
List of joints

"SpineShoulder", "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft", "HandTipLeft", "ThumbLeft",
                "ShoulderRight", "ElbowRight", "WristRight", "HandRight", "HandTipRight", "ThumbRight"
'''
def read_pca_features():
    '''
    Find planes for projection of data from DATA_DIR 
    '''
    all_rig_points = []
    all_object_points = []

    no_of_samples = 0

    for file_name in glob.glob(os.path.join(DATA_DIR, '*.txt')):
        tree = ET.parse(file_name)
        doc = tree.getroot()
        for session_element in doc.findall('session'):
            frame_elements = session_element.findall('data/frame')

            for frame_element in frame_elements:
                object_point_elements = frame_element.findall('o')

                rig = object_point_elements[0]
                o1, o2 = object_point_elements[1], object_point_elements[2]

                rig_points = []
                for s in rig.text.split(','):
                    rig_points.append(float(s))

                o1_points = []
                for s in o1.text.split(','):
                    o1_points.append(float(s))

                o2_points = []
                for s in o2.text.split(','):
                    o2_points.append(float(s))


                all_rig_points.append(rig_points)
                all_object_points.append(o1_points)
                all_object_points.append(o2_points)

    # (13 * samples, 3)
    all_rig_array = np.array(all_rig_points).reshape((-1, 3))

    print (all_rig_array.shape)

    # (8 * samples, 3)
    all_object_array = np.array(all_object_points).reshape((-1, 3))

    print (all_object_array.shape)

    '''
    PCA for inter-object relationship
    This PCA would be an estimation of projecting on the table surface
    '''
    inter_object_pca = PCA(n_components=2)
    all_points = np.concatenate((all_rig_array, all_object_array), axis = 0)
    
    # (13 * samples + 8 * samples, 2)
    inter_object_pca.fit(all_points)

    '''
    PCA for intra-object relationship
    '''
    intra_object_pca = PCA(n_components=2)

    intra_object_pca.fit(all_object_array)

    '''
    PCA for intra-rig relationship
    This would be a good 
    '''
    intra_rig_pca = PCA(n_components=2)

    intra_rig_pca.fit(all_rig_array)

    data_length = None
    _, project_data = read_project_data()

    for project_name in project_data:
        print '-----------------------------------'
        print project_name
        print '-----------------------------------'
        for session_data in project_data[project_name]:

            point_datas = session_data[SESSION_DATA]

            new_session_datas = []

            for point_data in point_datas:
                new_session_data = []

                "---------------------------------------------"
                # (21, 2)
                inter_object_data = inter_object_pca.transform( np.array(point_data).reshape((-1, 3)) )

                # Centroid of three objects projected using inter_object_pca
                new_session_data.append( (inter_object_data[1] + inter_object_data[7] ) / 2 ) # Body centroid
                new_session_data.append( np.average( inter_object_data[5:6], axis = 0 ) )     # Left hand
                new_session_data.append( np.average( inter_object_data[11:12], axis = 0 ) )   # Right hand
                new_session_data.append( np.average( inter_object_data[13:17], axis = 0 ) )
                new_session_data.append( np.average( inter_object_data[17:21], axis = 0 ) )

                "---------------------------------------------"
                # (21, 2)
                intra_rig_data = intra_rig_pca.transform( np.array(point_data).reshape((-1, 3)) )

                # Average of two shoulders
                #new_session_data.append( (intra_rig_data[1] + intra_rig_data[7] ) / 2 )

                # Hand tip left
                #new_session_data.append( intra_rig_data[5] )

                # Hand tip right
                #new_session_data.append( intra_rig_data[11] )

                "---------------------------------------------"
                intra_object_data = intra_object_pca.transform( np.array(point_data).reshape((-1, 3)) )

                # Corners of objects (for each object, just pick two corners)
                new_session_data.append( intra_object_data[13] )
                new_session_data.append( intra_object_data[15] )
                new_session_data.append( intra_object_data[17] )
                new_session_data.append( intra_object_data[19] )

                point_data = np.concatenate( new_session_data )

                # Should be 16
                if data_length == None:
                    data_length = point_data.shape[0]

                new_session_datas.append( point_data.tolist() )

            session_data[SESSION_DATA] = new_session_datas

    return data_length, project_data

cdid = dict( (u, i) for (i, u) in enumerate( ['n', 'nw', 'w', 'sw', 's', 'se', 'e', 'ne', 'eq'] ))
mosd = dict( (u, i) for (i, u) in enumerate( ['s', 'm'] ))

def cardir_index ( cardir ):
    return cdid [cardir]

def mos_index ( mos ):
    return mosd [mos]

def turn_response_to_features(keys, qsrlib_response_message):
    feature_chain = []
    for t in qsrlib_response_message.qsrs.get_sorted_timestamps():
        features = []
        # print (qsrlib_response_message.qsrs.trace[t].qsrs.keys())
        for k in keys:
            if k in qsrlib_response_message.qsrs.trace[t].qsrs:
                v = qsrlib_response_message.qsrs.trace[t].qsrs[k]

                if 'cardir' in v.qsr:
                    f = v.qsr['cardir']
                    features.append(cardir_index(f))
                if 'argd' in v.qsr:
                    f = int( v.qsr['argd'] )
                    features.append(f)
                if 'mos' in v.qsr:
                    f = v.qsr['mos'] 
                    features.append(mos_index(f))
                
        feature_chain.append(features)
    
    if len(feature_chain) == 0:
        return feature_chain

    # The first frame doesn't has dyadic relation
    feature_chain[0] += [0, 0, 0]
    
    feature_chain = [ [feature_chain[t + 1][i] - feature_chain[t][i] for i in xrange(len(feature_chain[0])) ]
                  for t in xrange(len(feature_chain) - 1)]
    
    feature_chain = [[0 for i in xrange(len(feature_chain[0]))]] +  feature_chain
    return feature_chain
        

def qsr_feature_extractor ( qsrlib, session_data ):
    '''
    List of features from qsr

    21 features
    
    ('body', 'left_hand') - cardir_diff, argd_diff
    ('body', 'right_hand') - cardir_diff, argd_diff 
    ('left_hand', 'o1_centroid') - cardir_diff, argd_diff 
    ('right_hand', 'o1_centroid') - cardir_diff, argd_diff 
    ('left_hand', 'o2_centroid') - cardir_diff, argd_diff
    ('right_hand', 'o2_centroid') - cardir_diff, argd_diff 
    ('o1_centroid', 'o2_centroid') - cardir_diff, argd_diff 
    ('o1_corner1','o1_corner2') - cardir_diff, argd_diff 
    ('o2_corner1','o2_corner2') - cardir_diff, argd_diff
    'body' - mos
    'o1_centroid' - mos
    'o2_centroid' - mos
    '''
    len_data = len(session_data)
    # body centroid
    body_centroid = [Object_State(name="body", timestamp=i, x=session_data[i][0], y=session_data[i][1], width=0.1, length=0.1) 
      for i in xrange(len_data)]
    
    # left hand tip
    left_hand = [Object_State(name="left_hand", timestamp=i, x=session_data[i][2], y=session_data[i][3], width=0.1, length=0.1) 
      for i in xrange(len_data)]
    
    # right hand tip
    right_hand = [Object_State(name="right_hand", timestamp=i, x=session_data[i][4], y=session_data[i][5], width=0.1, length=0.1) 
      for i in xrange(len_data)]
    
    # centroid of o1 object
    o1_centroid = [Object_State(name="o1_centroid", timestamp=i, x=session_data[i][6], y=session_data[i][7], width=0.1, length=0.1) 
          for i in xrange(len_data)]
    
    # centroid of o2 object
    o2_centroid = [Object_State(name="o2_centroid", timestamp=i, x=session_data[i][8], y=session_data[i][9], width=0.1, length=0.1) 
          for i in xrange(len_data)]
    
    # o1
    o1_corner1 = [Object_State(name="o1_corner1", timestamp=i, x=session_data[i][10], y=session_data[i][11], width=0.1, length=0.1) 
          for i in xrange(len_data)]
    
    o1_corner2 = [Object_State(name="o1_corner2", timestamp=i, x=session_data[i][12], y=session_data[i][13], width=0.1, length=0.1) 
          for i in xrange(len_data)]
    
    # o2
    o2_corner1 = [Object_State(name="o2_corner1", timestamp=i, x=session_data[i][14], y=session_data[i][15], width=0.1, length=0.1) 
          for i in xrange(len_data)]
    
    o2_corner2 = [Object_State(name="o2_corner2", timestamp=i, x=session_data[i][16], y=session_data[i][17], width=0.1, length=0.1) 
          for i in xrange(len_data)]

    world = World_Trace()
    world.add_object_state_series(body_centroid)
    world.add_object_state_series(left_hand)
    world.add_object_state_series(right_hand)
    world.add_object_state_series(o1_centroid)
    world.add_object_state_series(o2_centroid)
    world.add_object_state_series(o1_corner1)
    world.add_object_state_series(o1_corner2)
    world.add_object_state_series(o2_corner1)
    world.add_object_state_series(o2_corner2)
    
    interest_argd_pairs = [('body', 'left_hand'), ('body', 'right_hand'), ('left_hand', 'o1_centroid'), 
                           ('right_hand', 'o1_centroid'), ('left_hand', 'o2_centroid'), 
                           ('right_hand', 'o2_centroid'), ('o1_centroid', 'o2_centroid'), 
                           ('o1_corner1','o1_corner2'), ('o2_corner1','o2_corner2')]
    interest_cardir_pairs = interest_argd_pairs
    
    interest_argd_pair_keys = [u + ',' + v for (u, v) in interest_argd_pairs]
    
    interest_mos_elements = ['body', 'o1_centroid', 'o2_centroid']
    
    qsrlib_request_message = QSRlib_Request_Message(which_qsr=['cardir', 'mos', 'argd'], input_data=world, 
                    dynamic_args = {'cardir': {'qsrs_for': interest_cardir_pairs},
                                    'mos' : {'qsrs_for': interest_mos_elements, 'quantisation_factor': 0.005},
                                    'argd': {'qsrs_for': interest_argd_pairs, 
                                            'qsr_relations_and_values' : dict(("" + str(i), i * 1.0 / 20) for i in xrange(20)) }})
    # request your QSRs
    qsrlib_response_message = qsrlib.request_qsrs(req_msg=qsrlib_request_message)

    # pretty_print_world_qsr_trace(['cardir', 'mos', 'argd'], qsrlib_response_message)
    
    return turn_response_to_features(interest_argd_pair_keys + interest_mos_elements, qsrlib_response_message)

def read_qsr_features():
    '''
    Find planes for projection of data from DATA_DIR 
    '''
    _, project_data = read_pca_features()

    qsrlib = QSRlib()

    data_length = None

    for project_name in project_data:
        print '-----------------------------------'
        print project_name
        print '-----------------------------------'
        for session_data in project_data[project_name]:

            point_datas = session_data[SESSION_DATA]

            new_point_datas = qsr_feature_extractor( qsrlib, point_datas )

            session_data[SESSION_DATA] = new_point_datas

            if data_length == None:
                data_length = len(new_point_datas[0])

    return data_length, project_data
