'''
Created on Mar 4, 2017

@author: Tuan
'''

import glob
import os

from nltk.stem.porter import PorterStemmer

from utils import DATA_DIR, SESSION_NAME, SESSION_DATA, SESSION_EVENTS, \
    from_str_labels_to_id_labels
import xml.etree.ElementTree as ET

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
            session_data[SESSION_DATA] = [[(session_data[SESSION_DATA][i][t] - session_data[SESSION_DATA][0][0])\
                 for t in xrange(data_length)]\
                 for i in xrange(0, len(session_data[SESSION_DATA]))]

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