'''
Created on Mar 6, 2017

@author: Tuan
'''
from collections import deque
import copy

import numpy as np
import tensorflow as tf

def gather_2d(params, indices):
    # only for two dim now
    shape = params.get_shape().as_list()
    assert len(shape) == 2, 'only support 2d matrix'
    indices_shape = indices.get_shape().as_list()
    assert indices_shape[1] == 2, 'only support indexing on both dimensions'
    
    flat = tf.reshape(params, [shape[0] * shape[1]])
#     flat_idx = tf.slice(indices, [0,0], [shape[0],1]) * shape[1] + tf.slice(indices, [0,1], [shape[0],1])
#     flat_idx = tf.reshape(flat_idx, [flat_idx.get_shape().as_list()[0]])
    
    flat_idx = indices[:,0] * shape[1] + indices[:,1]
    return tf.gather(flat, flat_idx)

def gather_2d_to_shape(params, indices, output_shape):
    flat = gather_2d(params, indices)
    return tf.reshape(flat, output_shape)

# x -> (x, size)
def expand( tensor, size, axis = 1 ):
    return tf.stack([tensor for _ in xrange(size)], axis = axis)

# x -> (size, x)
def expand_first( tensor, size ):
    return tf.pack( [tensor for _ in xrange(size)] )


class CRFTree(object):
    '''
    '''
    
    def __init__(self, node_types, dictionaries, edges):
        '''
        Parameters:
        ----------
        node_types:    list of String
        dictionaries:  gensim Dictionaries that store words in the tuples
        edges:         dictionary of list representation of edges
        '''
        self.node_types = node_types
        self.node_type_indices = dict((node_type, i) for i, node_type in enumerate(self.node_types))
        self.dictionaries = dictionaries
        self.edges = edges

        for node_type in node_types:
            if node_type not in dictionaries or len(dictionaries[node_type] ) == 0:
                raise Exception('There should be at least one label for each node_type')

    def initiate_crf(self):
        self.crf = {}

        with tf.variable_scope("crf"):
            for node_1 in self.edges:
                for node_2 in self.edges[node_1]:
                    edge = (node_1, node_2)
                    sorted_edge = tuple(sorted(edge) )
                    if not sorted_edge in self.crf:
                        source, target = sorted_edge
                        if source in self.dictionaries and target in self.dictionaries:
                            self.crf[sorted_edge] = tf.get_variable("A_" + source + '_' + target, 
                                                        [len(self.dictionaries[source]), len(self.dictionaries[target])])

    def is_tree(self):
        '''
        Check to see if the input graph is actually a tree
        
        Return:
        ------
        True If is a tree graph
        '''
        """
        Just start from any node, BFS through the tree, if it visited all nodes, and doesn't come back to any node
        """
        visited = dict([ (node, False) for node in self.node_types])
        print 'visited' , visited
        if len(self.node_types) == 0:
            return False
        
        start = self.node_types[0]
        
        q = deque([(None, start)])
        
        while len(q) != 0:
            parent, visit = q.popleft()

            visited[visit] = True
            for t in self.edges[visit]:
                if t != parent:
                    if visited[t]:
                        # Detect a circular
                        return False

                    q.append( (visit, t) )
        
        return all(visited.values())
    
    @staticmethod
    def look_for_collapsing_node(edges):
        '''
        Find a node that has the most edges connected to leaf
        '''
        leaves = set([node for node in edges if len(edges[node]) == 1])
        
        count_leaf = sorted( [ (node, set(edges[node]) & leaves)  for node in edges], key = lambda x: len(x[1]) )
        
        selected_node, collapsed_nodes = count_leaf[-1]
        
        return (selected_node, collapsed_nodes)
        
        
    @staticmethod
    def empty(edges):
        for source in edges:
            if edges[source] != None and len(edges[source]) != 0:
                return False
            
        return True
        
    def sum_over(self, crf_weight, batch_size, logits):
        '''
        Sum over all exponential term for all combination of values 
        
        Parameters:
        -----------
        crf_weight:     Weight for CRF 
        batch_size:     add a batch_size so that we don't have to recalculate
        logits:         for each node_type, an np array of ( batch_size, #node_type_targets )
        
        
        Return:
        -------
        log_sum =  numpy array of size = batch_size
        '''
        
        # Remove edges on the cloned edges, not from self.edges
        # We will remove until there are no edges left
        cloned_edges = copy.deepcopy( self.edges )
        
        def recursive_sum_over( edges, logits ):
            '''
            Recursively sum over the exponential components, given the current state of edges and logits
            
            Parameters:
            -----------
            edges:          Current state of edges (some nodes and edges might have been collapsed) 
            logits:         add a batch_size so that we don't have to recalculate
            
            
            Return:
            -------
            log_sum =  numpy array of size = batch_size
            '''
            if not CRFTree.empty(edges):
                # All nodes in collapsed_nodes will be collapsed into selected_node
                selected_node, collapsed_nodes = CRFTree.look_for_collapsing_node(edges)
                
                size_source = len(self.dictionaries[selected_node])
                log_sum = logits[size_source]
                
                for collapsed_node in collapsed_nodes:
                    sorted_edge = tuple(sorted((selected_node, collapsed_node)))
                    A = self.crf[sorted_edge]
                    logit = logits[collapsed_node]
                    
                    size_target = len(self.dictionaries[collapsed_node])
                    
                    if selected_node == sorted_edge[0]:
                        # Same order
                        log_edge = tf.reduce_min(crf_weight * A + expand(logit, size_source, axis = 2), 1)
                        
                        log_edge += tf.log(tf.reduce_sum(tf.exp(crf_weight * A +\
                                                                expand(logit, size_source, axis = 2) -\
                                                                expand(log_edge, size_target, axis = 1) ), 1))
                    else:
                        # Reverse order
                        log_edge = tf.reduce_min(crf_weight * tf.transpose(A) + expand(logit, size_source, axis = 2), 1)
                        
                        log_edge += tf.log(tf.reduce_sum(tf.exp(crf_weight * tf.transpose(A) +\
                                                                expand(logit, size_source, axis = 2) -\
                                                                expand(log_edge, size_target, axis = 1) ), 1))
                        
                    log_sum += log_edge
            
                logits[selected_node] = log_sum
                
                for collapsed_node in collapsed_nodes:
                    del edges[collapsed_node]
                    del logits[collapsed_node]
                
                # Remaining nodes that connected to collapsed_selected_node
                remaining_nodes = list(set(edges[selected_node]) - collapsed_nodes)
                
                edges[selected_node] = remaining_nodes
                
                return recursive_sum_over ( edges, logits )
            else:
                #There should be only one key in logits, otherwise throw an Error
                if len(logits) == 1:
                    remaining_node = logits.values()[0]
                    
                    # ( #remaining_node, batch_size)
                    logit = tf.transpose(logits[remaining_node])
                    
                    # ( batch_size )
                    log_sum = tf.reduce_min(logit, 0)
                    log_sum += tf.log(tf.reduce_sum(tf.exp(logit - log_sum), 0))
                    
                    return log_sum
                else:
                    raise Exception("At this state, there should be only one logit")
        
        return recursive_sum_over (cloned_edges, logits)
    
    def calculate_logit_correct(self, crf_weight, batch_size, logits, targets):
        '''
        Sum over all exponential term for all combination of values 
        
        Parameters:
        -----------
        crf_weight:     Weight for CRF 
        batch_size:     add a batch_size so that we don't have to recalculate
        logits:         for each node_type, an np array of ( batch_size, #node_type_targets )
        targets:        Correct labels for training, of size np.array ( batch_size, #self.node_types)
        
        Return:
        -------
        logit_correct =  numpy array of size = batch_size
        '''
        
        logit_correct = tf.zeros(batch_size)
        
        for source in self.edges:
            source_id = self.node_type_indices[source]
            for target in self.edges[source]:
                sorted_edge = tuple(sorted((source, target)))
                # Only count 1 for each edge
                if (source, target) == sorted_edge:
                    target_id = self.node_type_indices[target]
                    logit_correct += crf_weight * gather_2d ( self.crf[sorted_edge], tf.tranpose(tf.pack([targets[:, source_id], targets[:, target_id]])))
            
            logit_correct += gather_2d(logits[source], tf.transpose(tf.pack([tf.range(batch_size), targets[:, source_id]])))

        return logit_correct
    
    def predict(self, crf_weight, batch_size, logits ):
        '''
        Argmax over all exponential combinations of values.
        This is analogous to the process in sum_over
        
        Parameters:
        -----------
        crf_weight:     Weight for CRF 
        batch_size:     add a batch_size so that we don't have to recalculate
        logits:         for each node_type, an np array of ( batch_size, #node_type_targets )
        
        
        Return:
        -------
        out:            numpy array of size = (batch_size, len(self.node_types) )
        
        Here I use a kind of collapsing algorithm, each time, looking for a node to collapse on. All leaf nodes connected to this node 
        are collapsed into this center node.
        
        
        '''
        # Remove edges on the cloned edges, not from self.edges
        # We will remove until there are no edges left
        cloned_edges = copy.deepcopy( self.edges )
        
        def recursive_predict(edges, logits, best_combinations, best_values ):
            '''
            Parameters:
            -----------
            edges:          Current state of edges (some nodes and edges might have been collapsed) 
            logits:         add a batch_size so that we don't have to recalculate
            best_combinations: Current best combination states, map from node (only nodes still appear in edges) to a tensorflow of size (batch_size, # of values)
            
            Return:
            -------
            out:            numpy array of size = (batch_size, len(self.node_types) )
            '''
            if not CRFTree.empty(edges):
                pass
            else:
                # batch_size
                best_best_values = tf.argmax(best_values, 1)
                
                indices = tf.transpose( tf.pack([range(self.batch_size), best_best_values]))
                
                out = tf.transpose(tf.pack([gather_2d( best_combinations[t], indices ) for t in xrange(self.n_labels)]))
                
                return out
        
        """
        best_combinations is a dictionary from node to a tensor sized (self.batch_size, len(self.dictionaries[node]) )
        
        Everytime the graph is collapsed, 
        
        
        """
        best_combinations = {}
        
        """
        best_values is a dictionary from node to a tensor sized (self.batch_size, len(self.dictionaries[node]) )
        """
        best_values = {}
        return recursive_predict (cloned_edges , logits, best_combinations, best_values )
        
    
        no_of_theme = no_of_subject = no_of_object =  len(role_to_id)
        no_of_prep = len(prep_to_id)
        no_of_event = len(event_to_id)
            
        logit_s = self.logits[0]
        logit_o = self.logits[1]
        logit_t = self.logits[2]
        logit_e = self.logits[3]
        logit_p = self.logits[4]
        
        '''---------------------------------------------------------------'''
        '''Message passing algorithm to max over terms of all combinations'''
        '''---------------------------------------------------------------'''
        # For theme
        best_combination_theme = [tf.zeros((self.batch_size, no_of_theme), dtype=np.int64) for _ in xrange(self.n_labels)]

        # For subject
        best_combination_subject = [tf.zeros((self.batch_size, no_of_subject), dtype=np.int64) for _ in xrange(self.n_labels)]
        
        # (batch_size, #Theme)
        best_theme_values = logit_t + self.crf_weight * self.A_start_t
        
        best_combination_theme[2] = expand_first(range(no_of_theme), self.batch_size)
        
        # (#Object, batch_size, #Theme)
        o_values = [expand(logit_o[:, o], no_of_theme) + self.crf_weight * self.A_to[:,o]  for o in xrange(no_of_object)]
        best_theme_values += tf.reduce_max(o_values, 0)
        
        best_combination_theme[1] = tf.cast(tf.argmax(o_values, 0), np.int32)
        
        # (#Prep, batch_size, #Theme)
        p_values = [expand(logit_p[:, p],no_of_theme)  + self.crf_weight * self.A_tp[:,p] for p in xrange(no_of_prep)]
        best_theme_values += tf.reduce_max(p_values, 0)
        
        best_combination_theme[4] = tf.cast(tf.argmax(p_values, 0), np.int32)
        
        # (batch_size, #Subject)
        best_subject_values = logit_s
        
        # Message passing between Theme and Subject
        # (#Theme, batch_size, #Subject)
        t_values = [expand(best_theme_values[:, t], no_of_subject)  + self.crf_weight * self.A_ts[t,:] for t in xrange(no_of_theme)]
        best_subject_values += tf.reduce_max(t_values, 0)
        
        # (batch_size, #Subject)
        best_t = tf.argmax(t_values, 0)
        
        # (batch_size, #Subject)
        q = np.array([[i for _ in xrange(no_of_subject)] for i in xrange(self.batch_size)])
        
        # (batch_size x #Subject, 2)
        indices = tf.reshape( tf.transpose( tf.pack ( [q, best_t]), [1, 2, 0] ), [-1, 2]) 
        
        for index in xrange(self.n_labels):
            best_combination_subject[index] = gather_2d_to_shape(best_combination_theme[index], 
                                                 indices, (self.batch_size, no_of_subject))
        best_combination_subject[0] = expand_first(range(no_of_subject), self.batch_size)

        # Message passing between Subject and Verb
        # (#Event, batch_size, #Subject)
        e_values = [expand(logit_e[:, e], no_of_subject) + self.crf_weight * self.A_se[:,e] for e in xrange(no_of_event)]
        # (batch_size, #Subject)
        best_subject_values += tf.reduce_max(e_values, 0)
        
        best_combination_subject[3] = tf.cast(tf.argmax(e_values, 0), np.int32)

        # Take the best out of all subject values
        # batch_size
        best_best_subject_values = tf.argmax(best_subject_values, 1)
        
        # (batch_size, 2)
        # Indices on best_combination_subject[index] should have order of (self.batch_size, #Subject)
        indices = tf.transpose( tf.pack([range(self.batch_size), best_best_subject_values]))
        
        # (batch_size, self.n_labels)
        out = tf.transpose(tf.pack([gather_2d( best_combination_subject[t], indices ) for t in xrange(self.n_labels)]))