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
    
    flat_idx = indices[:, 0] * shape[1] + indices[:, 1]
    return tf.gather(flat, flat_idx)


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
        
        if not self.is_tree():
            print 'This is not a tree. Please insert a tree'
        
        self.crf_weight = {}
        
        with tf.variable_scope("crf"):
            for node_1 in edges:
                for node_2 in edges[node_1]:
                    edge = (node_1, node_2)
                    sorted_edge = sorted(edge) 
                    if not sorted_edge in self.crf_weight:
                        source, target = sorted_edge
                        self.crf_weight[sorted_edge] = tf.get_variable("A_" + source + '_' + target, 
                                                    [len(self.dictionaries[source]), len(self.dictionaries[target])])
                        
        
    def check_tree(self):
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
        if len(self.node_types) == 0:
            return False
        
        start = self.node_types[0]
        
        q = deque([start])
        
        while len(q) != 0:
            visit = q.popleft()
            
            visited[visit] = True
            for t in self.edges[visit]:
                if t != visit:
                    if  visited[t]:
                        # Detect a circular
                        return False
                    q.append(t)
                    
        return all(visited.values())
    
    @staticmethod
    def expand( logit, size ):
        return tf.matmul(tf.expand_dims(logit, axis = 1), tf.ones((1, size)) )
    
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
            if not CRFTree.empty(edges):
                # All nodes in collapsed_nodes will be collapsed into selected_node
                selected_node, collapsed_nodes = CRFTree.look_for_collapsing_node(edges)
                
                size_source = len(self.dictionaries[selected_node])
                log_sum = logits[size_source]
                
                for collapsed_node in collapsed_nodes:
                    sorted_edge = sorted((selected_node, collapsed_node))
                    A = self.crf_weight[sorted_edge]
                    logit = logits[collapsed_node]
                    
                    size_target = len(self.dictionaries[collapsed_node])
                    
                    if selected_node == sorted_edge[0]:
                        # Same order
                        log_edge = tf.reduce_min([(crf_weight * A[:,o] + CRFTree.expand(logit[:, o], size_source)) 
                                                for o in xrange(size_target)], 0)
                        
                        log_edge += tf.log(tf.reduce_sum([tf.exp(crf_weight * A[:,o] +\
                                                                 CRFTree.expand(logit[:, o], size_source) -\
                                                                  log_edge) 
                                                for o in xrange(size_target)], 0))
                    else:
                        # Reverse order
                        log_edge = tf.reduce_min([(crf_weight * A[o,:] + CRFTree.expand(logit[:, o], size_source)) 
                                                for o in xrange(size_target)], 0)
                        
                        log_edge += tf.log(tf.reduce_sum([tf.exp(crf_weight * A[o,:] +\
                                                                 CRFTree.expand(logit[:, o], size_source) -\
                                                                  log_edge) 
                                                for o in xrange(size_target)], 0))
                        
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
                sorted_edge = sorted((source, target))
                # Only count 1 for each edge
                if (source, target) == sorted_edge:
                    target_id = self.node_type_indices[target]
                    logit_correct += crf_weight * gather_2d ( self.crf_weight[sorted_edge], tf.tranpose(tf.pack([targets[:, source_id], targets[:, target_id]])))
            
            logit_correct += gather_2d(logits[source], tf.transpose(tf.pack([tf.range(batch_size), targets[:, source_id]])))

        return logit_correct