'''
Created on Mar 7, 2017

@author: Tuan
'''
import unittest

from crf_tree import CRFTree


class TestTreeCrf(unittest.TestCase):
    def setUp(self):
        self.tree_1 = CRFTree (["1", "2", "3", "4"], dict( (str(i), [str(i)]) for i in xrange(1, 5)) , {"1": ["2"], "2": ["1", "3", "4"], "3": ["2"], "4": ["2"] })
        self.tree_2 = CRFTree (["1", "2", "3", "4"], dict( (str(i), [str(i)]) for i in xrange(1, 5)), {"1": ["2", "3"], "2": ["1", "3", "4"], "3": ["1", "2"], "4": ["2"] })
        self.tree_3 = CRFTree (["1", "2", "3", "4"], dict( (str(i), [str(i)]) for i in xrange(1, 5)), {"1": ["2", "3"], "2": ["1", "3"], "3": ["1", "2"], "4": [] })
        self.tree_4 = CRFTree ([str(i) for i in xrange(1, 11)], dict( (str(i), [str(i)]) for i in xrange(1, 11)), 
                               {"1": ["2"], "2": ["1", "3", "4"], "3": ["2"], "4": ["2", "7", "8"], "5": ["7"], "6": ["7"], "7": ["5", "6", "4"],
                                "8": ["4", "9", "10"], "9": ["8"],  "10": ["8"]})
        unittest.TestCase.setUp(self)
        
    def tearDown(self):
        unittest.TestCase.tearDown(self)
        
    def test_is_tree(self):
        self.assertTrue(self.tree_1.is_tree())
        self.assertFalse(self.tree_2.is_tree())
        self.assertFalse(self.tree_3.is_tree())
        self.assertTrue(self.tree_4.is_tree())
        
    def test_look_for_collapsing_node(self):
        self.assertEqual( CRFTree.look_for_collapsing_node(self.tree_1.edges), ( "2", ["1", "3", "4"]))
        self.assertIn( CRFTree.look_for_collapsing_node(self.tree_4.edges), [ ("2", ["1", "3"]), ("7", ["5", "6"]), ("8", ["9", "10"]) ] )
        
if __name__ == '__main__':
    unittest.main()