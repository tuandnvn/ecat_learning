'''
Created on Mar 7, 2017

@author: Tuan
'''
import unittest

from crf_tree import CRFTree


class TestTreeCrf(unittest.TestCase):
    def setUp(self):
        self.tree_1 = CRFTree (["1", "2", "3", "4"], {}, {"1": ["2"], "2": ["1", "3", "4"], "3": ["2"], "4": ["2"] })
        self.tree_2 = CRFTree (["1", "2", "3", "4"], {}, {"1": ["2", "3"], "2": ["1", "3", "4"], "3": ["1", "2"], "4": ["2"] })
        self.tree_2 = CRFTree (["1", "2", "3", "4"], {}, {"1": ["2", "3"], "2": ["1", "3"], "3": ["1", "2"], "4": [] })
        unittest.TestCase.setUp(self)
        
    def tearDown(self):
        unittest.TestCase.tearDown(self)
        
    def test_is_tree(self):
        self.assertTrue(self.tree_1.check_tree())
        self.assertFalse(self.tree_2.check_tree())
        self.assertFalse(self.tree_3.check_tree())
        
if __name__ == '__main__':
    unittest.main()