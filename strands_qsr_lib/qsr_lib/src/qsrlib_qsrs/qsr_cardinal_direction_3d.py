# -*- coding: utf-8 -*-
from __future__ import print_function, division
from qsrlib_qsrs.qsr_dyadic_abstractclass import QSR_Dyadic_1t_Abstractclass
import math
import itertools 

class QSR_Cardinal_Direction_3d(QSR_Dyadic_1t_Abstractclass):
    """Cardinal direction relations in 3 dimensions
    Following the discussion here (with a twist):
    http://web.mst.edu/~chaman/home/pubs/2014xCardinalDirectionsMIKE2014.pdf

    Values of the abstract properties
        * **_unique_id** = "cardir"
        * **_all_possible_relations** = (all Cartesian products of ["north", "south", "o"], ["east", "west", "o"], ["above", "below", "o"])
        * **_dtype** = "bounding_boxes_3d"

    Some explanation about the QSR or better link to a separate webpage explaining it. Maybe a reference if it exists.
    """

    _unique_id = "cardir3d"
    """str: Unique identifier name of the QSR."""

    _all_possible_relations = tuple(itertools.product('nso', 'ewo', 'abo'))
    """tuple: All possible relations of the QSR."""

    _dtype = "bounding_boxes_3d"
    """str: On what kind of data the QSR works with."""

    def __init__(self):
        """Constructor."""
        super(QSR_Cardinal_Direction_3d, self).__init__()

    def _compute_qsr(self, data1, data2, qsr_params, **kwargs):
        """Compute QSR relation.

        :param data1: 3d Bounding box.
        :type data1: list or tuple of int or floats
        :param data2: 3d Bounding box.
        :type data2: list or tuple of int or floats
        :return: QSR relation.
        :rtype: str
        """
        # Finds the differnece between the centres of each object
        dx = ((data2[0]+data2[3])/2.0) - ((data1[0]+data1[3])/2.0)
        dy = ((data2[1]+data2[4])/2.0) - ((data1[1]+data1[4])/2.0)
        dz = ((data2[2]+data2[5])/2.0) - ((data1[2]+data1[5])/2.0)

        dx_c = 'o'
        dy_c = 'o'
        dz_c = 'o'

        min_distance_for_o = qsr_params['distance_threshold']

        if dx >= min_distance_for_o:
            dx_c = 'n'
        elif dx <= min_distance_for_o:
            dx_c = 's'

        if dy >= min_distance_for_o:
            dy_c = 'e'
        elif dy <= min_distance_for_o:
            dy_c = 'w'

        if dz >= min_distance_for_o:
            dz_c = 'a'
        elif dz <= min_distance_for_o:
            dz_c = 'b'

        # Lookup labels and return answer
        return dx_c + dy_c + dz_c
