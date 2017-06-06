#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import abstractmethod, ABCMeta
from qsrlib_qsrs.qsr_dyadic_abstractclass import QSR_Dyadic_Abstractclass
from qsrlib_io.world_qsr_trace import *
from exceptions import Exception, AttributeError
import numpy as np

# todo /use/bin/env python not needed here.

class QTCException(Exception):
    """?"""
    pass


class QSR_QTC_Simplified_3d(QSR_Dyadic_Abstractclass):
    """QTCS abstract class.

    """

    """

    print "where,\n" \
    "it is always necessary to have two agents in every timestep:\n"\
    "x, y: the xy-coords of the agents\n" \
    "quantisation_factor: the minimum distance the agents must diverge from the double cross between two timesteps to be counted as movement. Must be in the same unit as the x,y coordinates.\n"\
    Following the discussion from 
    QTC3D: Extending the Qualitative Trajectory Calculus to Three Dimensions


    A:  k is moving towards/away from l
    B: l is moving away from k
    C: k is faster/slower than l
    D: k is moving towards the right/left side of (kl)
    E: l is moving towards the right/left side of (lk)
    F: the angle between vk and (kl) is larger/smaller than the angle between vl and (lk)
    G: Yaw angle <> 0
    H: Pitch angle <> 0
    I: Roll angle <> 0
    """
    __metaclass__ = ABCMeta

    __global_unique_id = "qtcs3d"
    """?"""

    __no_state__ = 9.
    """?"""

    _dtype = "points"
    """str: QTC specific type."""

    def __init__(self):
        """Constructor."""
        super(QSR_QTC_Simplified_3d, self).__init__()

        # todo should be private abstractproperty
        self.qtc_type = ""
        """?"""

        # todo commenting for following could go in the class docstring
        self.__qsr_params_defaults= {
            "quantisation_factor": 0.0,
            "angle_quantisation_factor": np.pi/ 20,
            "distance_threshold": 1.22
        }

        self._all_possible_relations = tuple(self.return_all_possible_state_combinations()[0])

    def return_all_possible_state_combinations(self):
        """Return all possible state combinations for the qtc_type defined for this class instance.

        :return: All possible state combinations.
        :rtype:
                * String representation as a list of possible tuples, or,
                * Integer representation as a list of lists of possible tuples

        """
        ret_str = []
        ret_int = []
        for A in xrange(1, 4):
            for B in xrange(1, 4):
                for G in xrange(1, 4):
                    for H in xrange(1, 4):
                        for I in xrange(1, 4):
                            new_item = [A-2, B-2, G-2, H-2, I-2]
                            ret_int.append(new_item)
                            ret_str.append(','.join(str(a) for a in new_item))
            
        return [s.replace('-1','-').replace('1','+') for s in ret_str], ret_int

    def _nan_equal(self, a, b):
        """Uses assert equal to compare if two arrays containing nan values are equal.

        :param a: First array.
        :type a: ?
        :param b: Second array.
        :type b: ?
        :return: `True` or `False`
        :rtype: bool
        """
        try:
            np.testing.assert_equal(a,b)
        except AssertionError:
            return False
        return True

    def _create_qtc_representation(self, pos_k, pos_l, quantisation_factor=0, angle_quantisation_factor=0):
        """Create the QTCCS representation for the given data.

        Uses the double cross to determine to which side of the lines the points are moving.

        :param pos_k: Array of positions for agent k, exactly 3 entries of x,y,z positions.
        :type pos_k: ?
        :param pos_l: Array of positions for agent l, exactly 3 entries of x,y,z positions
        :type pos_l: ?
        :param quantisation_factor: The minimum distance the points have to diverge from either line to be regarded a non-0-state.
        :type quantisation_factor: float
        :return: QTCC3D 5-tuple (A, B, G, H, I) for the movement of the two agents: [k[0],l[0]].
        :rtype: numpy.array

        Decomposition of a rotation matrix into three angles (yaw, pitch, and roll):

        http://planning.cs.uiuc.edu/node103.html

        alpha = atan2(r_21,r_11)
        beta = atan2( -r_31, sqrt(r_32^2 + r_33^2) )
        gamma = atan2(r_32, r_33)
        """
        #print "######################################################"
        pos_k = np.array(pos_k).reshape(-1, 3)
        pos_l = np.array(pos_l).reshape(-1, 3)

        k_to_l = pos_l[-2] - pos_k[-2]
        l_to_k = pos_k[-2] - pos_l[-2]

        A_value = np.dot( k_to_l, pos_k[-2] - pos_k[-1]) / np.linalg.norm(k_to_l)^2
        B_value = np.dot( l_to_k, pos_l[-2] - pos_l[-1]) / np.linalg.norm(k_to_l)^2

        A = QSR_QTC_Simplified_3d._get_distance_symbol ( A_value )
        B = QSR_QTC_Simplified_3d._get_distance_symbol ( B_value )

        moving_vectors = [pos_k[-1] - pos_k[-2], pos_l[-1] - pos_l[-2]]
        moving_vectors_prev = [pos_k[-2] - pos_k[-3], pos_l[-2] - pos_l[-3]]

        if np.allclose( moving_vectors[0], np.zeros(3)) or np.allclose( moving_vectors[1], np.zeros(3))\
            or np.allclose( moving_vectors_prev[0], np.zeros(3)) or np.allclose( moving_vectors_prev[1], np.zeros(3)):
            return numpy.append(A, B, 0, 0, 0)

        tangent_vectors = np.array([x/np.linalg.norm(x) for x in moving_vectors])
        tangent_vectors_prev = np.array([x/np.linalg.norm(x) for x in moving_vectors_prev])
        tangent_diff_vectors = tangent_vectors - tangent_vectors_prev
        normal_vectors = np.array([x/np.linalg.norm(x) for x in tangent_diff_vectors])
        binormal_vectors = np.array([ np.cross(x,y) for x, y in zip(tangent_vectors, normal_vectors)])

        k_frame = np.append(
                tangent_vectors[0],
                normal_vectors[0],
                binormal_vectors[0]
            )

        l_frame = np.append(
                tangent_vectors[1],
                normal_vectors[1],
                binormal_vectors[1]
            )

        '''
        I don't even need to care about inversion of degenerated matrix
        because k_frame couldn't be degenerated
        '''
        transition_matrix = np.matmul(l_frame, numpy.linalg.inv(k_frame))

        yaw = np.arctan2(transition_matrix[2,1], transition_matrix[1,1])
        pitch = np.arctan2(-transition_matrix[3,1], np.sqrt( transition_matrix[3,2]^2 + transition_matrix[3,3]^2 ) )
        roll = np.arctan2(transition_matrix[3,2], transition_matrix[3,3])

        G = QSR_QTC_Simplified_3d._get_angle_symbol(yaw, angle_quantisation_factor)
        H = QSR_QTC_Simplified_3d._get_angle_symbol(pitch, angle_quantisation_factor)
        I = QSR_QTC_Simplified_3d._get_angle_symbol(roll, angle_quantisation_factor)
        
        return numpy.append(A, B, G, H, I)

    @staticmethod
    def _get_distance_symbol( value, quantisation_factor ):
        res = 0
        if value > quantisation_factor:
            res = 1
        elif value < -quantisation_factor:
            res = -1

        return res

    @staticmethod
    def _get_angle_symbol( angle, angle_quantisation_factor ):
        res = 0
        if angle > angle_quantisation_factor:
            res = 1
        elif angle < -angle_quantisation_factor:
            res = -1

        return res

    def _custom_checks_world_trace(self, input_data, qsr_params):
        """Custom check of input data.

        :param input_data: Input data.
        :type input_data: :class:`World_Trace <qsrlib_io.world_trace.World_Trace>`
        :param qsr_params: QSR specific parameters passed in `dynamic_args`.
        :type qsr_params: dict
        :return: False for no problems.
        :rtype: bool
        :raises:
            * ValueError: "Data for at least three separate timesteps has to be provided."
            * KeyError: "Only one object defined for timestep %f. Two objects have to be present at any given step."
            * ValueError: "Coordinates x: %f, y: %f are not defined correctly for timestep %f."
        """
        timestamps = input_data.get_sorted_timestamps()
        if len(timestamps) < 3:
            raise ValueError("Data for at least three separate timesteps has to be provided.")
        objects_names = sorted(input_data.trace[timestamps[0]].objects.keys())
        for t in timestamps:
            for o in objects_names:
                try:
                    input_data.trace[t].objects[o]
                except KeyError:
                    raise KeyError("Only one object defined for timestep %f. Two objects have to be present at any given step." % t)
                if np.isnan(input_data.trace[t].objects[o].x) or np.isnan(input_data.trace[t].objects[o].y):
                    raise ValueError("Coordinates x: %f, y: %f are not defined correctly for timestep %f." % (input_data.trace[t].objects[o].x, input_data.trace[t].objects[o].y, t))
        return False

    def _process_qsr_parameters_from_request_parameters(self, req_params, **kwargs):
        """Get the QSR specific parameters from the request parameters.

        :param req_params: Request parameters.
        :type req_params: dict
        :param kwargs: kwargs arguments.
        :return: QSR specific parameters.
        :rtype: dict
        """
        qsr_params = self.__qsr_params_defaults.copy()

        try: # global namespace
            if req_params["dynamic_args"]["for_all_qsrs"]:
                for k, v in req_params["dynamic_args"]["for_all_qsrs"].items():
                    qsr_params[k] = v
        except KeyError:
            pass

        try: # General case
            if req_params["dynamic_args"][self.__global_unique_id]:
                for k, v in req_params["dynamic_args"][self.__global_unique_id].items():
                    qsr_params[k] = v
        except KeyError:
            pass

        try: # Parameters for a specific variant
            if req_params["dynamic_args"][self._unique_id]:
                for k, v in req_params["dynamic_args"][self._unique_id].items():
                    qsr_params[k] = v
        except KeyError:
            pass

        for param in qsr_params:
            if param not in self.__qsr_params_defaults and param not in self._common_dynamic_args:
                raise KeyError("%s is an unknown parameter" % str(param))

        return qsr_params

    def make_world_qsr_trace(self, world_trace, timestamps, qsr_params, req_params, **kwargs):
        """Compute the world QSR trace from the arguments.

        :param world_trace: Input data.
        :type world_trace: :class:`World_Trace <qsrlib_io.world_trace.World_Trace>`
        :param timestamps: List of sorted timestamps of `world_trace`.
        :type timestamps: list
        :param qsr_params: QSR specific parameters passed in `dynamic_args`.
        :type qsr_params: dict
        :param req_params: Request parameters.
        :type req_params: dict
        :param kwargs: kwargs arguments.
        :return: Computed world QSR trace.
        :rtype: :class:`World_QSR_Trace <qsrlib_io.world_qsr_trace.World_QSR_Trace>`
        """
        ret = World_QSR_Trace(qsr_type=self._unique_id)
        qtc_sequence = {}
        for t, tp, tpp in zip(timestamps[2:], timestamps[1:], timestamps):
            world_state_now = world_trace.trace[t]
            world_state_previous = world_trace.trace[tp]
            world_state_previous_previous = world_trace.trace[tpp]

            qsrs_for = self._process_qsrs_for([world_state_previous_previous.objects.keys(),  
                                                world_state_previous.objects.keys(), 
                                                world_state_now.objects.keys()],
                                                req_params["dynamic_args"])

            for o1_name, o2_name in qsrs_for:
                between = str(o1_name) + "," + str(o2_name)
                qtc = np.array([], dtype=int)

                k = [world_state_previous_previous.objects[o1_name].x,
                     world_state_previous_previous.objects[o1_name].y,
                     world_state_previous_previous.objects[o1_name].z,
                     world_state_previous.objects[o1_name].x,
                     world_state_previous.objects[o1_name].y,
                     world_state_previous.objects[o1_name].z,
                     world_state_now.objects[o1_name].x,
                     world_state_now.objects[o1_name].y,
                     world_state_now.objects[o1_name].z]
                l = [world_state_previous_previous.objects[o2_name].x,
                     world_state_previous_previous.objects[o2_name].y,
                     world_state_previous_previous.objects[o2_name].z,
                     world_state_previous.objects[o2_name].x,
                     world_state_previous.objects[o2_name].y,
                     world_state_previous.objects[o2_name].z,
                     world_state_now.objects[o2_name].x,
                     world_state_now.objects[o2_name].y,
                     world_state_now.objects[o2_name].z]
                qtc = self._create_qtc_representation(
                    k,
                    l,
                    qsr_params["quantisation_factor"],
                    qsr_params["angle_quantisation_factor"]
                )

                try:
                    qtc_sequence[between] = np.append(
                        qtc_sequence[between],
                        qtc
                    ).reshape(-1,4)
                except KeyError:
                    qtc_sequence[between] = qtc

        for between, qtc in qtc_sequence.items():
            qtc = qtc if len(qtc.shape) > 1 else [qtc]
            for idx, q in enumerate(qtc):
                qsr = QSR(
                    timestamp=idx+1,
                    between=between,
                    qsr=self.qtc_to_output_format(q)
                )
                ret.add_qsr(qsr, idx+1)

        return ret

    def _postprocess_world_qsr_trace(self, world_qsr_trace, world_trace, world_trace_timestamps, qsr_params, req_params, **kwargs):
        '''
        Just to conform to general interface
        '''
        return world_qsr_trace

    def create_qtc_string(self, qtc):
        return ','.join(map(str, qtc.astype(int))).replace('-1','-').replace('1','+')

    def qtc_to_output_format(self, qtc):
        """Return QTCCS.

        :param qtc: Full QTCC tuple [q1,q2,q4,q5].
        :type qtc: list or tuple
        :return: {"qtccs": "q1,q2,q4,q5"}
        :rtype: dict
        """
        return self._format_qsr(self.create_qtc_string(qtc))