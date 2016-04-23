#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`method`
==================

Created by hbldh <henrik.blidh@nedomkull.com>
Created on 2016-04-23

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import


def acc_to_ratio(x):
    return (x + self.__mid_v) / self.__max_v


# Accelerometer calibration methods

def calibrate_accelerometer(points, **kwargs):
    """Perform calibration of accelerometer.

    Computes the Zero G levels, Sensitivity, Scale factor Matrix and the
    bias vector of a MEMS accelerometer.

    The procedure exploits the fact that, in static conditions, the
    modulus of the accelerometer output vector matches that of the
    gravity acceleration. The calibration model incorporates the bias
    and scale factor for each axis and the cross-axis symmetrical
    factors. The parameters are computed through Gauss-Newton
    nonlinear optimization.

    The mathematical model used is  A = M(V - B)
    where M and B are scale factor matrix and bias vector respectively.
    Note that the vector V has elements in [0, 1.0] representing the
    normed output value of the sensor. This is calculated by
    `(a_raw + (2 ** 15)) / ((2 ** 16) - 1)`.

    M = [ Mxx Mxy Mxz; Myx Myy Myz; Mzx Mzy Mzz ]
    where  Mxy = Myx; Myz = Mzy; Mxz = Mzx;
    B = [ Bx; By; Bz ]

    The diagonal elements of M represent the scale factors along the
    three axes, whereas the other elements of M are called cross-axis
    factors. These terms allow describing both the axes’ misalignment
    and the crosstalk effect between different channels caused
    by the sensor electronics. In an ideal world, M = 1; B = 0

    First, six points of +/-1 g on each axis is recorded. From these readings
    a first estimate of zero G offset and primary axis scale factors is obtained.
    To find the zero values of your own accelerometer, note the max and
    minimum of the ADC values for each axis and use the following formula:
    Zero_x = (Max_x - Min_x)/2; ...
    To find the Sensitivity use the following formula:
    Sensitivity_x = 2 / (Max_x - Min_x); ...

    Reference:
    Iuri Frosio, Federico Pedersini, N. Alberto Borghese
    "Autocalibration of MEMS Accelerometers"
    IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT, VOL. 58, NO. 6, JUNE 2009

    This is a Python reimplementation of the Matlab routines found at
    `Matlab File Central <http://se.mathworks.com/matlabcentral/fileexchange/
    33252-mems-accelerometer-calibration-using-gauss-newton-method>`_.

    """
    _acc_calibration_points = np.array(points)
    _perform_accelerometer_calibration_optimisation(
        self.acc_to_ratio(_acc_calibration_points))


def calibrate_accelerometer_with_stored_points(points):
    """Perform calibration of accelerometer with stored points.

    :param points: Calibration points recorded earlier.
    :type points: :py:class:`numpy.ndarray`

    """
    if self.acc_scale_factor_matrix is not None:
        raise PyBerryIMUError('This object has already been calibrated!')

    self._acc_zero_g = np.zeros((3, ), 'float')
    self._acc_sensitivity = np.zeros((3, ), 'float')
    for index in six.moves.range(3):
        this_axis_points = []
        for side in [0, 1]:
            this_axis_points.append(self.acc_to_ratio(points[index * 2 + side, index]))

        v_max, v_min = max(this_axis_points), min(this_axis_points)
        self._acc_zero_g[index] = (v_max + v_min) / 2
        v_max = v_max - self._acc_zero_g[index]
        v_min = v_min - self._acc_zero_g[index]
        self._acc_sensitivity[index] = 2 / (v_max - v_min)
    points = self.acc_to_ratio(np.array(points))
    self._perform_accelerometer_calibration_optimisation(points)


def do_six_point_one_g_calibration():
    """Perform six recording of +/- 1g on each accelerometer axis.

    :return: List of points added.
    :rtype: list

    """
    points = []
    self._acc_zero_g = np.zeros((3, ), 'float')
    self._acc_sensitivity = np.zeros((3, ), 'float')

    # Method for polling until desired axis is oriented as requested.
    def _wait_for_compliance():
        keep_waiting = 10
        while keep_waiting > 0:
            a = client.read_accelerometer()
            norm_a = np.linalg.norm(a)
            norm_diff = np.abs(np.abs(a[index]) - norm_a) / norm_a

            if norm_diff < 0.05 and cmp(a[index], 0) == side:
                keep_waiting -= 1
            else:
                keep_waiting = 10
            time.sleep(0.1)

    axes_names = ['x', 'y', 'z']
    for index in six.moves.range(3):
        this_axis_points = []
        for side in [-1, 1]:
            print('Position BerryIMU {0} axis {1}...'.format(
                axes_names[index], 'downwards' if side < 0 else 'upwards'))
            _wait_for_compliance()
            raw_input('Correct orientation. Start calibration of sensor {0} '
                      'axis {1} ({2}) by pressing Enter.'.format(axes_names[index],
                                                                 'downwards' if side < 0 else 'upwards',
                                                                 client.read_accelerometer()))
            acc_values = []
            t = time.time()
            while (time.time() - t) < 5:
                acc_values.append(client.read_accelerometer())

            points.append(np.mean(acc_values, axis=0).tolist())
            this_axis_points.append(self.acc_to_ratio(points[-1][index]))

        v_max, v_min = max(this_axis_points), min(this_axis_points)
        self._acc_zero_g[index] = (v_max + v_min) / 2
        v_max = v_max - self._acc_zero_g[index]
        v_min = v_min - self._acc_zero_g[index]
        self._acc_sensitivity[index] = 2 / (v_max - v_min)

    return points


def add_additional_accelerometer_points(point):
    """Add more calibration points.

    Six calibration points have already been recorded in the six
    direction zero G/sensitivity part of the calibration. At least three
    more has to be added to be able to perform optimisation for scale
    factors and bias.

    :return: List of points added.
    :rtype: list

    """
    points = []
    while True:
        ch = raw_input('At least {0} more points are required. '
                       'Add another calibration point? (y / n) '.format(max([3 - len(points), 0])))
        if ch == 'y':
            raw_input('Make sure sensor is static and then start gathering data by pressing Enter.')
            acc_values = []
            t = time.time()
            while (time.time() - t) < 5:
                acc_values.append(client.read_accelerometer())
            points.append(np.mean(acc_values, axis=0).tolist())
        elif ch == 'n':
            break
        else:
            pass
    return points


def perform_accelerometer_calibration_optimisation(points):
        """Perform the Gauss-Newton optimisation for parameters."""
        nbr_points = len(points)
        if nbr_points < 9:
            raise ValueError('Need at least 9 measurements for the calibration procedure!')

        # Optimisation error function.
        def error_function(M_mat, b_vec, y):
            return np.sum((M_mat.dot((y - b_vec)) ** 2)) - 1

        # Method for calculating the Jacobian.
        def _jacobian(M_mat, b_vec, point):
            # TODO: Clean up Jacobian calculation code. Make it more concise.
            jac = np.zeros((9, ), 'float')

            jac[0] = 2 * (b_vec[0] - point[0]) * (
                M_mat[0, 0] * (b_vec[0] - point[0]) + M_mat[0, 1] * (b_vec[1] - point[1]) + M_mat[0, 2] * (
                    b_vec[2] - point[2]))
            jac[1] = 2 * (b_vec[1] - point[1]) * (
                M_mat[0, 0] * (b_vec[0] - point[0]) + M_mat[0, 1] * (b_vec[1] - point[1]) + M_mat[0, 2] * (
                    b_vec[2] - point[2])) + 2 * (b_vec[0] - point[0]) * (
                M_mat[0, 1] * (b_vec[0] - point[0]) + M_mat[1, 1] * (b_vec[1] - point[1]) + M_mat[1, 2] * (
                    b_vec[2] - point[2]))
            jac[2] = 2 * (b_vec[0] - point[0]) * (
                M_mat[0, 2] * (b_vec[0] - point[0]) + M_mat[1, 2] * (b_vec[1] - point[1]) + M_mat[2, 2] * (
                    b_vec[2] - point[2])) + 2 * (b_vec[2] - point[2]) * (
                M_mat[0, 0] * (b_vec[0] - point[0]) + M_mat[0, 1] * (b_vec[1] - point[1]) + M_mat[0, 2] * (
                    b_vec[2] - point[2]))
            jac[3] = 2 * (b_vec[1] - point[1]) * (
                M_mat[0, 1] * (b_vec[0] - point[0]) + M_mat[1, 1] * (b_vec[1] - point[1]) + M_mat[1, 2] * (
                    b_vec[2] - point[2]))
            jac[4] = 2 * (b_vec[1] - point[1]) * (
                M_mat[0, 2] * (b_vec[0] - point[0]) + M_mat[1, 2] * (b_vec[1] - point[1]) + M_mat[2, 2] * (
                    b_vec[2] - point[2])) + 2 * (b_vec[2] - point[2]) * (
                M_mat[0, 1] * (b_vec[0] - point[0]) + M_mat[1, 1] * (b_vec[1] - point[1]) + M_mat[1, 2] * (
                    b_vec[2] - point[2]))
            jac[5] = 2 * (b_vec[2] - point[2]) * (
                M_mat[0, 2] * (b_vec[0] - point[0]) + M_mat[1, 2] * (b_vec[1] - point[1]) + M_mat[2, 2] * (
                    b_vec[2] - point[2]))
            jac[6] = 2 * M_mat[0, 0] * (
                M_mat[0, 0] * (b_vec[0] - point[0]) + M_mat[0, 1] * (b_vec[1] - point[1]) + M_mat[0, 2] * (
                    b_vec[2] - point[2])) + 2 * M_mat[0, 1] * (
                M_mat[0, 1] * (b_vec[0] - point[0]) + M_mat[1, 1] * (b_vec[1] - point[1]) + M_mat[1, 2] * (
                    b_vec[2] - point[2])) + 2 * M_mat[0, 2] * (
                M_mat[0, 2] * (b_vec[0] - point[0]) + M_mat[1, 2] * (b_vec[1] - point[1]) + M_mat[2, 2] * (
                    b_vec[2] - point[2]))
            jac[7] = 2 * M_mat[0, 1] * (
                M_mat[0, 0] * (b_vec[0] - point[0]) + M_mat[0, 1] * (b_vec[1] - point[1]) + M_mat[0, 2] * (
                    b_vec[2] - point[2])) + 2 * M_mat[1, 1] * (
                M_mat[0, 1] * (b_vec[0] - point[0]) + M_mat[1, 1] * (b_vec[1] - point[1]) + M_mat[1, 2] * (
                    b_vec[2] - point[2])) + 2 * M_mat[1, 2] * (
                M_mat[0, 2] * (b_vec[0] - point[0]) + M_mat[1, 2] * (b_vec[1] - point[1]) + M_mat[2, 2] * (
                    b_vec[2] - point[2]))
            jac[8] = 2 * M_mat[0, 2] * (
                M_mat[0, 0] * (b_vec[0] - point[0]) + M_mat[0, 1] * (b_vec[1] - point[1]) + M_mat[0, 2] * (
                    b_vec[2] - point[2])) + 2 * M_mat[1, 2] * (
                M_mat[0, 1] * (b_vec[0] - point[0]) + M_mat[1, 1] * (b_vec[1] - point[1]) + M_mat[1, 2] * (
                    b_vec[2] - point[2])) + 2 * M_mat[2, 2] * (
                M_mat[0, 2] * (b_vec[0] - point[0]) + M_mat[1, 2] * (b_vec[1] - point[1]) + M_mat[2, 2] * (
                    b_vec[2] - point[2]))

            return jac

        # Convenience method for moving between optimisation vector and correct lin.alg. formulation.
        def optvec_to_M_and_b(v):
            return np.array([[v[0], v[1], v[2]], [v[1], v[3], v[4]], [v[2], v[4], v[5]]]), v[6:].copy()

        gain = 1  # Damping Gain - Start with 1
        damping = 0.01    # Damping parameter - has to be less than 1.
        tolerance = 1e-12
        R_prior = 100000
        self._acc_calibration_errors = []
        nbr_iterations = 200

        # Initial Guess values of M and b.
        x = np.array([self._acc_sensitivity[0], 0.0, 0.0,
                      self._acc_sensitivity[1], 0.0, self._acc_sensitivity[2],
                      self._acc_zero_g[0], self._acc_zero_g[1], self._acc_zero_g[2]])
        last_x = x.copy()
        # Residuals vector
        R = np.zeros((nbr_points, ), 'float')
        # Jacobian matrix
        J = np.zeros((nbr_points, 9), 'float')

        for n in six.moves.range(nbr_iterations):
            # Calculate the Jacobian at every iteration.
            M, b = optvec_to_M_and_b(x)            
            for i in six.moves.range(nbr_points):
                R[i] = error_function(M, b, points[i, :])
                J[i, :] = _jacobian(M, b, points[i, :])

            # Calculate Hessian, Gain matrix and apply it to solution vector.
            H = np.linalg.inv(J.T.dot(J))
            D = J.T.dot(R).T
            x -= gain * (D.dot(H)).T
            R_post = np.linalg.norm(R)
            if self._verbose:
                print("{0}: {1} ({2})".format(n, R_post, ", ".join(["{0:0.9g}".format(v) for v in x])))

            # This is to make sure that the error is decreasing with every iteration.
            if R_post <= R_prior:
                gain -= damping * gain
            else:
                gain *= damping

            # Iterations are stopped when the following convergence criteria is satisfied.
            if abs(max(2 * (x - last_x) / (x + last_x))) <= tolerance:
                self.acc_scale_factor_matrix, self.acc_bias_vector = optvec_to_M_and_b(x)
                break

            last_x = x.copy()
            R_prior = R_post
            self._acc_calibration_errors.append(R_post)
