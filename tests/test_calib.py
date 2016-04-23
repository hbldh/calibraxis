#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`test_standard`
==================

.. module:: test_standard
   :platform: Unix, Windows
   :synopsis: 

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2015-07-04, 12:00

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import pytest


@pytest.fixture(scope='module')
def points_1():
    return np.array([[-4772.38754098, 154.04459016, -204.39081967],
                     [3525.0346179, -68.64924886, -34.54604833],
                     [-658.17681729, -4137.60248854, -140.49377865],
                     [-564.18562092, 4200.29150327, -130.51895425],
                     [-543.18289474, 18.14736842, -4184.43026316],
                     [-696.62532808, 15.70209974, 3910.20734908],
                     [406.65271419, 18.46827992, -4064.61085677],
                     [559.45926413, -3989.69513798, -174.71879106],
                     [597.22629169, -3655.54153041, -1662.83257031],
                     [1519.02616089, -603.82472204, 3290.58469588]])


@pytest.fixture(scope='module')
def points_2():
    return np.array([[-1575.43324607, 58.07787958, -72.69371728],
                     [1189.53102547, -11.92749837, -23.37687786],
                     [-212.62989556, -1369.82898172, -48.73498695],
                     [-183.42717178, 1408.61463096, -33.89745265],
                     [-162.57253886, 23.43005181, -1394.36722798],
                     [-216.76963011, 19.37118754, 1300.13822193],
                     [-809.20208605, 69.1029987, -1251.60104302],
                     [-1244.03955901, -866.0843061, -67.02594034],
                     [-1032.3692107, 811.19178082, 699.69602087],
                     [-538.82617188, -161.6171875, -1337.34895833]])


def test_calibration_points_1(points_1):
    calibrate_accelerometer_with_stored_points(self.test_points_1)
    np.testing.assert_almost_equal(sc._acc_calibration_errors[-1], 0.0, 2)


def test_calibration_points_2(points_2):
    calibrate_accelerometer_with_stored_points(self.test_points_2)
    np.testing.assert_almost_equal(sc._acc_calibration_errors[-1], 0.0, 2)
