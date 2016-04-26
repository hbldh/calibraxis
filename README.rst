Calibraxis
==========

.. image:: https://travis-ci.org/hbldh/calibraxis.svg?branch=master
    :target: https://travis-ci.org/hbldh/calibraxis
.. image:: http://img.shields.io/pypi/v/calibraxis.svg
    :target: https://pypi.python.org/pypi/calibraxis/
.. image:: http://img.shields.io/pypi/dm/calibraxis.svg
    :target: https://pypi.python.org/pypi/calibraxis/
.. image:: http://img.shields.io/pypi/l/calibraxis.svg
    :target: https://pypi.python.org/pypi/calibraxis/
.. image:: https://coveralls.io/repos/github/hbldh/calibraxis/badge.svg?branch=master
    :target: https://coveralls.io/github/hbldh/calibraxis?branch=master

An Python/NumPy implementation of the accelerometer calibration method
described in  [#FRO2009]_. This is a Python reimplementation of the
Matlab routine found at [#MLCENTRAL]_.

Installation
------------

.. code:: bash

    $ pip install git+git://github.com/hbldh/calibraxis.git


Usage
-----

.. code-block:: python

    import numpy as np
    from calibraxis import Calibraxis

    c = Calibraxis()
    points = np.array([[-4772.38754098, 154.04459016, -204.39081967],
                       [3525.0346179, -68.64924886, -34.54604833],
                       [-658.17681729, -4137.60248854, -140.49377865],
                       [-564.18562092, 4200.29150327, -130.51895425],
                       [-543.18289474, 18.14736842, -4184.43026316],
                       [-696.62532808, 15.70209974, 3910.20734908],
                       [406.65271419, 18.46827992, -4064.61085677],
                       [559.45926413, -3989.69513798, -174.71879106],
                       [597.22629169, -3655.54153041, -1662.83257031],
                       [1519.02616089, -603.82472204, 3290.58469588]])
    # Add points to calibration object's storage.
    c.add_points(points)
    # Run the calibration parameter optimization.
    c.calibrate_accelerometer()

    # Applying the calibration parameters to the calibration data.
    c.apply(points[0 :])
    >>> (-0.9998374717802275, 0.018413117166568103, -0.015581921828828033)
    c.batch_apply(points)
    >>> [(-0.9998374717802275, 0.018413117166568103, -0.015581921828828033),
         (0.9992961622260429, -0.013214366898928225, 0.02485664909901566),
         (-0.019529368790511807, -0.9999036558762957, -0.0016168646941819831),
         (0.02495705262007455, 0.9997148237911497, 0.002962712686085044),
         (0.01976766176204912, -0.004116860997835083, -0.9989226575863294),
         (-0.01861952448274546, -0.0030340053509653056, 0.9994716286085392),
         (0.2486658848595297, -0.0015217968569550546, -0.9695063568748282),
         (0.2743240898265507, -0.9612564659612206, -0.01023892300189375),
         (0.2845586995260631, -0.8814105592109305, -0.37753891563574526),
         (0.5138552246439876, -0.14594841230046982, 0.8459602354269684)]

Testing
-------

Run tests with:

.. code:: bash

    $ python setup.py test

or with `Pytest <http://pytest.org/latest/>`_:

.. code:: bash

    $ py.test tests.py

Documentation
-------------

TBW.

References
----------

.. [#FRO2009] `Frosio, I.; Pedersini, F.; Alberto Borghese, N.,
    "Autocalibration of MEMS Accelerometers," Instrumentation and Measurement,
    IEEE Transactions on, vol.58, no.6, pp.2034,2041, June 2009
    <http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4655611&isnumber=4919430>`_

.. [#MLCENTRAL] `Matlab File Central: MEMS Accelerometer Calibration using Gauss Newton Method
    <http://se.mathworks.com/matlabcentral/fileexchange/
    33252-mems-accelerometer-calibration-using-gauss-newton-method>`_
