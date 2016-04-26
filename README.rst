Calibraxis
==========

.. image:: https://travis-ci.org/hbldh/calibraxis.svg?branch=master
    :target: https://travis-ci.org/hbldh/calibraxis
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

    # Measuring range +/-8g
    c = Calibraxis(8)
    points =np.array([[-4772.38754098, 154.04459016, -204.39081967],
                      [3525.0346179, -68.64924886, -34.54604833],
                      [-658.17681729, -4137.60248854, -140.49377865],
                      [-564.18562092, 4200.29150327, -130.51895425],
                      [-543.18289474, 18.14736842, -4184.43026316],
                      [-696.62532808, 15.70209974, 3910.20734908],
                      [406.65271419, 18.46827992, -4064.61085677],
                      [559.45926413, -3989.69513798, -174.71879106],
                      [597.22629169, -3655.54153041, -1662.83257031],
                      [1519.02616089, -603.82472204, 3290.58469588]])
    c.add_points(points)
    c.calibrate_accelerometer()
    c.apply(points[0 :])
    >>> (-0.9998374717802275, 0.018413117166568103, -0.015581921828828033)

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

.. [#MLCENTRAL] `Matlab File Central <http://se.mathworks.com/matlabcentral/
    fileexchange/33252-mems-accelerometer-calibration-using-gauss-newton-method>`_.
