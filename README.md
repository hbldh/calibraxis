# Calibraxis


## Installation

TBD.

## Usage

### Basic use

TDB


### Theory

Calibration of accelerometer is performed using the method described in 
[Frosio, I.; Pedersini, F.; Alberto Borghese, N., 
"Autocalibration of MEMS Accelerometers," 
Instrumentation and Measurement, IEEE Transactions on , 
vol.58, no.6, pp.2034,2041, June 2009]
(http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4655611&isnumber=4919430).

First it prompts the user to position the BerryIMU such that Earth's gravity acts on 
only one of the axes at a time, in both directions. This six point calibration gives a
zero G value and a sensitivity for each axis. At least three more points are needed to
complete the calibration, which can be chose arbitrarily with the only restriction that
the BerryIMU is static. When these have been collected, an optimisation is done to fit
final calibration parameters.
