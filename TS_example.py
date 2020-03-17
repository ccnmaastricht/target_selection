# -----------------------------------------------------------------------------
# Distributed under the GNU General Public License.
#
# Contributors: Mario Senden mario.senden@maastrichtuniversity.nl
# -----------------------------------------------------------------------------
# File description:
#
# Simulates target selection in a visual search paradigm with a target (located
# at x = -0.35, y = 0) and a single distractor (located at x = 0.35, y = 0).
# The visual field is the unit square.
# Generates a figure displaying the activity profiles of all movement neurons
# (those engaged in the decision making process).
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams
from models import target_selection

# define additional variables
cm2inch = .394  # inch/cm
t_relax = 100  # ms

# figure setup
rcParams.update({'figure.autolayout': True})

fig_size = np.multiply([17.6, 11.6], cm2inch)
ppi = 1200
face = 'white'
edge = 'white'

fig = pl.figure(facecolor=face, edgecolor=edge, figsize=fig_size)


'''set up model

parameters
-----------------
tau   : time constant of visual neurons (in ms)
Ns    : number of neurons in one dimensions of square grid
params: parameters of target_selection model / movement neurons
    dt         : time step for numerical integration (in ms)
    N          : total number of neurons (Ns^2)
    lambda     : passive decay rate
    sigma_rec  : sigma controlling lateral inhibition (2d Gaussian)
    sigma_noise: sigma controlling Gaussian noise on input (all neurons)
    scaling_rec: scaling of lateral inhibition
    scaling_ff : scaling of feedforward connections (to SG model)
    theta      : input threshold
    rate       : initial rate of each neuron

'''
tau = 20.
Ns = 20
params = {
    'dt': 0.05,
    'N': Ns**2,
    'lambda': 0.0175,
    'sigma_rec': .25,
    'sigma_noise': .2,
    'scaling_rec': .1,
    'scaling_ff': 1.,
    'theta': .33,
    'rate': .0}
ts = target_selection(params)
ts.init_weights()


'''set up experiment

functions
-------------------------------------
gauss     : 2-dimensional isotropic Gaussian

coordinate system
-------------------------------------
x         : x-coordinates
y         : y-coordinates
target    : location and size of target
distractor: location and size of distractor

input & output
-------------------------------------
Im        : input image (reflecting salience of target and distractor)
VN        : rate of visual neurons
MN        : rate of movement neurons

time protocol (in ms)
-----------------------
preStim   : time interval prior to stimulus presentation
Stim      : time interval for which stimulus is presented
postStim  : time interval after stimulus presentation

time vector T
-------------
since neuron activity is plotted as a function of time,
time vector T is pre-calculated based on time protocol
t_start   : start of experiment
t_end     : end of experiment
t_steps   : number of simulated time steps
'''


def gauss(params, X, Y):
    return np.exp(-(np.power(params[0] - X, 2) + np.power(params[1] - Y, 2)) /
                  (2. * np.power(params[2], 2)))


x = (np.linspace(0, params['N'] - 1,
                 params['N']) % Ns) / (Ns - 1) * Ns - 0.5 * Ns
y = np.floor(np.linspace(params['N'] - 1, 0,
                         params['N']) / Ns) / (Ns - 1) * Ns - 0.5 * Ns

target = np.multiply([-0.35, 0., 0.05], Ns)
distractor = np.multiply([0.35, 0., 0.05], Ns)

Im = 0.8 * gauss(target, x, y) + 0.6 * gauss(distractor, x, y)

preStim = 25
Stim = 275
postStim = 50

t_start = 0
t_end = preStim + Stim + postStim
t_steps = int((t_end - t_start) / params['dt']) - 1
T = np.linspace(t_start, t_end, t_steps)


# (state) variables
VN = np.zeros(params['N'])
MN = np.zeros((params['N'], t_steps))

# numerical integration (simple Euler for visual neurons)
dsig_v = np.sqrt(params['dt'] / tau) * params['sigma_noise']
for t in range(1, t_steps):
    s = float(((params['dt'] * t) > preStim) &
              ((params['dt'] * t) <= (preStim + Stim)))

    VN += params['dt'] * (-VN + s * Im) / tau + dsig_v * \
        np.random.randn(params['N'])
    ts.VN = VN
    ts.update()
    MN[:, t] = ts.MN


# plot
pl.plot(T, np.transpose(MN))
pl.xlabel('time (ms)')
pl.ylabel('rate')
pl.ylim([0, 30])
pl.show()
