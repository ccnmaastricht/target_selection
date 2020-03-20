# -----------------------------------------------------------------------------
# Contributors: Mario Senden mario.senden@maastrichtuniversity.nl
# -----------------------------------------------------------------------------

import numpy as np

'''SACCADE GENERATOR (SG; 14333)

parameters
----------
tau    : time constant (in ms)
params : contains all parameters necessary to set up the model.
  dt   : integration time step
  rate : initial firing rate

integration
-----------
Integrators of Exponential Euler (EE) method for each neuron.
Note that the tonic neuron cannot be integrated using the EE method
and is integrated using Forward Euler instead.
lin    : linear part of numerical integrator
nonlin : nonlinear part of numerical integrator

properties
----------
L     : instantaneous rate of LLBN
dL    : change in rate of LLBN
E     : instantaneous rate of EBN
dE    : change in rate of EBN
B     : instantaneous rate of IBN
dB    : change in rate of IBN
T     : instantaneous rate of TN
dT    : change in rate of TN
P     : instantaneous rate of OPN
dP    : change in rate of OPN
I     : input

functions
---------
r     : threshold linear function used for rectifying rates
g     : sigmoidal gain function
update: perform numerical integration for single time step
reset : reset the model
'''


class saccade_generator:

    def __init__(self, params):
        self.params = params
        self.tau = 50.
        self.llbn = {
            'lin': -1.3 * self.params['dt'] / self.tau,
            'nonlin': -1.0 / 1.3 * np.expm1(-1.3 * self.params['dt'] / self.tau)}
        self.ebn = {
            'lin': -3.5 * self.params['dt'] / self.tau,
            'nonlin': -1.0 / 3.5 * np.expm1(-3.5 * self.params['dt'] / self.tau)}
        self.ibn = {
            'lin': -2.4 * self.params['dt'] / self.tau,
            'nonlin': -1.0 / 2.4 * np.expm1(-2.4 * self.params['dt'] / self.tau)}
        self.opn = {
            'lin': -0.2 * self.params['dt'] / self.tau,
            'nonlin': -1.0 / 0.2 * np.expm1(-0.2 * self.params['dt'] / self.tau)}
        self.tn = {
            'nonlin': self.params['dt'] / self.tau}

        self.L = np.full((4), self.params['llbn']['rate'])
        self.dL = np.zeros(4)
        self.E = np.full((4), self.params['ebn']['rate'])
        self.dE = np.zeros(4)
        self.B = np.full((4), self.params['ibn']['rate'])
        self.dB = np.zeros(4)
        self.T = np.full((4), self.params['tn']['rate'])
        self.dT = np.zeros(4)
        self.P = self.params['opn']['rate']
        self.dP = 0.
        self.I = np.zeros(4)

    def r(self, x):
        return np.maximum(x, 0.)

    def g(self, x):
        return pow(x, 4) / (pow(0.1, 4) + pow(x, 4))

    def update(self):
        k = [1, 0, 3, 2]
        for i in range(4):
            self.dL[i] = (
                self.llbn['lin'] * self.L[i] +
                self.llbn['nonlin'] * (self.I[i] - 2. * self.B[i]))
            self.dE[i] = (
                self.ebn['lin'] * self.E[i] +
                self.ebn['nonlin'] * ((2. - self.E[i]) * (5. * self.L[i] + 1.) -
                                      (self.E[i] + 1.) * (10. * self.L[k[i]] +
                                                          20. * self.g(self.P))))
            self.dB[i] = (
                self.ibn['lin'] * self.B[i] +
                self.ibn['nonlin'] * (3. * self.E[i]))
            self.dT[i] = (
                self.tn['nonlin'] * 0.1 * (self.E[i] - self.E[k[i]]))

        self.dP = (
            self.opn['lin'] * self.P +
            self.opn['nonlin'] * (1.2 * (1. - self.P) -
                                  3.5 * (self.P + 0.4) * sum(self.g(self.L))))

        self.L = self.r(self.L + self.dL)
        self.E = self.r(self.E + self.dE)
        self.B = self.r(self.B + self.dB)
        self.T += self.dT
        self.P = self.r(self.P + self.dP)

    def reset(self):
        self.L = np.full((4), self.params['llbn']['rate'])
        self.dL = np.zeros(4)
        self.E = np.full((4), self.params['ebn']['rate'])
        self.dE = np.zeros(4)
        self.B = np.full((4), self.params['ibn']['rate'])
        self.dB = np.zeros(4)
        self.T = np.full((4), self.params['tn']['rate'])
        self.dT = np.zeros(4)
        self.P = self.params['opn']['rate']
        self.dP = 0.
        self.I = np.zeros(4)


'''TARGET SELECTION (TS; 14331)

parameters
----------
tau         : time constant (in ms)
params      : contains all parameters necessary to set up the model.
  dt        : integration time step
  rate      : initial firing rate

integration
-----------
prop        : Exponential Euler (EE) propagator
 lin        : linear part of numerical integrator
 nonlin     : nonlinear part of numerical integrator

properties
----------
MN          : instantaneous rate of movement neurons
W_rec       : lateral connectivity profile (inhibition)
W_ff        : feedforward connectivity profile
out         : output to SG
VN          : input from visual neurons

functions
---------
r           : threshold linear function used for rectifying rates
gauss       : 2-dimensional isotropic Gaussian
init_weights: initialize lateral and feedforward weights
update      : perform numerical integration for single time step
reset       : reset the model
'''


class target_selection:

    def __init__(self, params):
        self.params = params
        self.tau = 1.
        self.prop = {
            'lin': -self.params['lambda'] *
            self.params['dt'] / self.tau,
            'nonlin': -1.0 / self.params['lambda'] *
            np.expm1(-self.params['lambda'] *
                     self.params['dt'] / self.tau),
            'noise': np.sqrt(-0.5 / self.params['lambda'] *
                             np.expm1(-2. * self.params['lambda'] *
                                      self.params['dt'] / self.tau))}
        self.MN = np.full((self.params['N']), self.params['rate'])
        self.W_rec = np.zeros([self.params['N'], self.params['N']])
        self.W_ff = np.zeros([4, self.params['N']])
        self.out = np.zeros(4)
        self.VN = []

    def r(self, x):
        return np.maximum(x, 0.)

    def gauss(self, x, y, X, Y, sigma):
        return np.exp(-(np.power(x - X, 2) + np.power(y - Y, 2)
                        ) / (2. * np.power(sigma, 2)))

    def init_weights(self):
        pi = np.pi
        Ns = int(np.sqrt(self.params['N']))
        r = np.linspace(-1., 1., Ns)
        X, Y = np.meshgrid(r, r)
        X = np.reshape(X, [self.params['N'], ])
        Y = np.reshape(Y, [self.params['N'], ])
        A = np.arctan2(Y, X)

        for i in range(self.params['N']):
            self.W_rec[:, i] = self.params['scaling_rec'] * \
                self.gauss(X[i], Y[i], X, Y, self.params['sigma_rec'])
            self.W_rec[i, i] = 0.

        ID = (A >= 0.) & (A < pi / 2.)
        self.W_ff[1,
                  ID] = self.params['scaling_ff'] * X[ID]
        self.W_ff[3,
                  ID] = self.params['scaling_ff'] * Y[ID]

        ID = (A >= pi / 2.) & (A < pi)
        self.W_ff[0,
                  ID] = self.params['scaling_ff'] * np.abs(X[ID])
        self.W_ff[3,
                  ID] = self.params['scaling_ff'] * Y[ID]

        ID = (A >= -pi / 2.) & (A < 0)
        self.W_ff[1,
                  ID] = self.params['scaling_ff'] * X[ID]
        self.W_ff[2,
                  ID] = self.params['scaling_ff'] * np.abs(Y[ID])

        ID = (A >= -pi) & (A < -pi / 2.)
        self.W_ff[0,
                  ID] = self.params['scaling_ff'] * np.abs(X[ID])
        self.W_ff[2,
                  ID] = self.params['scaling_ff'] * np.abs(Y[ID])

    def update(self):
        dMN = (self.prop['lin'] *
               self.MN +
               self.prop['nonlin'] *
               (self.VN -
                self.params['theta'] -
                self.r(np.dot(self.W_rec, self.MN))) +
               self.prop['noise'] *
               self.params['sigma_noise'] *
               np.random.randn(self.params['N']))
        self.MN = self.MN + dMN
        self.out = np.dot(self.W_ff, self.MN)

    def reset(self):
        self.MN = np.full((self.params['N']), self.params['rate'])
        self.out = np.zeros(4)
        self.VN = []
