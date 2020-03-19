# -----------------------------------------------------------------------------
# Contributors: Mario Senden mario.senden@maastrichtuniversity.nl
#               Vaishnavi Narayanan vaishnavi.narayanan@maastrichtuniversity.nl
# -----------------------------------------------------------------------------

import numpy as np

'''DECISION MAKING (DM)
      Based on Wong-Wang model of 2006 - DOI: 10.1523/JNEUROSCI.3733-05.2006
This model simulates decision-making between two excitatory populations that
exhibit switching at a certain frequency, i.e., only one population has a high
firing activity at any given time point. The activity of this population drops
while the activity of the other population goes up. This pattern continues as
long as there is an active input.

parameters
----------
gamma           : controls slow rise of NMDA channels (dimensionless)
tau_s           : time constant of fraction of open channels (seconds)
tau_ampa        : time constant of AMPA receptors / noise (seconds)
params          : contains all parameters necessary to set up the model.
  J_within      : within pool connectivity (nA)
  J_between     : between pool connectivity (nA)
  J_ext         : external drive (nA * Hz^-1)
  mu            : unbiased external input
  theta         : threshold input
  I_0           : unspecific background input (nA)
  sigma_noise   : standard deviation of unspecific background input (nA)
  dt            : integration time step (seconds)

properties
----------
R               : instantaneous firing rate

functions
---------
f(x)            : (a * x - b) / (1 - exp(-d  * (a * x - b) ))
  parameters
  ----------
  a             : 270 (VnC)^-1
  b             : 108 Hz
  d             : 0.154 seconds

simulate(time)  : simulate model for "time" seconds
update          : perform numerical integration for single time step
reset           : reset the model
set_coherence(x): set coherence to value x
set_mu(x)       : set mu to value x
'''

class DM:
    def __init__(self, params):
        self.gamma = 0.641
        self.tau_s = 0.1
        self.tau_ampa = 0.002
        self.params = params
        self.coherence = 0.
        self.mu = self.params['mu']
        self.s = np.ones(2) * 0.1
        self.x = np.zeros(2)
        self.r = np.zeros(2)
        self.Inp = np.zeros(2)
        self.W = np.array([[self.params['J_within'], -self.params['J_between']],
                           [-self.params['J_between'], self.params['J_within']]])
        self.I_noise = np.random.randn(2) * self.params['sigma_noise']
        self.dsig = np.sqrt(self.params['dt'] / self.tau_ampa) *\
                    self.params['sigma_noise']

    def f(self, x):
        return (270. * x - 108) / (1. - np.exp(-0.154 * (270. * x - 108.)))

    def update(self):
        I_ext = np.array([self.params['J_ext'] * self.mu *
                          (1. + self.coherence / 100.),
                          self.params['J_ext'] * self.mu *
                          (1. - self.coherence / 100.)])
        self.Inp += self.params['dt'] * ( -self.Inp + (self.params['theta']-self.r)/5. * I_ext )
        I_rec = np.dot(self.W, self.s)
        self.I_noise += self.params['dt'] * (self.params['I_0'] - self.I_noise) /\
                        self.tau_ampa + self.dsig * np.random.randn(2)
        self.x = I_rec + self.Inp + self.I_noise
        self.r = self.f(self.x)
        self.s += self.params['dt'] * (-self.s / self.tau_s + (1. - self.s) *
                             self.gamma * self.r)

    def simulate(self,time):
        t_steps = int(time / self.params['dt']) + 1
        R = np.zeros((2, t_steps))
        for t in range(t_steps):
            self.update()
            R[:, t] = self.r
        return R

    def set_coherence(self, x):
        self.coherence = x

    def set_mu(self, x):
        self.mu = x

    def reset(self):
        self.coherence = 0.
        self.mu = self.params['mu']
        self.s = np.ones(2) * 0.1
        self.x = np.zeros(2)
        self.r = np.zeros(2)
        self.Inp = np.zeros(2)
        self.W = np.array([[self.params['J_within'], -self.params['J_between']],
                           [-self.params['J_between'], self.params['J_within']]])
