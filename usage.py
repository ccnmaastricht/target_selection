# -----------------------------------------------------------------------------
# Distributed under the GNU General Public License.
#
# Contributors: Mario Senden - mario.senden@maastrichtuniversity.nl
#               Vaishnavi Narayanan - vaishnavi.narayanan@maastrichtuniversity.nl
# -----------------------------------------------------------------------------
# File description:
# Test decision-making model for 2 excitatory populations described
# in modelsetup.py
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from modelsetup import DM

### Set up model
params = {'J_within': 0.2609,
          'J_between': 0.0497,
          'J_ext': 5.2e-4,
          'I_0':  0.3255,
          'sigma_noise': 0.01,
          'mu': 0.,
          'theta':15.,
          'dt': 1e-3}
dm = DM(params)

### Simulate till equilibrium
R = dm.simulate(2.)

### Test model with input
dm.set_mu(40.)
dm.set_coherence(10.)
R = np.append(R, dm.simulate(10.), axis = 1)

### Simulate shutdown
dm.set_mu(0.)
dm.set_coherence(0.)
R = np.append(R, dm.simulate(3.), axis = 1)

### Plot decision variables over time
fig0 = plt.figure(0)
plt.title('Firing rates over time')
plt.xlabel('time (ms)')
plt.ylabel('firing rate (Hz)')
plt.plot(np.transpose(R[0]), label='E0', color = 'b')
plt.plot(np.transpose(R[1]), label='E1', color = 'g')
plt.legend()
plt.show()
