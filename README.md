# Visual Target Selection Model

This repository contains the target visual selection model and example code on how to use the model.

The model is based on the decision-making model given by [Wong and Wang (2006)](http://www.jneurosci.org/cgi/doi/10.1523/JNEUROSCI.3733-05.2006). Here, we simulate two mutually inhibiting excitatory populations that exhibit competition with switching, i.e., the two populations switch between high and low firing activity out of phase at a certain frequency. This is useful for visual search paradigms where several salient objects cause competition in the FEF neurons so as to ensure fixation on one of the objects for a few (milli)seconds until another salient object causes the corresponding FEF neuron(s) to 'win'. 

### Model architecture
---
The model consists of two mutually inhibiting excitatory population with recurrent excitatory connections and both receive external and background input.  
![alt text](https://github.com/ccnmaastricht/target_selection/blob/master/images/arch.png)

### Requirements
---
| Package       | Version       | 
|:-------------:|:-------------:| 
| python        | >= 2.7        |
| numpy         | >= 1.8        |
| matplotlib    | >= 1.3        |

The code was created and tested on Linux and MacOS. 

### Executing code
---
To execute example code, run `python usage.py` in a command line interface. 

### Contact
---
For questions, bug reports, and suggestions about this work, please create an [issue](https://github.com/ccnmaastricht/target_selection/issues) in this repository.

