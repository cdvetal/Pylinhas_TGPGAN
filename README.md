# Pylinhas TGPGAN

This framework uses the Pylinhas renderer to replace the generator component of a Generative Adversarial Network.
Pylinhas is a part of a bigger collection of CMA-ES (Covariance Matrix Adaptation Evolution Strategy) driven renderers available at [lmagoncalo/ES_Engine (github.com)](https://github.com/lmagoncalo/ES_Engine).

## Overview

In Pylinhas an individual is composed of a set of lines. In this framework, for each training step, the CMA-ES algorithm is run for n generations. Using Pylinhas, the individuals of the final population (i.e. individuals resulting from the last generation) are converted to their image phenotypes.

These images will constitute the fake batch that will later be fed to the discriminator, which is a standard convolutional network.

Fitness assessment is then performed as a forward pass on the discriminator network followed by a sigmoid normalization.

This ensures that whatever solutions the generator comes up with, generated solutions will be optimized to fool the network as we are maximizing over the direct discriminator's evaluation.
While individuals in the first training step are randomly initialized by the EA, in subsequent training steps a given portion of the best-fitted individuals are taken from the last population of the previous training step, simulating elitism across training steps.

After retrieving and converting the fake batch, another batch of images is fetched from the original dataset and both batches are passed to the discriminator for weights updating using standard backpropagation and the Adam Optimizer.

## Features

The code provided uses the MNIST dataset. 
This can be changed in the <code># LOAD DATASET</code> section of the <code>pylinhas_tgpgan.py</code> file.

The file <code>config.py</code> provides a set of useful parameters to control evolution. 


## Installation

The required packages are contained in the file "requirements.txt", which you can install with pip:

<code>pip install -r requirements.txt</code>.

If you are having issues, install the <code>pytorch_wavelets</code> as specified in the original repository:
