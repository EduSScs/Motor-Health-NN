Electric Motor Health Classification Using Neural Networks
This repository contains a Python implementation of a three-layer feed-forward Neural Network (NN) designed to discriminate between good and bad electric motors based on their current draw.

Data
The data used in this project originates from electric motors drawing current from their power source. Under normal circumstances, a good motor's current draw will have a smooth acceleration. However, a bad motor will have a noisy feedback.

The raw data consists of time data sampled for 3 seconds at 10,000 Hz, generating 30,000 data points. This raw time-domain data isn't very useful in its original format, so it has been converted into frequency-domain data by performing a Fast Fourier Transform (FFT). Frequency bins were defined, and sine and cosine coefficients were calculated. These coefficients are used to define the harmonic frequency amplitude in each frequency bin, essentially creating a frequency spectrum from the time data collected.

We have created several data files with varying number of frequency bins (16, 25, 32, 64, 150, 1000), each offering a different resolution in the FFT. Each file contains 53 samples, with 19 bad and the rest good motors. Each sample starts with a class label (0 for good, 1 for bad), followed by the bin data.

The Project
The main goal of this project is to create a 3-layer feed-forward Neural Network capable of discriminating between the two types of motors based on the frequency spectra derived from the motors' current draws.

Key stages of the project include:

Data Selection: Choose individual data files or combine several of these through concatenation. The choice of data file will determine the input layer of the neural network.

Neural Network Implementation: Construct a 3-layer feed-forward neural network, using the chosen data file or combination of files as input. The output layer could be a single neuron (0 for good, 1 for bad) or two neurons (one for each class).

Training: Initially, the entire data file is used to train the neural network, essentially using it as a memory to debug the code until it is confirmed to be working correctly. Standard Backpropagation (BP) is used for the training process.

This project provides a practical application of signal processing and machine learning techniques for the purpose of equipment health monitoring, with potential uses in predictive maintenance and fault detection.
