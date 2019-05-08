# Indroduction
This repository contains the implementation of various experiments made to analyze various compression techniques to analyze deep learning models used in communication systems ([report](https://www.dropbox.com/s/vwbdbb1kgq9af6w/BTP.pdf?dl=0)). The implementation is mostly in PyTorch and partly in Keras. Since we wish to compress networks used in communication, we have made use of such neural networks currently in the literature and have referenced them for the curious reader. To make this repository easier to navigate, we elaborate on the various folders that make up the repository. Note that some of the code (particularly those that analyse and plot diagrams) only execute correctly when the model has already been trained and saved.

# Channel Modulation
The objective here is to estimate the channel (which is AWGN) and using that information, to encode messages into real symbols and decode the received symbols altered by noise such that the Bit Error Rate (BER) is minimized. To analyze the deep learning model, we have implemented ways to plot the constellation diagram of the encoder and decision boundaries learned by the decoder. The work here is based on  [An Introduction to Deep Learning for the Physical Layer](https://ieeexplore.ieee.org/document/8054694)

# Channel without model
Here, following the work of [End-to-End Learning of Communications Systems Without a Channel Model](https://arxiv.org/abs/1804.02276), we implemented a deep reinforcement learning model to do accomplish the same task as in the preceding section but without imposing any prior on the channel. The disadvantage of using this method is that it demands more computationally intensive train times.

# Multiple Transmitters and Receivers
This is a variant of the channel modulation task with multiple transmitters and receivers. Here, not only should the not only should the messages be encoded such that they are protected against channel noise, but also from interference from the other transmitters. The reference literature used is also the same as that of channel modlulation.

# Modulation Recognition
This dataset and basic implementation was taken from [here](https://www.deepsig.io/datasets) and the code was modified to execute in python3. The CNN here takes an input transmission and attempts to classify the type of modulation used in the transmission. This can be used in communication devices to decide whether the received transmission needs to be processed or also decide which decoding strategy to use.

# Knowledge Distillation
Here we modify the code used in the preceding sections to try to distill the knowledge of neural networks to similarly designed networks of smaller sizes. The most successful application of knowledge distillation is in modulation recognition. Knowledge distillation seems to fail for channel modulation as the network used is too small to compress any further.

# Weights Pruning
Another approach to improving the efficiency of neural networks is to remove the weights of the neural network that don't conntribute to the final output and represent the layers using sparse matrices. Here, we empirically analyse the effect of pruning of weights on the performance of the network on communication tasks. We also prune a regularized version of the neural network and compare its performance against the vanilla version of pruning for various values of pruning rates. The code used to prune the unnecessary weights was cloned from [here](https://github.com/wanglouis49/pytorch-weights_pruning)

# Quantization 
While we have not folder a chapter to quantization, there are scripts that investigate quantization in some of the preceding folders. Our implementation does not improve the efficiency of the neural network, rather it analyses the effect of the LSB errors of quantization on network performance. The only exception is the modulation recognition case where we quantized the network using tensorflow-lite which allowed us to measure the inference time of the quantized network in comparison with the unquantized network.
