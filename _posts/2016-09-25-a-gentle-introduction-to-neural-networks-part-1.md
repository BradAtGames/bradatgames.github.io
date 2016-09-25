---
layout: post
comments: true
date: 2016-09-25
title: "A Gentle Introduction To Neural Networks (Part 1)"
abstract: "In the first part of this series I will introduce neural networks and the MNIST dataset"
---

The purpose of this series of articles is to explain what neural networks are and how they work.
I find that the best way to learn is by example and therefore this series will consist of the step by step development of a neural net in [Python](http://www.python.org).
Normally I would make heavy use of a library like [TensorFlow](https://www.tensorflow.org), [Theano](http://deeplearning.net/software/theano/), or [Caffe](http://caffe.berkeleyvision.org).
However, these libraries can abstract away significant amounts of detail that I feel are necessary for understanding neural networks at a deeper level.
To repeat, I am only developing this from scratch as a teaching exercise -- don't reinvent the wheel without reason when developing your own neural nets!

Neural Networks are an incredibly math heavy topic so anyone that expects to be able to follow along with this article should have at least a basic understanding of the following **prerequisites**:

  * The Python programming language
  * Statistics and Probability
  * Linear algebra
  * Calculus
  * A splash of graph theory

OK, so maybe gentle is a bit of a misnomer, but this is a fairly advanced topic. Now, let's get started!

### What is a Neural Network Anyway?
A neural network is a mathematical model for computation that is heavily inspired by biology; in particular the brain.

A neuron in the brain has incoming connections called dendrites that carry a stimulus that may or may not activate the neuron and outgoing connections called axons that carry outgoing stimuli.
A connection between a pair of neurons is called a synapse; and it is through chaining multiple synapses together that we achieve computation.
For our purposes, we represent the incoming state as a tensor and each dendrite as a weight that we conveniently store in another tensor.
Each incoming state is multiplied by the weight of its corresponding edge and then all results are summed and run through an activation function which determines the outgoing state along the outgoing edges.

You may have noticed by now that everything I've described is made up of linear transformations and that a combination of linear transformations will remain linear.
This would cause us to only be able to approximate linear functions for which, we may as well just stick with simple matrix multiplication.
This necessitates the introduction of nonlinearities to the network so that we can extend our computational capabilities beyond those of matrix multiplications.
This is where our activation functions come in; the most commonly cited of which is a sigmoid activation function.

<img class="post-image" title="A Sigmoid Function" src="/public/images/a-gentle-introduction-to-neural-networks/sigmoid.png" />

The sigmoid function is expressed by the equation $$\sigma(x) = \frac{1}{1 + e^{-x}}$$ and has a few desirable properties for our purposes.
  * The sigmoid maps values to a range of (0,1), that is that very large inputs map to values close to 1 and very small inputs map to values close to 0.
  * When we train the network, we're performing convex optimization so it is convenient that sigmoid functions are monotonic.
  * Since eventually we'll be needing to use our good friend calculus on the activation function it is nice that sigmoid functions are easily differentiable.

However sigmoid functions do have some downsides that need to be considered:
  * Their output is not zero-centered. This will have implications during the training of the network -- more on this later.
  * Most importantly sigmoid functions *saturate* at the extremes which leads to what is known as [the vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem).
  
There are several alternative activation functions such as $$tanh$$, and more recently variants of the rectified linear unit (ReLU) have become very popular.
However, sigmoid functions are very common in the literature and therefore we will make use of it *despite their negative aspects* so that you can be better prepared to do your own reading in the future.
[Wikipedia](https://en.wikipedia.org/wiki/Activation_function) has an excellent overview of several alternative activation functions.

### The Goal of Our Network
Our network will be attempting to recognize the popular [MNIST](http://yann.lecun.com/exdb/mnist/) dataset which consists of 28x28 pixel images of hand written numerals 0 through 9.
As a benchmark the current state of the art for recognizing the MNIST dataset is a [convolutional neural network (CNN)](http://cs231n.github.io/convolutional-networks/) that achieves an error rate below .28%.
The simple network we develop here will not be that accurate since we will be using a significantly less advanced network design to teach basic principals.

First let's have a look at a subset of the data using python and matplotlib.

~~~python
'''
The MNIST dataset is broken up into 4 files.
Training Images, Training Labels, Test Images, Test Labels all of which are stored in big endian format.
For the purposes of viewing a few examples we're only concerned with the training images file.
'''
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

training_images = open('../train-images-idx3-ubyte', 'rb')

'''
The first four items in the file are 32-bit integers representing the following metadata:
    A magic number that should be equal to decimal 2051
    The number of images in the file
    The number of rows per image
    The number of columns per image

Let's read them now
'''
magic_number, num_images, rows_per_image, cols_per_image = struct.unpack('>IIII', training_images.read(16))
print('This version of MNIST had the following header: {0}, {1}, {2}, {3}'.format(magic_number, num_images, rows_per_image, cols_per_image))

'''
Lets put the remaining data into a large numpy array and then reshape it into a 3d array of 2d images.
In this case we don't need to worry about endian conversions since it's irrelevant for 8-bit integers.
'''
images = np.fromfile(training_images, dtype=np.uint8, count=-1)
images = images.reshape([num_images, rows_per_image, cols_per_image])

'''
Now that we have an array of images we can finally visualize a subset of them.
Lets go with 100 images.
'''
disp_rows = 10
disp_cols = 10
gs = gridspec.GridSpec(disp_rows, disp_cols, top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
for g, i in zip(gs, range(0,disp_rows*disp_cols)):
    ax = plt.subplot(g)
    ax.imshow(images[i], cmap="Greys_r")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
plt.show()
~~~

Here is the output of our print statement of the header as well as the resulting subset of images:

This version of MNIST had the following header: 2051, 60000, 28, 28

<img class="post-image" title="100 elements from MNIST" src="/public/images/a-gentle-introduction-to-neural-networks/100-elements-from-mnist.png" />

Next time, we'll fully load the MNIST dataset and begin development of the neural net.

A Jupyter Notebook containing the python code for this article can be found [here](https://github.com/brad-rathke/gentle-intro-to-neuralnets-part-1).
