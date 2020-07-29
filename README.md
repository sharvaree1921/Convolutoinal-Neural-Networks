# Convolutoinal-Neural-Networks
## ML Learner's Space course and Deep Learning

Convolutional Neural Networks(CNN) or ConvNet are type of neural networks specifically taking inputs as images.It has the same function to optimize the weights and biases like the neural networks.
Now let's say you were to do the same process described above, but with image pixel values as your input. Considering even a small image, of say 50x50 pixels, number of input variables will be equal to 2500. And lets say you are mapping these input pixel values to a fully connected layer with 1000 elements. Obviously, you cannot directly map the input to a smaller feature space, since that will result in a lot of image features not getting learnt by your model, defeating the purpose of using a DNN. So, the number of parameters for this layer is equal to 2500 x 1000 = 2,500,000. The number of parameters already cross a million and imagine what will happen with more layers. The total number of parameters to be trained will be huge! And remember, this is just for a 50x50 pixel input image. What if you were to use a 1024x1024 sized image? The training time for your model will be even larger this time and hence using fully connected layers is not an efficient approach to deal with image inputs.
This is where the concept of Convolutional Layers comes in, and a network comprised of convolutional layers (along with fully connected and other layers) is known as a Convolutional Neural Network (ConvNet).

We previously saw that each neuron in a hidden layer was connected to every neuron in its previous layer, i.e. the neuron is in **fully connected layer**.The last fully connected layer is the output layer.However in CNNs, one neuron in the hidden layer may be connected to only (maybe) say 10 neurons in the previous layers.This reduces the time and increses efficiency of neural net drastically.

Unlike, regular Neural network, CNN takes the input in 3D(i.e. width,height,depth.Here depth refers to the RGB channels).Every ConvNet takes in a 3D input volume and gives a 3D output volume of neuron activations.

### ConvNet Architecture
CNN Architecture do not contain just simple hidden layers with monotonic work.There are special layers in CNN, to train the model more efficiently.The various types of Layers present in CNN are mentioned as follows with respect to their occurence.
- Input Layer
- Convolutional Layer
- RELU
- Pooling Layer
- Output Layer(Fully Connected Layer)

Convolutional layer and fully connected layer perform transformations that are a function of not only the activations in the input volume, but also of the parameters (the weights and biases of the neurons). On the other hand, the RELU/POOL layers will implement a fixed function. The parameters in the CONV/FC layers will be trained with gradient descent so that the class scores that the ConvNet computes are consistent with the labels in the training set for each image.

Now we eill study about these layers in more detail:

### Convolutional Layer
The Conv layer is the core building block of a Convolutional Network that does most of the computational heavy lifting.
A conv layer basically takes in the input layer(3D volume of Neurons) ,convolve(i.e. take dot product) the filter over this layer and outputs an another 2D layer.

![CNN](https://miro.medium.com/max/1200/1*UN1oMSV2qWdDzjoZILKCRg.png)

Consider this process via an example:
Let our input image be 32x32x3 sized and our filter be 5x5x3.Note that every filter is small spatially (along width and height), but extends through the full depth of the input volume.During the forward pass, we slide (more precisely, convolve) each filter across the width and height of the input volume and compute dot products between the entries of the filter and the input at any position.As we slide the filter, we will obtain a 2D sheet of output values which contains hte answer(i.e. the dot product).Imagine this process in 3D, to get better visualization.Keep in mind that that our filter has same depth as of input volume,that's why we are getting a 2D layer.

Suppose we have 12 filters,then we will get 12 such 2D outputs after convolution.We will stack these activation maps along the depth dimension and produce the output volume.

**Local Connectivity:** When dealing with high-dimensional inputs such as images, as we saw above it is impractical to connect neurons to all neurons in the previous volume. Instead, we will connect each neuron to only a local region of the input volume. The spatial extent of this connectivity is a hyperparameter called the receptive field of the neuron (equivalently this is the filter size). The extent of the connectivity along the depth axis is always equal to the depth of the input volume.
For this same example, each neuron in the convolutional layer(the layer obtained by convoluting a single filter onto the 3D input layer), will have 5x5x3=75 weights(and +1 bias parameter).We can notice that,if convolution didn't happen,each next neuron would be connected to every neuron in previous neuron.However, after performing convolution, each neuron is only connected to its specific locally connected region(in this case the 5x5x3 volume in the input layer).

The learning of weights and biases still remain the saem, as in usual neural network.

Suppose an input volume had size 16x16x20. Then using an example receptive field size of 3x3, every neuron in the Conv Layer would now have a total of 3*3*20 = 180 connections to the input volume. Notice that, again, the connectivity is local in space (e.g. 3x3), but full along the input depth (20).

**Spatial Arrangement:** We have explained the connectivity of each neuron in the Conv Layer to the input volume, but we haven’t yet discussed how many neurons there are in the output volume or how they are arranged. Three hyperparameters control the size of the output volume: the depth, stride and zero-padding. We discuss these next:

1. Depth: it corresponds to the number of filters we would like to use, each learning to look for something different in the input. For example, if the first Convolutional Layer takes as input the raw image, then different neurons along the depth dimension may activate in presence of various oriented edges, or blobs of color. We will refer to a set of neurons that are all looking at the same region of the input as a depth column (some people also prefer the term fibre).
2. Stride:When the stride is 1 then we move the filters one pixel at a time. When the stride is 2 then we move the filters two pixel at a time.
3. Zero Padding:If we pad the input volume with zeros around the border, we retreive the original size of the image.We will see later how zero padding is useful for neural network computations.

We can compute the spatial size of the output volume as a function of the input volume size (W), the receptive field size of the Conv Layer neurons (F), the stride with which they are applied (S), and the amount of zero padding used (P) on the border. You can convince yourself that the correct formula for calculating how many **neurons “fit”** is given by **(W−F+2P)/S+1**. For example for a 7x7 input and a 3x3 filter with stride 1 and pad 0 we would get a 5x5 output. With stride 2 we would get a 3x3 output.

**Parameter Sharing:**. Parameter sharing scheme is used in Convolutional Layers to control the number of parameters. Using the real-world example above, we see that there are 55*55*96 = 290,400 neurons in the first Conv Layer, and each has 11*11*3 = 363 weights and 1 bias. Together, this adds up to 290400 * 364 = 105,705,600 parameters on the first layer of the ConvNet alone. Clearly, this number is very high.

We can reduce these parameters by making an assumption: That if one feature is useful to compute at some spatial position (x,y), then it should also be useful to compute at a different position (x2,y2).n other words, denoting a single 2-dimensional slice of depth as a depth slice (e.g. a volume of size [55x55x96] has 96 depth slices, each of size [55x55]), we are going to constrain the neurons in each depth slice to use the same weights and bias.

With this parameter sharing scheme, the first Conv Layer in our example would now have only 96 unique set of weights (one for each depth slice), for a total of 96*11*11* 3 = 34,848 unique weights, or 34,944 parameters (+96 biases). Alternatively, all 55*55 neurons in each depth slice will now be using the same parameters.

Notice that if all neurons in a single depth slice are using the same weight vector, then the forward pass of the CONV layer can in each depth slice be computed as a convolution of the neuron’s weights with the input volume (Hence the name: **Convolutional Layer**). This is why it is common to refer to the sets of weights as a **filter** (or a kernel), that is convolved with the input.

Note that sometimes the parameter sharing assumption may not make sense. This is especially the case when the input images to a ConvNet have some specific centered structure,for ex. different eye-specific or hair-specific features could (and should) be learned in different spatial locations. In that case it is common to relax the parameter sharing scheme, and instead simply call the layer a Locally-Connected Layer.

For numpy implementation, refer [this article](https://cs231n.github.io/convolutional-networks/#conv)

**Summary**
- Accepts a volume of size W1×H1×D1
- Requires four hyperparameters:
  - Number of filters K
  - their spatial extent F
  - the stride S
  - the amount of zero padding P
- Produces a volume of size W2×H2×D2
   where:
   - W2=(W1−F+2P)/S+1
   - H2=(H1−F+2P)/S+1
   (i.e. width and height are computed equally by symmetry)
   - D2=K

- With parameter sharing, it introduces F⋅F⋅D1 weights per filter, for a total of (F⋅F⋅D1)⋅K weights and K biases.
- In the output volume, the d-th depth slice (of size W2×H2) is the result of performing a valid convolution of the d-th filter over the input volume with a stride of S, and then offset by d-th bias.

A common setting of the hyperparameters is F=3,S=1,P=1

Do check out [Convolutional Demo](https://cs231n.github.io/convolutional-networks/#conv)
Also, if you wish to see the backprop and matrix multiplication concepts of CNN, do check out the same article further.
