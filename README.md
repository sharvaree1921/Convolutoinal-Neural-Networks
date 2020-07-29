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
