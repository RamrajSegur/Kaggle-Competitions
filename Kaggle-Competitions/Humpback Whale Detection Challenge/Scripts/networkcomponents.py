import keras

def conv2D_layer(
        input, num_of_channels,kernel_size, name, activation = 'relu',
        batch_normalization= True, padding = 'same'):
    """
    Create the convolutional layer

    Parameters:
    input : Input to the convolutional layer
    num_of_channels : Number of Channels in the output of the network
    name : Name of the layer
    activation : type of activation to perform
    batch_normalization : To add batch normalizaiton layer or not
    padding : same or valid padding

    Return :
    layer : Result of the convolutional layer
    """
    layer = keras.layers.Conv2D(num_of_channels, kernel_size, padding = padding,
                                activation = activation)(input)
    if batch_normalization:
        layer = keras.layers.BatchNormalization(name = name+'_BN')(layer)

    return layer

def residual_addition(
        input1, input2, name, activate = True, 
        activation = 'relu'):
    """
    Add the layers element-wise

    Parameters:
    input1 : First input
    input2 : Second input
    activate :  Is it required to activate the output from the addition layer
                (Default : True)
    activation : Method of activation to be used

    Return:
    layer : Result of residual addition of two layers
    """

    layer = keras.layers.Add(name = name)([input1, input2])
    if activate:
        layer = keras.layers.Activation(activation,
                                        name =name + '_activation')(layer)

    return layer
