import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception


def model_name_model_file(model_name):
    """
    Return the corresponding keras fucntion name and model weights filename

    Parameters:
    model_name : Name of the model

    Return:
    Tuple of the form (keras_func_name, model_weights_filename) for the corresponding model name argument
    """
    dict_model = {'VGG16':(VGG16, 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'),
                  'VGG19':(VGG19, 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'),
                  'Inception_V3':(InceptionV3,'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'),
                  'Densenet_121':(DenseNet121,'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'),
                  'Densenet_169':(DenseNet169, 'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'),
                  'Densenet_201':(DenseNet201,'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'),
                  'Inception_ResNet_V2':(InceptionResNetV2, 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'),
                  'MobileNet':(MobileNet,'mobilenet_1_0_224_tf_no_top.h5'),
                  'Nasnet_large':(NASNetLarge,'nasnet_large_no_top.h5'),
                  'Nasnet_mobile':(NASNetMobile,'nasnet_mobile_no_top.h5'),
                  'ResNet50':(ResNet50,'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'),
                  'Xception':(Xception,'xception_weights_tf_dim_ordering_tf_kernels_notop.h5') }

    return dict_model[model_name]

def pretrained_model(input_tensor , model_name , input_shape, trainable = False):
    """Return the pretrained model

    Parameters:
    model_name: name of the pretrained model
                enter one from the list below:
                    1. Densenet_121
                    2. Densenet_169
                    3. Densenet_201
                    4. Inception_ResNet_V2
                    5. Inception_V3
                    6. MobileNet
                    7. Nasnet_large
                    8. Nasnet_mobile
                    9. ResNet50
                    10.VGG16
                    11.VGG19
                    12.Xception
    input_tensor: Tensor input to the pretrained model
    trainable : Boolean input to train or not train the pretrained model

    Return:
    Return the Keras Model Instance
    """
    keras_func = model_name_model_file(model_name)[0] # Get the keras function corresponding to the model name
    weights_filename = model_name_model_file(model_name)[1] # Get the weights filename corresponding to the model name
    model = keras_func(weights='../input/full-keras-pretrained-no-top/'+weights_filename,
                       include_top=False, input_shape = input_shape,input_tensor = input_tensor) # fetch the pretrained model
    model.trainable = trainable # Assign the trainable parameter of the model

    return model
