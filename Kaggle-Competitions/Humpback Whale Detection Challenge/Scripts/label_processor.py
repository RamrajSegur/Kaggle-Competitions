import numpy as np

def label_creator(index, num_of_classes):
    """
    Return the one hot encoded array with all zeros but 1 in the index mentioned

    Parameters:
    index : where 1 has to be placed
    num_of_classes : the size of the one hot encoded array (num of columns)

    Returns:
    one_hot_encoded_array : numpy array of dimension 1 x num_of_classes
    """
    one_hot_encoded_array = np.zeros((num_of_classes))

    one_hot_encoded_array[index]=1

    return one_hot_encoded_array

def label_generator(index, num_of_classes):
    """
    Return the one hot encoded array with all zeros but 1 in the index mentioned
    and make it compatible for generator
    
    Parameters:
    index : where 1 has to be placed
    num_of_classes : the size of the one hot encoded array (num of columns)

    Returns:
    one_hot_encoded_array : numpy array of dimension 1 x num_of_classes
    """

    one_hot_encoded_array = label_creator(index, num_of_classes)

    one_hot_encoded_array_result = np.expand_dims(one_hot_encoded_array, axis=0)

    return one_hot_encoded_array_result
