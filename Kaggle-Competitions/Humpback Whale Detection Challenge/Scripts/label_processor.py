import numpy as np

def label_generator(index, num_of_classes):
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
