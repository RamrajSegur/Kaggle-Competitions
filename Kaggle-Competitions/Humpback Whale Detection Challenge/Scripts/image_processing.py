import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Any results you write to the current directory are saved as output.

def resize(image, width, height):

    """
	Return the resized image accoording to given height and width

	Parameters:
	dir : path for the image file
	height: height of resized image
	width: width of resized image

	Return:
	resized image
	"""
    resized_image = cv2.resize(image, (width, height), 0, 0, cv2.INTER_LINEAR)

    return resized_image


def RGB2Gray(img):

    """
	Return the gray-scale image

	Parameters:
	img : image in the form of numpy array

	Return:
	gray-scale image
	"""

    gray_image = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    return gray_image

def Gray2RGB(img):

    """
	Return the gray-scale image

	Parameters:
	img : image in the form of numpy array

	Return:
	RGB Image
	"""

    RGB_image = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    return RGB_image


def conversion_resize(
        color_conversion, filename_with_path,
        resize_ = 'False', width = None, height = None):

    """
	Do the conversion to the image as specified

    Parameters:
    color_conversion: To convert the image from one type to other
                    RGB2Gray : RGB -> Grayscale Conversion
                    Gray2RGB : Grayscale -> RGB
    filename_with_path : name of the image file with the pathname
    resize :  Flag that denotes if the file has to be resized
              Default :  False  or True
    width : Width of the output image required
    height : Height of the output image required

    Return:
    Returns the image in the form of numpy array

	"""
    image = mpimg.imread(filename_with_path)
    channels = len(image.shape)

    if (color_conversion == 'RGB2Gray' and channels == 3):
        image = RGB2Gray(image)

    if (color_conversion == 'Gray2RGB' and channels != 3):
        image = Gray2RGB(image)

    if (resize_ == 'True'):
        image = resize(image, width, height)

    return image
