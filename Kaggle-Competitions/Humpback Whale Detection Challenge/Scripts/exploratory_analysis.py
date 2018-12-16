import pandas as pd #pandas library import
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def description(file):
	"""
	Give the description of the pandas dataframe object
	
	Parameters:
	file - pandas dataframe
	
	Return:
	Print First five rows of the dataframe
	Print The shape of the dataframe
	
	"""
	# Check if the parameter is of pandas dataframe type
	if not isinstance(file,pd.core.frame.DataFrame):
		raise ValueError("The parameter should a pandas object")
	# Print five sample rows
	print("Sample of the file with 5 rows:")
	print("===============================")
	print(file.head())
	print("")
	print("Shape of the file dataframe :" + str(file.shape))
	
def class_frequency_count(file, ID):
	"""
	Give the class and counts in a column of the pandas dataframe 
	
	Parameters:
	file - pandas dataframe
	ID - Column ID of the pandas dataframe as String
	
	Return:
	Class_and_count - classes and frequency counts in the form of pandas series	
	"""
	if not isinstance(ID, str):
		raise ValueError("The ID - parameter should be a string")
	if not isinstance(file,pd.core.frame.DataFrame):
		raise ValueError("The parameter should a pandas object")
	# Store the frequencies of values in the ID column of pandas dataframe
	class_and_count = file[ID].value_counts()
	return class_and_count
	
def class_count_plot(pandas_series):
	"""
	Plot the counts histogram for classes
	
	Parameters:
	pandas_series - pandas core series
	
	Return:
	Print the plot for instances vs classes
	"""
	if not isinstance(pandas_series, pd.core.series.Series):
		raise ValueError("The pandas_series parameter should be pandas.core.series.Series type")
		
	series_counts = pandas_series.value_counts()
	series_counted_sorted = series_counts.sort_index(ascending=True)
	series_counted_sorted_filtered = series_counted_sorted[0:-2]
	plt.plot(series_counted_sorted_filtered)
	plt.xlabel('Number of Instances')
	plt.ylabel('Number of Classes')
	plt.title('Instances vs Classes plot (Except new_whale instance)')
	
def first_ten_image_filenames_of_given_id(file, fileID, columnID):
	"""
	Return the first ten image filenames belong to fileID in the given columnID
	
	Parameters:
	file : pandas dataframe
	fileID: ID of the whale
	columnID: ID of the column to look in the pandas dataframe

	Return:
	list_of_ten_filenames
	"""
	# file check
	if not isinstance(file,pd.core.frame.DataFrame):
		raise ValueError("File is not a pandas dataframe type")
	
	if not isinstance(fileID, str):
		raise ValueError("fileID should be a string")
	
	if not isinstance(columnID, str):
		raise ValueError("columnID should be a string")
		
	names = file.loc[file[columnID]==fileID]
	list_of_ten_filenames = list(names['Image'][0:10])
	
	return list_of_ten_filenames
	
def image_subplot_from_list_filenames_and_dir(image_filenames_list, dir, title):
	"""
	Plots the images for all the files in the filenames list
	
	Parameters:
	image_filenames_list : list of filenames
	dir : directory _address as text
	title : title for the subplot
	
	Return:
	Sub plot all the images for filenames in the list
	"""
	
	plt.figure(figsize=(40,15))

	for n in range(1,len(image_filenames_list)+1):
		columns = len(image_filenames_list)//2
		ax = plt.subplot(2,columns,n)
		ax.set_title(title)
		image = plt.imread(dir+image_filenames_list[n-1])
		plt.imshow(image)
		
def filenames_shape(dataframe, dir_path,columnID):
	"""
	Prepare and return the numpy array with shape of Nx2 
	first column: filenames of the image in dir
	second column: shape of the respective images
	
	Parameters:
	dataframe : pandas dataframe having filenames
	dir : path for the directory having the images 
	columnID : column of the dataframe containing filenames
	
	Return:
	combined : the numpy array with
					first column: filenames of the image in dir
					second column: shape of the respective images
	"""
	if not isinstance(dataframe, pd.core.frame.DataFrame):
		raise ValueError("DataFrame is not a pandas dataframe type")
		
	if not isinstance(columnID, str):
		raise ValueError("columnID should be a string")
	 
	filenames = [] #List to store filenames
	shapes = [] # List to store shapes of the image files
	for index, row in dataframe.iterrows():  # Iterate through each row in dataframe
		filename = row[columnID] # Read the filename stored in columnID
		shape = mpimg.imread(dir_path+filename).shape # Find the shape of the image
		filenames.append(filename) # Append the filename of the image to filenames list
		shapes.append(shape) # Append the shape of the image to shapes list
	shapes_array = np.array(shapes) # Convert the list to array
	filenames_array = np.array(filenames) # Convert the list to array
	combined = np.vstack((filenames_array, shapes_array)).T
	
	return combined
	
def filenames_shape_hist_distribution_plot3D(dataframe,columnID1, columnID2, xrange, yrange):
	"""
	Plot the histogram distribution of number of images for each combination of width and height dimension of images
	Return the width interval and height interval which covers 90% (approx) of the data
	
	Parameters:
	dataframe - pandas dataframe containing the data for the plot
	columnID1 - column ID of the dataframe that can be taken as xaxis data (height)
	columnID2 - column ID of the dataframe that can be taken as yaxis data (width)
	xrange - range of values in the columnID1 (max x value) : min x value is considered as 0
	yrange - range of values in the columnID2 (max y value) : min y value is considered as 0
	
	Return:
	Plot the histogram distribution 
	Print the min and max height and width range that contains 90% of data
	Return the range of  width and height in form of list of tuples [(width_max, width_min),(height_max,height_min)]
	
	"""
	if not isinstance(dataframe, pd.core.frame.DataFrame):
		raise ValueError("DataFrame is not a pandas dataframe type")
	
	if not isinstance(columnID1, str):
		raise ValueError("columnID1 should be a string")
	
	if not isinstance(columnID2, str):
		raise ValueError("columnID2 should be a string")
		
	if not isinstance(xrange, int):
		raise ValueError("xrange should be an integer")
	
	if not isinstance(yrange, int):
		raise ValueError("yrange should be an integer")
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
	# Create the histogram
	hist, xedges, yedges = np.histogram2d(dataframe[columnID1], dataframe[columnID2], bins=20, range=[[0, xrange], [0, yrange]])

	xpos, ypos = np.meshgrid(xedges[:-1] + 1, yedges[:-1] + 1, indexing="ij")
	xpos = xpos.ravel() # Storing the xpositions (heights)
	ypos = ypos.ravel() # Storing the ypositions (widths)
	zpos = 0 # Z-position always zero

	# Construct arrays with the dimensions for the 16 bars.
	dx = dy = 200* np.ones_like(zpos)
	dz = hist.ravel() # Number of images (third dimension) at each position of width and height

	ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='r') # Create 3D bar plot
	# Labeling the axes
	ax.set_xlabel("Height") 
	ax.set_ylabel("Width")
	ax.set_zlabel("Number of Images")
	
	plt.show()
	
	perc_dz = (dz/np.sum(dz))*100 # Normalize the distribution and convert to percentage
	
	width_max = max(xpos[perc_dz>5]) # Maximum width of the range having 90% of the images
	width_min = min(xpos[perc_dz>5]) # Minimum width of the range having 90% of the images
	
	height_max = max(ypos[perc_dz>5]) # Maximum height of the range having 90% of the images
	height_min = min(ypos[perc_dz>5]) # Minimum height of the range having 90% of the images
	
	print(" 90% of images have width in between the range: " + str(width_max) + " and " + str(width_min))
	print(" 90% of images have height in between the range: " + str(height_max) + " and " + str(height_min))
	
	return [(width_max, width_min),(height_max, height_min)]
	
def find_uniq_values_counts(file, ID):
    """
    Find the unique values and their counts along a column in the dataframe
    
    Parameters:
    file - Pandas dataframe
    ID - Column ID to look for unique values
    
    Return:
    dict_uniq_Ids : {unique_value: [unique_identification_number, number of instances of unique value in that column]}
    
    """
        
    dict_unique_Ids = {} # Initialize the dictionary
    unique_file_Ids = file[ID].unique() # Get the unique elements from the column "ID" of the file dataframe
    unique_file_Ids.sort() # Sort the elements  
    class_count = file[ID].value_counts() # Get the count of unique element from the column "ID" of the dataframe
    identification = 1 # Initialize the identification number to be assigned for each unique class
    for unique_Id in unique_file_Ids: 
        number_of_instances = class_count.loc[unique_Id] # Get the number of instances of that 'unique ID'
        dict_unique_Ids[unique_Id] = [identification, number_of_instances] # Storing the key value pair in the dictionary
        identification = identification + 1 # Update the identification number for the next class

    return dict_unique_Ids # Return the dictionary created
