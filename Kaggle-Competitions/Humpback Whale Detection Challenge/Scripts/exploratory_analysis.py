import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

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