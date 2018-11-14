## @package Preprocessor
# The preprocessor converts a floor plan in PDF format into a jpeg file ready for input to a neural network.

class Preprocessor:
	## The constructor 
	def __init__(N, M, k):
		self.N = N ## The number of rows in a floor plan image after preprocessing
		self.M = M ## The number of columns in a floor plan image after preprocessing
		self.k = k ## The number of dimensions retained after PCA
	
	## The Preprocessor layer's transfer function
	#
	# The preprocessor layer accepts a raw floor plan image as input; the raw image 
	# contains both the floor plan image and the corresponding legend. The Preprocessor's
	# Input function first isolates the floor plan image from the legend via OCR, then 
	# applies PCA, retaining the top self.k dimensions, for dimensionality reduction.
	# 
	# @param [in] PDFFilename the name of a PDF file that contains a raw floor plan image; 
	# this raw input image contains both the floor plan and its corresponding legend.
	# @return A preprocessed, NxM image of the floor plan, with the legend removed
	def Input(self, PDFFilename):
		# TODO: OCR to isolate the floor plan image from the legend
		
		# TODO: Scale the image to dimensions NxM, padding with black pixels if necessary
		
		# TODO: Perform PCA, retaining the top self.k dimensions
		
		# TODO: Return an NxM preprocessed image
		pass