from configHelper import getConfig
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import docstring
from PIL import Image
from fits_io import readFitsImg, saveFits, getSubMatrixFileName
from os import path

##### MATRICES FUNCTIONS #####
# helper just to take a quick look at matrices
def matrixDiag(m):
	print "shape : {}".format(m.shape)
	print "min   : {}".format(m.min())
	print "max   : {}".format(m.max())
	print "median: {}".format(np.median(m))
	print "mean  : {}".format(np.mean(m))

def processValue(value):
	isScalar = not np.iterable(value)
	if isScalar:
		value = [value]
	dtype = np.min_scalar_type(value)
	if np.issubdtype(dtype, np.integer) or dtype.type is np.bool_:
		dtype = np.promote_types(dtype, np.float32)
	# ensure data passed in as an ndarray subclass are interpreted as an ndarray
	mask = np.ma.getmask(value)
	data = np.asarray(value)
	result = np.ma.array(data, mask=mask, dtype=dtype, copy=True)
	return result, isScalar

def validMinMax(vmin, vmax):
	return vmin < vmax and vmin > 0

def logNorm(imageData, vmin, vmax, fileName):
	result, isScalar = processValue(imageData)
	result  = np.ma.masked_less_equal(result, 0, copy=False)

	if not validMinMax(vmin, vmax):
		minMaxMsg = "Not valid vmin & vmax for filename {}. Filling matrix with zeros".format(fileName)
		logging.warning(minMaxMsg)

		result.fill(0)
	else:
		resdat	= result.data
		mask	= result.mask
		if mask is np.ma.nomask:
			mask = (resdat <= 0)
		else:
			mask |= resdat <= 0	

		np.copyto(resdat, 1, where=mask)
		np.log(resdat, resdat)

		resdat -= np.log(vmin)
		resdat /= (np.log(vmax) - np.log(vmin))

		result = 255*(resdat - resdat.min()) / (resdat.max() - resdat.min())

	if isScalar:
		scalarMsg = "Matrix data for filename {} is scalar.".format(fileName)
		logging.warning(scalarMsg)	

		result = result[0]

	return result

def saveImgCustom(imageData, fileName, perc):
	plt.imshow(imageData, cmap='gray',
		norm=colors.LogNorm(vmin=np.median(imageData),vmax=np.percentile(imageData,perc)))
	plt.box(False)
	plt.tick_params(
			axis='both', 
			which='both', 
			bottom=False, 
			top=False, 
			labelbottom=False, 
			right=False, 
			left=False, 
			labelleft=False)
	plt.savefig(fileName, bbox_inches='tight', transparent=True, pad_inches=0)
	plt.clf()
	plt.close()
	return fileName

def saveImgPIL(imageData, fileName, perc):
	vmin = 1.2 * np.median(imageData) 
	vmax = np.percentile(imageData, perc)

	normData = logNorm(imageData, vmin, vmax, fileName)

	img = Image.fromarray(normData).convert('RGB')

	img.save(fileName, 'jpeg')

def saveSubMatrix(data, expID, ccd, subMatIndex, perc):
	baseFileName = getSubMatrixFileName(expID, ccd, subMatIndex)
	fitsName = baseFileName+".fits"
	imgName  = baseFileName+".jpeg"

	# check if files already exist
	if path.isfile(fitsName) and path.isfile(imgName):
		foundMsg = "Files {} and {} found. Skipping subImages.".format(fitsName, imgName)
		logging.info(foundMsg)
	else:
		saveFits(data, fitsName)
		saveImgPIL(data, imgName, perc)

def getSubMatricesFromFits(ccdData, expID, ccd, n=6, perc=100, ccdOption='name'):
	subMatrices = []
	# try to split in n submatrices 
	## best behaviour with n=6
	### if not feasible, raise warning and skip this ccd from this expID 
	try:
		left, right = np.hsplit(ccdData, 2)
		if getConfig("Image","ccdByNumOrName") == 'name' and ccd.startswith("F"):
			heightDiv = n/3 # heightDiv = 2
		else:
			heightDiv = n/2 # heightDiv = 3
		
		subMatrices = np.append(np.asarray(np.vsplit(left, heightDiv)),
						np.asarray(np.vsplit(right, heightDiv)), axis=0)

	except:
		failSplitMsg = "Failed to split CCD {} from expID {} into submatrices. Skipping this CCD.".format(ccd, expID)
		logging.warning(failSplitMsg)

	for subMatIndex, subMatrix in enumerate(subMatrices):
		# fits_io.maskFits performs the image clip (border drop) 
		## if masking (keeping the same size), replace "--" with median value
		if not getConfig("Image","maskBorder"):
			subMatrix[subMatrix == "--"] = np.median(subMatrix)
		saveSubMatrix(subMatrix, expID, ccd, subMatIndex, perc)

	return subMatrices
