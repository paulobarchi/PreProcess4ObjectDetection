import logging
from time import time
from configHelper import *
from fits_io import *
from image import *
from dataFrame import *
import pandas as pd
# from numba import jit
# from multiprocessing import Pool
from multiprocessing import cpu_count
from joblib import Parallel, delayed


##### LOG FUNCTION #####
def createLog():
	logLevel = getConfig("Log","level")
	logFileName = getConfig("Log","logFile")
	myFormat = '[%(asctime)s] [%(levelname)s]\t%(module)s - %(message)s'
	
	if logLevel == 'DEBUG':
		logging.basicConfig(filename=logFileName, level=logging.DEBUG, format=myFormat)
	else:
		logging.basicConfig(filename=logFileName, level=logging.INFO,  format=myFormat)

##### PARALLEL FUNCTION #####
# @jit(nopython=True, parallel=True)
def processCCDs(kwargs, expID, fitsFileName, ccdIndex, ccd):

		ccdData = readFitsImg(fitsFileName, ccdIndex)
		maskedData = maskFits(ccdData, ccd)
		subMatrices	= getSubMatricesFromFits(maskedData, expID, ccd, 
			**{k: v for k, v in kwargs.items() if v is not None})

		for subMatIndex, subMatrix in enumerate(subMatrices):

			sextractorFile = runSextractor(expID, ccd, subMatIndex)
			getImgDataFrame(sextractorFile, subMatrix.shape)

			
##### MAIN FUNCTION #####
def main():
	createLog()
	logging.info('Starting Pre-Processing for Object Detection.')

	total_t0 = time()

	if isOperationSet(operation="subImages"):

		logging.info('Processing subImages.')		

		checkOutputPath()

		files	= getListOfCompressedFiles()
		kwargs	= getImgConfig()

		for file in files:

			expID = removeExtension(file)

			expMsg = 'Processing exposure: {}.'.format(expID)
			logging.info(expMsg)

			fitsFileName = runFunpack(file)
			decam = getCCDNames(fitsFileName)

			Parallel(n_jobs=cpu_count())(delayed(processCCDs)
				(kwargs, expID, fitsFileName, ccdIndex, ccd) 
					for ccdIndex, ccd in enumerate(decam))

			## serial for reference and tests:
			# for ccdIndex, ccd in enumerate(decam):
				# processCCDs(kwargs, expID, fitsFileName, ccdIndex, ccd)


	if isOperationSet(operation="createDataSets"):

		logging.info('Creating and splitting datasets.')

		createDataSets()

	# save total computing time
	totalTime = time() - total_t0
	totalTimeMsg = "Total time: {}s".format(totalTime)
	logging.info(totalTimeMsg)
	logging.info('All done.')

if __name__ == "__main__":
	main()
