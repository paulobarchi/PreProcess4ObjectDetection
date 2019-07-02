import logging
from configHelper import getConfig
from sys import exc_info
from os import popen, path, makedirs
# from shlex import split 
from subprocess import Popen, PIPE
from glob import glob
import errno
import numpy as np
import numpy.ma as ma
from astropy.io import fits

##### FILE/DIR FUNCTIONS #####
def checkOutputPath():
	outputPath = getConfig("Paths","outputPath")
	if not path.isdir(outputPath):
		try:
			makedirs(outputPath)
		except OSError as exc:  # Python >2.5 # Python >3 has exist_ok flag
			if exc.errno == errno.EEXIST and path.isdir(outputPath):
				pass
			else:
				raise

def getListOfCompressedFiles():
	inputPath = getConfig("Paths","inputPath")
	if (not inputPath):
		logging.critical("inputPath NOT defined! Exiting.")
		exit()

	singleFile = getConfig("InputFiles","singleFZFile")
	if (singleFile):
		return [inputPath+singleFile]
	return glob(inputPath+"*.fits.fz")

def removeExtension(fileName):
	return fileName.split('/')[-1].split('.')[0]

def getSubMatrixFileName(expID, ccd, subMatIndex):
	separator = "_"
	return getConfig("Paths","outputPath") + separator.join([expID, ccd, str(subMatIndex)])


##### FUNPACK FUNCTION #####
def runFunpack(fileName):
	unpackedFile = fileName.replace('.fz','')
	if path.isfile(unpackedFile):
		foundMsg = "File {} found. Skipping funpack.".format(unpackedFile)
		logging.info(foundMsg)
	else:
		try:
			cmd = "funpack {}".format(fileName)
			popen(cmd).read()
		except:
			failMsg = "Unexpected error: {}".format(exc_info()[0])
			logging.critical(failMsg)
			exit()

	return unpackedFile


##### FITS FUNCTIONS #####
def readFitsImg(fitsFileName, ccdIndex):
	with fits.open(fitsFileName) as fitsFile: 
		return np.array(fitsFile[ccdIndex+1].data, np.float32)

def saveFits(data, fileName):
	hdu = fits.PrimaryHDU(data)
	hdu.writeto(fileName, overwrite=True)

def getCCDNames(fitsFileName):
	ccdOption = getConfig("Image","ccdByNumOrName")
	ccdInfo   = 'CCDNUM' if ccdOption == 'num' else 'EXTNAME'

	try:
		with fits.open(fitsFileName) as fitsFile:
			return [fitsFile[i].header[ccdInfo] for i in range(1, len(fitsFile))]
	except:
		failMsg = "Unexpected error: {}".format(exc_info()[0])
		logging.critical(failMsg)
		exit()

def getFitsDataShape(fitsFileName, ccdIndex):
	with fits.open(fitsFileName) as fitsFile: 
		return np.array(fitsFile[ccdIndex].data, np.float32).shape

##### MASK FUNCTIONS #####
def createMask(ccdData, maskBorder=180):
	h, w = ccdData.shape
	mask = np.ones((h,w))
	mask[maskBorder:-maskBorder,maskBorder:-maskBorder] = 0
	return mask
	
def loadMaskForCCD(fullPathFile):
	return np.loadtxt(fullPathFile)

def getMask(ccdData, ccd, maskPath, maskExtension):
	fullPathMaskFile = maskPath+ccd+"."+maskExtension
	if not path.isfile(fullPathMaskFile):
		maskMsg = "maskPath or CCD {} or extension {} NOT found. Creating mask.".format(ccd, maskExtension)
		logging.info(maskMsg)
		maskBorder = getConfig("Image","maskBorder")
		try:
			maskBorder = int(maskBorder) if maskBorder else maskBorder
			return createMask(ccdData, maskBorder)
		except:
			return createMask(ccdData)
	return loadMaskForCCD(fullPathMaskFile)

def maskFits(ccdData, ccd):
	maskBorder = getConfig("Image","maskBorder")
	if maskBorder:
		maskBorder = int(maskBorder)
		return ccdData[maskBorder:-maskBorder, maskBorder:-maskBorder]
	else:
		mask = getMask(ccdData, ccd, getConfig("Paths","maskPath"),
				getConfig("Image","maskExtension"))
		return ma.masked_array(ccdData, mask=mask, copy=False)


##### SEXTRACTOR FUNCTION #####
def runSextractor(expID, ccd, subMatIndex):
	# Future: fine-tuning?
	## Are sextractor config files alright?! Detection in WCS?  
	baseFileName   = getSubMatrixFileName(expID, ccd, subMatIndex)
	fileName 	   = baseFileName+'.fits'
	sextractorFile = baseFileName+'.cat'
	
	if path.isfile(sextractorFile):
		foundMsg = "File {} found. Skipping sextractor.".format(sextractorFile)
		logging.info(foundMsg)
	
	else:
		if not path.isfile(fileName):
			logging.critical("File {} NOT found! Exiting.".format(fileName))
			exit()
		
		cmd = "sextractor {} -CATALOG_NAME {}".format(fileName, sextractorFile)
		
		try:
			popen(cmd).read()
		except:
			failMsg = "Unexpected error: {}".format(exc_info()[0])
			logging.critical(failMsg)
			exit()

	return sextractorFile
