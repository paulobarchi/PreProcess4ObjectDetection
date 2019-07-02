import logging
from configHelper import getConfig
from os import popen, path, makedirs
from astropy.io import ascii
import numpy as np
import pandas as pd
from glob import glob
# from fits_io import removeExtension

SEXTRACTOR_COLS = ['XMIN_IMAGE', 'YMIN_IMAGE', 'XMAX_IMAGE', 'YMAX_IMAGE']
TENSORFLOW_COLS = ['filename', 'width', 'height', 'class', 
						'xmin', 'ymin', 'xmax', 'ymax']
LABEL			= 'obj'

##### FILES & DIRECTORIES - FUNCTIONS #####

def createDataDir():
	outputPath = getConfig("Training","parentDirWithCSVs")
	outDataDir = outputPath+"data/"
	if not path.isdir(outDataDir):
		makedirs(outDataDir)
	return outDataDir

def createLabelMap(outDataDir):
	labelMapFileName = getConfig("Training","labelMapFile")
	if labelMapFileName:
		labelMapFileName = outDataDir + labelMapFileName 
	else:
		labelMapFileName = outDataDir + 'labelmap.pbtxt'

	content = "item {{\n  id: 1\n  name: '{0}'\n}}".format(LABEL)
	with open(labelMapFileName, "w") as labelMapFile:
		labelMapFile.write(content)

##### DATAFRAME FUNCTIONS #####
def createTensorflowDF():
	return pd.DataFrame(columns=TENSORFLOW_COLS)

# Function to build up 1 dataframe for each image exactly like sextractor cat 
def getSextractorDataFrame(sextractorFile):
	dfFile = sextractorFile.replace('.cat','.csv')
	if path.isfile(dfFile):
		foundMsg = "File {} found. Reading from it.".format(dfFile)
		df = pd.read_csv(dfFile)
		logging.info(foundMsg)
	else:
		if not path.isfile(sextractorFile):
			logging.critical("File {} NOT found! Exiting.".format(sextractorFile))
			exit()
		cat = ascii.read(sextractorFile, format='sextractor')
		df  = cat.to_pandas()
		df.to_csv(dfFile, index=False)

	return df

# Function to build up 1 dataframe for each image already with info for tensorflow
def getImgDataFrame(sextractorFile, imgSize):
	dfFile = sextractorFile.replace('.cat','.csv')
	
	if path.isfile(dfFile):
		foundMsg = "File {} found. Reading from it.".format(dfFile)
		df = pd.read_csv(dfFile)
		logging.info(foundMsg)
	
	else:
		if not path.isfile(sextractorFile):
			logging.critical("File {} NOT found! Exiting.".format(sextractorFile))
			exit()
		
		cat = ascii.read(sextractorFile, format='sextractor')
		sextractorDF = cat.to_pandas()

		catMsg  = "Cat file {} has len(cat) {}".format(sextractorFile, len(cat))
		logging.debug(catMsg)

		catDFMsg  = "DF from cat file {} has len(df) {}".format(sextractorFile, len(sextractorDF))
		logging.debug(catDFMsg)

		# imgFile = removeExtension(sextractorFile)+'.jpeg'
		imgFile = sextractorFile.replace('.cat','.jpeg')
		n = len(sextractorDF)

		dfDict = dict(zip(TENSORFLOW_COLS, 
					[np.repeat(imgFile, n, axis=0), 
					 np.repeat(imgSize[1], n, axis=0), # width
					 np.repeat(imgSize[0], n, axis=0), # height
					 np.repeat(LABEL, n, axis=0), 
					 sextractorDF[SEXTRACTOR_COLS[0]], # XMIN_IMAGE
					 sextractorDF[SEXTRACTOR_COLS[1]], # YMIN_IMAGE
					 sextractorDF[SEXTRACTOR_COLS[2]], # XMAX_IMAGE
					 sextractorDF[SEXTRACTOR_COLS[3]]  # YMAX_IMAGE
					 ]))

		tfDF = pd.DataFrame(dfDict, columns=TENSORFLOW_COLS)

		tfDFMsg  = "Dataframe file {} has len(df) {}".format(dfFile, len(tfDF))
		logging.debug(tfDFMsg)

		tfDF.to_csv(dfFile, index=False)

# Function to add objs from one image to total dataframe
def addObjsToDF(sextractorFile, imgSize, totalDF):
	dfFile	= sextractorFile.replace('.cat','.csv')
	imageDF = getImgDataFrame(sextractorFile)
	# imgFile = removeExtension(sextractorFile)+'.jpeg'
	imgFile = sextractorFile.replace('.cat','.jpeg')

	for index, row in imageDF.iterrows():
		newRow  = dict(zip(TENSORFLOW_COLS, 
					[imgFile, 
					 imgSize[1], # width
					 imgSize[0], # height
					 LABEL,
					 row[SEXTRACTOR_COLS[0]], # XMIN_IMAGE
					 row[SEXTRACTOR_COLS[1]], # YMIN_IMAGE
					 row[SEXTRACTOR_COLS[2]], # XMAX_IMAGE
					 row[SEXTRACTOR_COLS[3]]  # YMAX_IMAGE
					]))
		totalDF = totalDF.append(newRow, ignore_index=True)

	return totalDF

# get total dataframe using csv files inside parentDirWithCSVs
def getTotalDF():
	parentDir = getConfig("Training","parentDirWithCSVs")

	filesInParentDir = glob(parentDir+"*.csv")

	if filesInParentDir:

		logging.info('parentDirWithCSVs has at least one csv file. Return dataframe with csv(s) in this level.')

		return pd.concat([pd.read_csv(dfFile) for dfFile in filesInParentDir])

	logging.info('parentDirWithCSVs does not have csv files in this level. Return dataframe with csv in children\'s level.')

	childrenDFs = []

	for childDir in glob(parentDir+"*/"):
		childCSVs = glob(childDir+"*.csv")
		if childCSVs:
			childrenDFs.append(pd.concat([pd.read_csv(childFile) for childFile in childCSVs]))

	return pd.concat(childrenDFs, ignore_index=True)
	# return pd.append(childrenDFs, ignore_index=True)


def getTrainSize(groupedList):
	trainFrac	= getConfig("Training","trainSizeFraction")
	trainFrac	= float(trainFrac) if trainFrac else 0.9 # default = 0.9

	trainFracMsg  = "trainFrac: {}".format(trainFrac)
	logging.debug(trainFracMsg)

	return int(trainFrac * len(groupedList))

def getDfFileNames(outDataDir):
	totalFile	= getConfig("Training", "totalFile")
	totalFile	= totalFile if totalFile else "totalDF.csv"

	trainFile	= getConfig("Training", "trainFile")
	trainFile	= trainFile if trainFile else "train.csv"
	
	testFile	= getConfig("Training", "testFile")
	testFile	= testFile if testFile else "test.csv"

	totalMsg 	= "totalFile: {}".format(totalFile)
	trainMsg 	= "trainFile: {}".format(trainFile)
	testMsg  	= "testFile: {}".format(testFile)

	logging.debug(totalMsg)
	logging.debug(trainMsg)
	logging.debug(testMsg)

	files = [totalFile, trainFile, testFile]

	return [outDataDir+file for file in files]


def saveDataFrames(totalDF, train, test):
	outDataDir = createDataDir()
	createLabelMap(outDataDir)
	totalFile, trainFile, testFile = getDfFileNames(outDataDir)

	# log csv's len
	totalMsg = "len(totalDF): {}".format(len(totalDF))
	trainMsg = "len(train): {}".format(len(train))
	testMsg  = "len(test): {}".format(len(test))

	logging.debug(totalMsg)
	logging.debug(trainMsg)
	logging.debug(testMsg)

	totalDF.to_csv(totalFile, index=False) # backup csv as it is
	train.to_csv(trainFile, index=False)
	test.to_csv(testFile,   index=False)

def getSplitSets(totalDF, groupedList):
	trainSize	= getTrainSize(groupedList)
	trainIndex	= np.random.choice(len(groupedList), size=trainSize, replace=False)
	testIndex	= np.setdiff1d(list(range(len(groupedList))), trainIndex)
	train		= pd.concat([groupedList[i] for i in trainIndex])
	test		= pd.concat([groupedList[i] for i in testIndex])

	saveDataFrames(totalDF, train, test)

def createDataSets():
	totalDF = getTotalDF()

	np.random.seed(1) # specify random seed for reproducibility 
	grouped		= totalDF.groupby('filename')
	groupedList = [grouped.get_group(x) for x in grouped.groups]
	
	getSplitSets(totalDF, groupedList)
