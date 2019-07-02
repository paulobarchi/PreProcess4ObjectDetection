import ConfigParser
import logging

##### CONFIG FUNCTIONS #####
def getConfig(section, item, boolean=False, userConfigFile="preprocess_config.ini"):
	configFile = ConfigParser.ConfigParser()
	configFile.read(userConfigFile)

	if (not configFile.has_option(section, item)):
		msg = '{item} from [{setion}] NOT found in config file: {userConfigFile}!'.format(item=item, section=section, userConfigFile=userConfigFile)

		if (section != 'Log' and item != 'singleFile'):
			logging.warning(msg)
		return ""

	msg = '{item}: {value}'.format(item=item, value=configFile.get(section, item))
	if (section != 'Log'):
		logging.debug(msg)

	if boolean:
		return configFile.getboolean(section, item)

	return configFile.get(section, item)

def isOperationSet(operation, section="Operations"):
	return getConfig(section=section, item=operation, boolean=True)

def getImgConfig():
	n_subImages = getConfig("Image","n_subImages")
	percentile 	= getConfig("Image","percentile")
	n_subImages = int(n_subImages)	if n_subImages	else n_subImages
	percentile  = float(percentile)	if percentile	else percentile
	return dict(n=n_subImages, perc=percentile)
