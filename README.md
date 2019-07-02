# Preprocessing DES raw data 
## Preprocessing DES raw data for object detecting (deep learning -- tensorflow) 

This preprocessing system extracts fits files from DES raw data with funpack, process CCDs data in parallel, divide each CCD into subimages, save fits and jpegs of them, run SExtractor upon them, and transform the data into tensorflow records for object detection.

Automatic, parallelized, with configurable options, and logging feature through all stages.

## Setting up (Ubuntu)

### funpack
```bash
sudo apt install libcfitsio-bin
```

### SExtractor

```bash 
sudo apt-get update -y
sudo apt-get install -y sextractor
```

More info about SExtractor:
https://sextractor.readthedocs.io/en/latest/Installing.html
https://www.astromatic.net/pubsvn/software/sextractor/trunk/doc/sextractor.pdf

### python requirements

```bash
pip install -r requirements.txt
```

## File structure

Cwd should have these files:

	.   
	├── configHelper.py
	├── dataFrame.py
	├── default.conv
	├── default.param
	├── default.sex
	├── fits_io.py
	├── image.py
	├── preprocess_config.ini
	├── preprocess.py
	├── README.md
	├── requirements.txt
	└── tensorflow_records.py

## Configuration file

A configuration file is required to run: 

	preprocess_config.ini

This file has the following sections: **[Paths]**, **[InputFiles]**, **[Log]**, **[Image]**, **[Operations]**, and **[Training]**.

It is recommended to save a backup version before changing this config file:
```bash
cp  preprocess_config.ini preprocess_config_BCKP.ini
```

Configure each item from each section. Commentaries about the configuration are provided below.

### [Paths]

All paths should end with "/".

 * ***inputPath***: path to one or more fits.fz (compacted) files.
 * ***outputPath***: path where all outputs (images and catalogs) will be saved.
 * ***maskPath***: if you want to mask the raw exposures, provide the mask path here. 

### [InputFiles]

 * ***singleFZFile***: if you have more than one file in ***inputPath*** and want to process just one of them, provide the file name here.
 
### [Log]

The system logs everything that happens in the following format:

	<datetime> [<LEVEL>]	<file> - <msg>

Example of a line of the log file (first line):

	[2019-06-26 10:48:02,652] [INFO]	preprocess - Starting Pre-Processing for Object Detection.

* ***logFile***: log file name.

* ***level***: you can specify one of two log levels: ***INFO*** or ***DEBUG***. 

  * ***INFO*** is the default value for log level, with which you will get *INFO*, *WARNING* and *CRITICAL* messages.

  * ***DEBUG*** outputs all messages from the system, including values of the variables in each calculation step -- this should be used only on debug runs -- if something goes wrong or if you want to trace every detail. 

### [Image]

Ih the ***Image*** subsection, ***dpiw*** and ***dpih*** are not in use in this current version.

* ***n_subImages***: in how many subimages should the original exposure be split (default empyrically set to 6 -- divide horizontally in two, then split each half in three).
* ***percentile***: ***n***th percentile of the matrix data to be used as max value in normalization (default: 100).
* ***ccdByNumOrName***: we can use CCDs names or numbers as suffix to output files (default: name).
* ***maskBorder***: if not applying mask, how much (in pixels) should be cut from the image border (default: 180).
* ***maskExtension***: if using mask files, what is the extension.

### [Operations]
Define whether to run or not each operation in boolean values (***True*** or ***False***).

* ***subImages***: the whole processing of fits(.fz) files -- decompress with funpack, go thorough all exposures in each compressed file, process all CCDs in parallel saving jpeg, fits, and obtain object catalogs (cat and csv) by running sextractor as well.
* ***createDataSets***: gather all csv files into one total csv file, and split it into training and test datasets.

### [Training]
* ***parentDirWithCSVs***: directory with csv files or parent directory for which children directories have csv files.
* ***totalFile***: csv file name with all information generated in one run (default: totalDF.csv). 
* ***trainSizeFraction***: Fraction of total data to be used as training set (default: 0.9).
* ***trainFile***: csv file name with training set (default: train.csv).
* ***testFile***: csv file name with test set (default: test.csv).
* ***labelMapFile***: file with label map information for tensorflow. Current file in use:

```
item {
  id: 1
  name: 'obj'
}
```
which means the object detector just has to identify one class: 'obj'.

## Running preprocessing
Recommended: run ***subImages*** opeartion to all .fits.fz first; then run ***createDataSets***.

```bash
python preprocess.py <configFile>
```
