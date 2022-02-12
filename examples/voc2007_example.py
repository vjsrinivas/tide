import os
import sys
import tidecv
from tidecv.drivers.fileBB import TIDE_FromListFolder
from tidecv.quantify import TIDE

groundTruthFolder = './voc2007/gt'
detectionFolder = './voc2007/det'
testFile = './voc2007/2007_test.txt'
gtData, detData = TIDE_FromListFolder(testFile, groundTruthFolder, detectionFolder)
tide = TIDE()
tide.evaluate_range(gtData, detData)
tide.summarize()
tide.plot()