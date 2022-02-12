import sys
import os
sys.path.append('../')
from tidecv.data import Data 
from tidecv.quantify import TIDE
from tidecv.drivers.classLabels import *

def loadGroundTruth(gtFile:str, gtFolder:str, data, classes):
    with open(gtFile, 'r') as f:
        gtIds = list(map(str.strip, f.readlines()))
        
    bb = []
    for ids in gtIds:
        _file = os.path.join(gtFolder, "%s.txt"%(ids))
        with open(_file, 'r') as f:
            bboxes = list(map(str.strip, f.readlines()))
            for i in range(len(bboxes)):
                bboxes[i] = bboxes[i].split(' ')
                bboxes[i][1] = int(bboxes[i][1])
                bboxes[i][2] = int(bboxes[i][2])
                bboxes[i][3] = int(bboxes[i][3])
                bboxes[i][4] = int(bboxes[i][4])
            bb.append(bboxes)
    
    for img_id, img in enumerate(bb):
        for _bb in img:
            _cls_id = classes.index(_bb[0])
            _bbox = _bb[1:]
            #print(_cls_id, _bbox)
            data.add_ground_truth(img_id, _cls_id, _bbox)
    #return bb

def loadDetections(detFile:str, detFolder:str, data, classes):
    with open(detFile, 'r') as f:
        gtIds = list(map(str.strip, f.readlines()))
        
    bb = []
    for ids in gtIds:
        _file = os.path.join(detFolder, "%s.txt"%(ids))
        with open(_file, 'r') as f:
            bboxes = list(map(str.strip, f.readlines()))
            for i in range(len(bboxes)):
                bboxes[i] = bboxes[i].split(' ')
                bboxes[i][1] = float(bboxes[i][1])
                bboxes[i][2] = int(bboxes[i][2])
                bboxes[i][3] = int(bboxes[i][3])
                bboxes[i][4] = int(bboxes[i][4])
                bboxes[i][5] = int(bboxes[i][5])
            bb.append(bboxes)

    for img_id, img in enumerate(bb):
        for _bb in img:
            _cls_id = classes.index(_bb[0])
            _score = _bb[1]
            _bbox = _bb[2:]
            #print(_cls_id, _bbox)
            data.add_detection(img_id, _cls_id, _score, _bbox)

def initializeData(data, imageIds, classes):    
    for i,c in enumerate(classes):
        data.add_class(i,c)

    with open(imageIds, 'r') as f:
        ids = list(map(str.strip, f.readlines()))
        for _ids in ids:
            data.add_image(i,"%s.jpg"%(_ids))

def TIDE_FromListFolder(fileList, gtFolder, predFolder, gt_name='gt', det_name='det', classes=VOC_CLASSES):
    """Function assembles TIDE Data object based off a "file list" and "detection & ground truth"
    folder structure. This function assumes the detection & ground truth files are plaintext files
    represent one image and contain every detection in [class(str), conf(float), x1, y1, x2, y2]

    Args:
        fileList (str): The path to a file that contains all the image IDs of the dataset.
        The image IDs should be seperated by a newline and contain no file extensions.
        gtFolder (str): The folder path to where all the ground truth files are stored. They should
        be the same length as the number of entries in the fileList.
        predFolder (str): The folder path to where all the detection files are stored.
        They should be the same length as the number of entries in the fileList.

    Returns:
        tuple: Two TIDE Data objects containing the image ids and annotations (bounding boxes)
        for both ground truth and detection
    """

    # create data object:
    gtData = Data(gt_name, max_dets=10000)
    detData = Data(det_name, max_dets=10000)
    # init data:
    initializeData(gtData, fileList, classes)
    initializeData(detData, fileList, classes)
    
    # load in gt and dets:
    loadGroundTruth(fileList, gtFolder, gtData, classes)
    loadDetections(fileList, predFolder, detData, classes)

    return gtData, detData