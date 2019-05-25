import os
import scipy.io
import json
import re
from random import shuffle

def acquire_labeled_img():
    dirs = []
    for x in os.walk('LABELED_IMAGES'):
        dirs.append(x[0])
    dirs.pop(0)
    suffix = ['B', 'H', 'S', 'T']

    locations = ["OFFICE", "COURTYARD", "LIVINGROOM"]
    activities = ["CHESS", "JENGA", "PUZZLE", "CARDS"]

    def getFullStringPath(substr, li):
        for str in li:
            if substr in str: return str

    data = {}
    for loc in locations:
        for actvty in activities:
            suffix_num = 1
            for s in suffix:
                target_mat = actvty + '_' + loc + '_' + str(suffix_num) + '.mat'
                source_img_dir = getFullStringPath(actvty + '_' + loc + '_' + s, dirs)
                target_mat = 'OUTPUT_MASKS/' + target_mat
                for (dirpath, dirnames, filenames) in os.walk(source_img_dir):
                    sorted_files = sorted(filenames)
                    for idx, f in enumerate(sorted_files):
                        source_img = source_img_dir + '/' +  f
                        if (source_img[-4:] != '.mat'):
                            # Frame number is from 0 to 99 since it is labeled by idx
                            data[source_img] = str(idx) + '-' + target_mat
                suffix_num = suffix_num + 1

    with open('data.json', 'w') as fp:
        json.dump(data, fp)

# returns the string of the nth line (line_num) of a given file.
# line_num range --->>> (1,2,3,...)
def readNthLineOfFile(filename, line_num):
    with open(filename) as fp:
        for i, line in enumerate(fp):
            if i == line_num-1:
                return line

# Separate labeled images into their corresponding counterpart segmentation label (0,1,2)
def createSegmentationLabelDataset():
    label_0 = {}
    label_1 = {}
    label_2 = {}

    datafile = 'data.json'
    with open(datafile, 'r') as fp:
        data = json.load(fp)

    loc_action_names = set([])
    for key in data.keys():
        result = re.search('LABELED_IMAGES/(.*)/frame*', key)
        fullTextFileName = "egohands_labels/text_labels/" + result.group(1) + ".txt"
        # Line number equivalent to the frame number
        lineNum = int(data[key].split('-')[0]) # Returns idx num (0,1,...)
        lineStr = readNthLineOfFile(fullTextFileName, lineNum+1)
        lineStr = lineStr[:-1]

         # 0,1,2 segmentation
         # 0 - no occluded hands
         # 1 - occluded hands, with occlusion segmented
         # 2 - occluded hands, but no occlusion segmented
        segment_label_num = int(lineStr[-1:])
        if (segment_label_num == 0):
            label_0[key] = data[key]
        elif (segment_label_num == 1):
            label_1[key] = data[key]
        elif (segment_label_num == 2):
            label_2[key] = data[key]
        else:
            raise ValueError('Invalid label')

    with open('label_0.json', 'w') as fp:
        json.dump(label_0, fp)
    with open('label_1.json', 'w') as fp:
        json.dump(label_1, fp)
    with open('label_2.json', 'w') as fp:
        json.dump(label_2, fp)

# Create train, test, and valid json files from data.json
def createTrainValidTestSet():
    valid_split = 400 # 400 total valid images
    test_split = 800 # 800 total test images
    valid_dict = {}
    test_dict = {}

    datafile = 'data.json'
    with open(datafile, 'r') as fp:
        data = json.load(fp)
    keys = data.keys()
    shuffle(keys)

    for i in range(0, valid_split):
        valid_dict[keys[i]] = data[keys[i]]
        data.pop(keys[i], None)
    for i in range(valid_split, valid_split + test_split):
        test_dict[keys[i]] = data[keys[i]]
        data.pop(keys[i], None)
    with open('train_data.json', 'w') as fp:
        json.dump(data, fp)
    with open('valid_data.json', 'w') as fp:
        json.dump(valid_dict, fp)
    with open('test_data.json', 'w') as fp:
        json.dump(test_dict, fp)

if __name__ == "__main__":
    createSegmentationLabelDataset()
