import shutil
import cv2
import os

MAP_SIZE_MULTIPLIER = 2 # Try various value (10 is safe)
ROOT_PATH = 'C:\\Development\\Tools\\caffe\\data\\ilsvrc12'
IMAGE_PATH = 'C:\\Development\\Tools\\caffe\\data\\ilsvrc12\\raw'
TRAIN_PATH = 'C:\\Development\\Tools\\caffe\\data\\ilsvrc12\\train'
TEST_PATH = 'C:\\Development\\Tools\\caffe\\data\\ilsvrc12\\test'
SYNSET_PATH = ROOT_PATH + '\\synset_words.txt'
TEST_LABEL_PATH = ROOT_PATH + '\\test_label.txt'
TRAIN_LABEL_PATH = ROOT_PATH + '\\train_label.txt'

def allFiles(path):
    res = []
    for root, dirs, files in os.walk(path):
        rootpath = os.path.join(os.path.abspath(path), root)
        for file in files:
            filepath = os.path.join(rootpath, file)
            res.append((filepath, file))
    return res

def resizeImage(img):
    vertShort = True if img.shape[0] < img.shape[1] else False
    if vertShort:
        gap = (img.shape[1] - img.shape[0]) // 2
        return cv2.resize(img[:, gap:gap + img.shape[0]], (RESOLUTION, RESOLUTION))
    else:
        gap = (img.shape[0] - img.shape[1]) // 2
        return cv2.resize(img[gap:gap + img.shape[1], :], (RESOLUTION, RESOLUTION))

# Operation Start
synsetText = ''
trainLabelText = ''
testLabelText = ''
count = 0
labels = os.listdir(IMAGE_PATH)
for i in range(len(labels)):
    synsetText += labels[i] + '\n'
    countPerLabel = 0
    for imagePath, fname in allFiles(IMAGE_PATH + '\\' + labels[i]):
        if countPerLabel % 100 == 0:
            testLabelText += '\\' + fname + ' ' + str(i) + '\n'
            shutil.copyfile(imagePath, TEST_PATH + '\\' + fname)
        else:
            trainLabelText += '\\' + fname + ' ' + str(i) + '\n'
            shutil.copyfile(imagePath, TRAIN_PATH + '\\' + fname)
        countPerLabel += 1
        count += 1
        if count % 100 == 0:
            print(count, i, 'done')

fp = open(SYNSET_PATH, 'w')
fp.write(synsetText[:-1])
fp.close()

fp = open(TEST_LABEL_PATH, 'w')
fp.write(testLabelText[:-1])
fp.close()

fp = open(TRAIN_LABEL_PATH, 'w')
fp.write(trainLabelText[:-1])
fp.close()
