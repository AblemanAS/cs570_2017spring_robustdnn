import cv2
import numpy as np
import caffe
import os
from preproc import preprocess

NET_3CH_DEFINITION = str('C:\\Development\\Tools\\caffe\\models\\nin_ilsvrc12\\deploy.prototxt')
NET_5CH_DEFINITION = str('C:\\Development\\Tools\\caffe\\models\\nin_ilsvrc12_prep\\deploy.prototxt')
NET_3CH_WEIGHT = str('C:\\Development\\Tools\\caffe\\models\\nin_ilsvrc12\\nin_imagenet_train_iter_20000.caffemodel')
NET_5CH_WEIGHT = str('C:\\Development\\Tools\\caffe\\models\\nin_ilsvrc12_prep\\nin_imagenet_prep_train_iter_20000.caffemodel')
SYNSET_PATH = 'C:\\Development\\Tools\\caffe\\models\\nin_ilsvrc12\\synset_words.txt'
DATA_PATH = 'C:\\Development\\Tools\\caffe\\data\\ilsvrc12\\fool'
BOTH_PATH = 'C:\\Development\\Tools\\caffe\\models\\both_list.txt'
NUM_IMAGE = 1

def allFiles(path):
    res = []
    for root, dirs, files in os.walk(path):
        rootpath = os.path.join(os.path.abspath(path), root)
        for file in files:
            filepath = os.path.join(rootpath, file)
            res.append((filepath, file))
    return res

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

# load net
net3ch = caffe.Net(NET_3CH_DEFINITION, 1, weights=NET_3CH_WEIGHT)
net5ch = caffe.Net(NET_5CH_DEFINITION, 1, weights=NET_5CH_WEIGHT)
fp = open(SYNSET_PATH, 'r')
synset = fp.readlines()
fp.close()

fooledCnt3ch = 0
fooledCnt5ch = 0
count = 0
bothlist = []

for imagePath, fname in allFiles(DATA_PATH):
    label = int(fname[9:].split('_')[0])
    if label > 200: continue

    # load image
    img = cv2.imread(imagePath)
    img = cv2.resize(img, (224, 224))
    img3ch = img.transpose((2, 0, 1))
    img5ch = preprocess(img3ch)

    # make batch
    batch3ch = np.zeros((NUM_IMAGE, *img3ch.shape)).astype(np.float32)
    batch5ch = np.zeros((NUM_IMAGE, *img5ch.shape)).astype(np.float32)
    batch3ch[0] = img3ch
    batch5ch[0] = img5ch

    # input, forward and get output
    net3ch.blobs['data'].data[...] = batch3ch
    net5ch.blobs['data'].data[...] = batch5ch
    out3ch = net3ch.forward()
    out5ch = net5ch.forward()
    probs3ch = out3ch['pool4'].reshape(1000)
    probs5ch = out5ch['pool4'].reshape(1000)
    answer3ch = probs3ch.argmax()
    answer5ch = probs5ch.argmax()
    confidence3ch = probs3ch[answer3ch]
    confidence5ch = probs5ch[answer5ch]

    if answer3ch == label:
        fooledCnt3ch += 1
        print('3ch :', answer3ch, ':', synset[answer3ch][:-1], '(', confidence3ch, ')')

    if answer5ch == label:
        fooledCnt5ch += 1
        print('5ch :', answer5ch, ':', synset[answer5ch][:-1], '(', confidence5ch, ')')

    if confidence3ch > 30 and confidence5ch > 30:
        bothlist.append(imagePath)
        #print(confidence3ch, confidence5ch, imagePath)

    count += 1
    if count % 100 == 0:
        print(count, 'done')

print('fooled count :', fooledCnt3ch, fooledCnt5ch, 'in', count)
for i in range(len(bothlist)):
    bothlist[i] += '\n'
fp = open(BOTH_PATH, 'w')
fp.writelines(bothlist)
fp.close()
