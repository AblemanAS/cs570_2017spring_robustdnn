import cv2
import numpy as np
import caffe
import random
import lmdb

from preproc import preprocess

NET_3CH_DEFINITION = str('C:\\Development\\Tools\\caffe\\models\\nin_ilsvrc12\\deploy.prototxt')
NET_5CH_DEFINITION = str('C:\\Development\\Tools\\caffe\\models\\nin_ilsvrc12_prep\\deploy.prototxt')
NET_3CH_WEIGHT = str('C:\\Development\\Tools\\caffe\\models\\nin_ilsvrc12\\nin_imagenet_train_iter_20000.caffemodel')
NET_5CH_WEIGHT = str('C:\\Development\\Tools\\caffe\\models\\nin_ilsvrc12_prep\\nin_imagenet_prep_train_iter_20000.caffemodel')
SYNSET_PATH = 'C:\\Development\\Tools\\caffe\\models\\nin_ilsvrc12\\synset_words.txt'
DATA_PATH = 'C:\\Development\\Tools\\caffe\\data\\ilsvrc12\\fool'
BOTH_PATH = 'C:\\Development\\Tools\\caffe\\models\\both_list.txt'
LMDB_TEST_PATH = 'C:\\Development\\Tools\\caffe\\data\\ilsvrc12\\test_3ch'
RESOLUTION = 256
NUM_IMAGE = 1

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

# load net
net3ch = caffe.Net(NET_3CH_DEFINITION, 1, weights=NET_3CH_WEIGHT)
net5ch = caffe.Net(NET_5CH_DEFINITION, 1, weights=NET_5CH_WEIGHT)

# load FPs
fps = []
fp = open(BOTH_PATH, 'r')
bothlist = fp.readlines()
fp.close()
for b in bothlist:
    img = cv2.imread(b[:-1])
    img = cv2.resize(img, (224, 224))
    img = img * 0.2
    fps.append(img)

datum = caffe.proto.caffe_pb2.Datum()
datum.channels = 3
datum.height = RESOLUTION
datum.width = RESOLUTION

env = lmdb.open(LMDB_TEST_PATH, readonly=True)
fooledCnt3ch = 0
fooledCnt5ch = 0
count = 0

with env.begin() as txn:
    for key, value in txn.cursor():
        datum.ParseFromString(value)
        flat_x = np.fromstring(datum.data, dtype=np.uint8)
        x = flat_x.reshape(datum.channels, datum.height, datum.width)
        x = x * 0.8
        x = cv2.resize(x.transpose((1, 2, 0)), (224, 224))
        x += fps[random.randrange(len(bothlist))]
        x = x.astype(np.uint8)
        img3ch = x.transpose((2, 0, 1))
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
        if probs3ch.argmax() != datum.label: fooledCnt3ch += 1
        if probs5ch.argmax() != datum.label: fooledCnt5ch += 1

        count += 1
        if count % 100 == 0:
            print(count, 'done')

print('fooled count :', fooledCnt3ch, fooledCnt5ch)