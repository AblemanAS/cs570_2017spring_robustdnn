import lmdb
import numpy as np
import cv2
import caffe

LMDB_PATH = 'C:\\Development\\Tools\\caffe\\data\\ilsvrc12\\train_5ch'
RESOLUTION = 256

datum = caffe.proto.caffe_pb2.Datum()
datum.channels = 5
datum.height = RESOLUTION
datum.width = RESOLUTION

env = lmdb.open(LMDB_PATH, readonly=True)
count = 0
with env.begin() as txn:
    for key, value in txn.cursor():
        datum.ParseFromString(value)
        flat_x = np.fromstring(datum.data, dtype=np.uint8)
        x = flat_x.reshape(datum.channels, datum.height, datum.width)
        #cv2.imshow(' ', x[:3][:][:].transpose((1, 2, 0))) # 3ch
        #cv2.imshow(' ', x[3][:][:]) # HOG
        #cv2.imshow(' ', x[4][:][:]) # ORB
        #cv2.waitKey(10)
        count += 1
        if count % 100 == 0:
            print(count, 'looked')
