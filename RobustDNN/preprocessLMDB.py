
import numpy as np
import lmdb
import caffe
from preproc import preprocess

BUF_SIZE = 256 # image buffer size
N = 2561 # Maximum number of Images
MAP_SIZE_MULTIPLIER = 1.1 # Try various value (10 is safe)
LMDB_ORI_PATH = 'C:\\Development\\Tools\\caffe\\data\\ilsvrc12\\test_3ch'
LMDB_PREPROC_PATH = 'C:\\Development\\Tools\\caffe\\data\\ilsvrc12\\test_5ch'
RESOLUTION = 256

map_size = int((1 * 5 * RESOLUTION * RESOLUTION) * MAP_SIZE_MULTIPLIER * N)
envOri = lmdb.open(LMDB_ORI_PATH, readonly=True)
envPreproc = lmdb.open(LMDB_PREPROC_PATH, map_size=map_size)

datumOri = caffe.proto.caffe_pb2.Datum()
datumPreproc = caffe.proto.caffe_pb2.Datum()
datumPreproc.channels = 5
datumPreproc.height = RESOLUTION
datumPreproc.width = RESOLUTION

curKey = None
count = 0
while True:
    with envOri.begin() as txnOri:
        cursor = txnOri.cursor()
        if curKey: cursor.set_key(curKey)
        # read Ori datum
        buffer = []
        for i in range(BUF_SIZE):
            if not cursor.next(): break
            curKey = cursor.key()
            datumOri.ParseFromString(cursor.value())
            flat_x = np.fromstring(datumOri.data, dtype=np.uint8)
            x = flat_x.reshape(datumOri.channels, datumOri.height, datumOri.width)
            datumPreproc.data = preprocess(x).tobytes()  # or .tostring() if numpy < 1.9
            datumPreproc.label = datumOri.label
            buffer.append((curKey, datumPreproc.SerializeToString()))
            count += 1
            if count % 100 == 0:
                print(count, datumOri.label, 'done')
    if len(buffer) == 0: break

    with envPreproc.begin(write=True) as txnPreproc:
        txnPreproc.cursor().putmulti(buffer)