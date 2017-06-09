
import numpy as np
import lmdb
import caffe
from preproc import preprocess

# Set to your root path of LMDB chunks (Each LMDB chunk is a DIRECTORY that contains data.mdb, lock.mdb)
LMDB_ROOT_PATH = 'C:\\Development\\Tools\\caffe\\data\\ilsvrc12\\train_5ch'

# Don't touch these unless you know what you're doing!
BUF_SIZE = 200  # image buffer size
N = 4000  # Maximum number of Images
MAP_SIZE_MULTIPLIER = 1.1  # Various value (10 is safe)
RESOLUTION = 256
map_size = int((1 * 5 * RESOLUTION * RESOLUTION) * MAP_SIZE_MULTIPLIER * N)

def preprocessLMDBDist(number):
    print('Starting with', number, 'th chunk')
    numTxt = ('0' if number < 10 else '') + str(number)
    envOri = lmdb.open(LMDB_ROOT_PATH + '\\dist' + numTxt, readonly=True)
    envPreproc = lmdb.open(LMDB_ROOT_PATH + '\\proc' + numTxt, map_size=map_size)

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
    print(number, 'th chunk done')


# Use this for preprocessing
# start = 2, end = 10 -> do preprocess from dist02 ~ dist9 => proc02 ~ proc9
start = 15
end = 20
for i in range(start, end):
    preprocessLMDBDist(i)

