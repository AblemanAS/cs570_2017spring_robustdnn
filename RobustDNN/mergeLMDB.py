import lmdb

BUF_SIZE = 4000 # image buffer size
N = 256000 # number of images
MAP_SIZE_MULTIPLIER = 1.1 # Try various value (10 is safe)
LMDB_DIST_PATH = 'C:\\Development\\Tools\\caffe\\data\\ilsvrc12\\train_5ch'
RESOLUTION = 256
DIST_CNT = 64

map_size = int((1 * 5 * RESOLUTION * RESOLUTION) * MAP_SIZE_MULTIPLIER * N)
envMerged = lmdb.open(LMDB_DIST_PATH, map_size=map_size)

count = 0
for i in range(DIST_CNT):
    buffer = []
    envDist = lmdb.open(LMDB_DIST_PATH + ('\\proc0' if i < 10 else '\\proc') + str(i), readonly=True)
    with envDist.begin() as txnDist:
        for record in txnDist.cursor():
            buffer.append(record)
            count += 1
            if count % 100 == 0:
                print(count, 'at', i, 'th chunk', 'read')

    with envMerged.begin(write=True) as txnMerged:
        txnMerged.cursor().putmulti(buffer)
