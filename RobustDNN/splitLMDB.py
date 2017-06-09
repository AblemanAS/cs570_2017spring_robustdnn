import lmdb

BUF_SIZE = 4000 # image buffer size
MAP_SIZE_MULTIPLIER = 1.1 # Try various value (10 is safe)
LMDB_ORI_PATH = 'C:\\Development\\Tools\\caffe\\data\\ilsvrc12\\train_3ch'
LMDB_DIST_PATH = 'C:\\Development\\Tools\\caffe\\data\\ilsvrc12\\train_5ch'
RESOLUTION = 256

map_size = int((1 * 3 * RESOLUTION * RESOLUTION) * MAP_SIZE_MULTIPLIER * BUF_SIZE)
envOri = lmdb.open(LMDB_ORI_PATH, readonly=True)

curKey = None
count = 0
distCnt = 0
while True:
    with envOri.begin() as txnOri:
        cursor = txnOri.cursor()
        if curKey: cursor.set_key(curKey)
        # read Ori datum
        buffer = []
        for i in range(BUF_SIZE):
            if not cursor.next(): break
            curKey = cursor.key()
            buffer.append((curKey, cursor.value()))
            count += 1
            if count % 100 == 0:
                print(count, 'done')
    if len(buffer) == 0: break

    print('writing', len(buffer), 'buffer :', str(buffer[0][0]), '~', str(buffer[-1][0]), 'to', ('dist0' if distCnt < 10 else 'dist') + str(distCnt))
    envDist = lmdb.open(LMDB_DIST_PATH + ('\\dist0' if distCnt < 10 else '\\dist') + str(distCnt), map_size=map_size)
    with envDist.begin(write=True) as txnDist:
        txnDist.cursor().putmulti(buffer)
    distCnt += 1
