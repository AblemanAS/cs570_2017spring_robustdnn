from alf_utils import *

def preprocessHOG(gray):
    height, width = gray.shape
    cell_size = (1, 1)  # h x w in pixels
    block_size = (4, 4)  # h x w in cells
    nbins = 9  # number of orientation bins

    # winSize is the size of the image cropped to an multiple of the cell size
    hog = cv2.HOGDescriptor(_winSize=(width // cell_size[1] * cell_size[1],
                                      height // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    n_cells = (height // cell_size[0], width // cell_size[1])
    hog_feats = hog.compute(gray) \
        .reshape(n_cells[1] - block_size[1] + 1,
                 n_cells[0] - block_size[0] + 1,
                 block_size[0], block_size[1], nbins) \
        .transpose((1, 0, 2, 3, 4))     # index blocks by rows first
    gradients = np.zeros((n_cells[0], n_cells[1], nbins))
    cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

    for off_y in range(block_size[0]):
        for off_x in range(block_size[1]):
            gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
            off_x:n_cells[1] - block_size[1] + off_x + 1] += \
                hog_feats[:, :, off_y, off_x, :]
            cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
            off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

    gradients /= cell_count
    bin = 5  # angle is 360 / nbins * direction
    return cv2.resize(gradients[:, :, bin], (height, width))

def preprocessORB(img):
    height, width, channels = img.shape
    orb = cv2.ORB_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    kp_sorted = sorted(kp, key=lambda keypo: keypo.response, reverse=True)
    total_kp = len(kp)
    kp_important = kp_sorted[:int(total_kp / 2)]
    maskit = sum(make_mask(kp_, (height, width), False) for kp_ in kp_important)
    if type(maskit) == int: return np.zeros((height, width))
    return ac_func_mtanh(maskit)

# This contains transpose (k, h, w)
def preprocess(img):
    imgTrans = img.transpose((1, 2, 0))
    gray = cv2.cvtColor(imgTrans, cv2.COLOR_BGR2GRAY)
    imgHOG = (preprocessHOG(gray) * 255).astype(np.uint8)
    imgORB = (preprocessORB(imgTrans) * 255).astype(np.uint8)
    return np.concatenate((img, imgHOG.reshape(1, *imgHOG.shape), imgORB.reshape(1, *imgORB.shape)), axis=0)
