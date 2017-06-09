import cv2
import numpy as np

def make_mask(keypoint, img_size, normalize_center=False):
	x, y = img_size
	px, py = keypoint.pt
	size = keypoint.size
	sigma = 0.3*((size-1)*0.5 - 1) + 0.8
	size = int(round(size))
	if not size % 2:
		size = size + 1 
	kernelx = cv2.getGaussianKernel(size, sigma)
	kernely = cv2.getGaussianKernel(size, sigma)
	kernel = kernelx * kernely.T
	_px = int(round(px))
	_py = int(round(py))
	center = (_py, _px)
	#print('center {}'.format(center))
	mat = np.zeros(img_size)
	x_inf = int(center[0] - (size - 1)/2)
	x_up = int(center[0] + (size - 1)/2 + 1)
	y_inf = int(center[1] - (size - 1)/2)
	y_up = int(center[1] + (size - 1)/2 + 1)
	#print (x_inf, x_up, y_inf, y_up)
	mat[x_inf : x_up, y_inf : y_up] = kernel

	if normalize_center:
		value = mat[center]
		value = 1/value
		mat = value*mat

	return mat

def ac_func_abs(mat):
	mat_ = mat.copy()
	maxim = mat_.max()
	maxim = 1./maxim
	mat_ = mat_*maxim	# linealy scaled for 0 - 1
	mat_1 = np.ones(mat_.shape) * ( mat_ > 1e-7)
	mat_2 = 0.2*np.ones(mat_.shape) * ( mat_ == 0.0 )
	mat_ = 1*mat_1 + 0*mat_2
	return mat_

def ac_func_circular(mat):
	def f(x):
		return np.sqrt(1 - (x-1)**2)
	mat_ = mat.copy()
	maxim = mat_.max()
	maxim = 1./maxim
	mat_ = mat_*maxim	# linealy scaled for 0 - 1
	mat_1 = f(mat_)
	#mat_2 = 0.1*np.ones(mat_.shape) * ( mat_1 == 0.0 )
	mat_ = mat_1 #+ mat_2 
	return mat_

# modified hiperbolic tangent
def ac_func_mtanh(mat):
	def f(x):
		a, b, c, d = 1, 82, 1, 82
		return((np.exp(a*x)-np.exp(-b*x))/(np.exp(c*x)+np.exp(-d*x)))
	mat_ = mat.copy()
	maxim = mat_.max()
	maxim = 1./maxim
	mat_ = mat_*maxim	# linealy scaled for 0 - 1
	mat_1 = f(mat_)
	#mat_2 = 0.1*np.ones(mat_.shape) * ( mat_ < 0.01 )
	mat_ = 1*mat_1 #+ mat_2
	return mat_

def apply_mask(img, mask):
	img_ = img.copy()
	# no chanles
	if len(img.shape) == 2:
		X, Y = img.shape
		for x in range(X):
			for y in range(Y):
				value_img = img.item(x,y)
				value_mask = mask[x,y]
				new_val = value_mask*value_img
				img_.itemset((x,y), new_val)
	# RGB chanels
	else:
		X, Y, C = img.shape
		for x in range(X):
			for y in range(Y):
				for c in range(C):
					value_img = img.item(x,y,c)
					value_mask = mask[x,y]
					new_val = value_mask*value_img
					img_.itemset((x,y,c), new_val)
	return img_
