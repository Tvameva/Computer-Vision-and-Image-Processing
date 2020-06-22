from cv2 import imread, imwrite
import numpy as np
import math as math

imageLocation = './task1.png'


def readImage(imageLocation):
    img = imread(imageLocation,0)
    return img


def writeImage(img, outputFileName):
	imwrite('output/Task1_Sol3_'+outputFileName+'.jpg', img)
	return 1


def getBoxIntensityDoublerKernel():
	boxIntensityDoublerKernel = np.array( 
					 					  [ [0,0,0], 
		               						[0,2,0], 
		               						[0,0,0] ] 
		           						)
	return boxIntensityDoublerKernel


def getSobelX():
	sobelX = np.array( 
					 [ [1,0,-1], 
		               [2,0,-2], 
		               [1,0,-1] ] 
		           )
	normalizedSobelX = sobelX / 8
	return normalizedSobelX


def getSobelY():
	sobelY = np.array( 
					 [ [-1,-2,-1], 
		               [0,0,0], 
		               [1,2,1] ] 
		           )
	normalizedSobelY = sobelY / 8
	return normalizedSobelY


def getGaussianKernel(kernelSize, sigma=0.1):
	if (kernelSize%2==0):
		kernelSize=kernelSize+1
	kernelSizeHalf = (int)(kernelSize/2)
	l=-1*kernelSizeHalf
	h=kernelSizeHalf
	kernel = np.zeros((kernelSize, kernelSize))
	sum=0
	for kernel_i in np.arange( l, h+1 ):
		for kernel_j in np.arange( l, h+1 ):
			kernel[kernel_i+kernelSizeHalf][kernel_j+kernelSizeHalf] = (1/( 2*(math.pi)*(sigma**2)*( (math.e)**( ( ((kernel_i/(2*kernelSize))**2)+((kernel_j/(2*kernelSize))**2) )/( 2*(sigma**2) ) ) ) ))
			sum = sum + kernel[kernel_i+kernelSizeHalf][kernel_j+kernelSizeHalf]
	for kernel_i in np.arange( l, h+1 ):
		for kernel_j in np.arange( l, h+1 ):
			kernel[kernel_i+kernelSizeHalf][kernel_j+kernelSizeHalf] = (kernel[kernel_i+kernelSizeHalf][kernel_j+kernelSizeHalf]) / sum
	return kernel


def convolve(img, kernel):
	kernelSize = (int)((kernel.shape[0])/2)
	convoluted_img_abs = img.copy()
	convoluted_img = np.zeros((convoluted_img_abs.shape[0], convoluted_img_abs.shape[1]))
	for img_i in np.arange( kernelSize, ((img.shape[0])-kernelSize) ):
		for img_j in np.arange( kernelSize, ((img.shape[1])-kernelSize) ):
			temp = 0
			for kernel_i in np.arange(-1, 2):
				for kernel_j in np.arange(-1, 2):
					kernel_coordinate = kernel[kernel_i+kernelSize, kernel_j+kernelSize]
					img_coordinate = img[img_i-kernel_i, img_j-kernel_j]
					temp = temp + (img_coordinate*kernel_coordinate)
			convoluted_img_abs[img_i][img_j] = abs(temp)
			convoluted_img[img_i][img_j] = temp
	return convoluted_img, convoluted_img_abs


def eliminateZeros_method1(convoluted_img):
	outImg = np.zeros((convoluted_img.shape[0], convoluted_img.shape[1]))
	minVal = (convoluted_img[0,0])
	maxVal = (convoluted_img[0,0])
	for convoluted_img_i in np.arange( 0, convoluted_img.shape[0] ):
		for convoluted_img_j in np.arange( 0, convoluted_img.shape[1] ):
			if (minVal>(convoluted_img[convoluted_img_i, convoluted_img_j])):
				minVal=(convoluted_img[convoluted_img_i, convoluted_img_j])
			if (maxVal<(convoluted_img[convoluted_img_i, convoluted_img_j])):
				maxVal=(convoluted_img[convoluted_img_i, convoluted_img_j])
	for convoluted_img_i in np.arange( 0, convoluted_img.shape[0] ):
		for convoluted_img_j in np.arange( 0, convoluted_img.shape[1] ):
			outImg[convoluted_img_i, convoluted_img_j] = 255.0 * ( (convoluted_img[convoluted_img_i, convoluted_img_j] - minVal) / (maxVal - minVal) )
	return outImg


def eliminateZeros_method2(convoluted_img):
	outImg = np.zeros((convoluted_img.shape[0], convoluted_img.shape[1]))
	maxVal = (convoluted_img[0,0])
	for convoluted_img_i in np.arange( 0, convoluted_img.shape[0] ):
		for convoluted_img_j in np.arange( 0, convoluted_img.shape[1] ):
			if (maxVal<convoluted_img[convoluted_img_i, convoluted_img_j]):
				maxVal=convoluted_img[convoluted_img_i, convoluted_img_j]
	for convoluted_img_i in np.arange( 0, convoluted_img.shape[0] ): 
		for convoluted_img_j in np.arange( 0, convoluted_img.shape[1] ):
			outImg[convoluted_img_i, convoluted_img_j] = 255.0 * ( convoluted_img[convoluted_img_i, convoluted_img_j] / maxVal )
	return outImg


def mergeSobelXYValues(convoluted_img_sobelX, convoluted_img_sobelY):
	convoluted_img = np.zeros((convoluted_img_sobelX.shape[0], convoluted_img_sobelX.shape[1]))
	for convoluted_img_i in np.arange( convoluted_img.shape[0] ):
		for convoluted_img_j in np.arange( convoluted_img.shape[1] ):
			convoluted_img[convoluted_img_i, convoluted_img_j] = math.sqrt( ((convoluted_img_sobelX[convoluted_img_i, convoluted_img_j])**2) + ((convoluted_img_sobelY[convoluted_img_i, convoluted_img_j])**2) )
	return convoluted_img


def main():
	# Reading and displaying the Input Image
	img = readImage(imageLocation)
	# Box Intensity Double Kernel of Image
	boxIntensityDoublerKernel = getBoxIntensityDoublerKernel()
	convoluted_img_boxIntensityDoublerKernel, convoluted_img_boxIntensityDoublerKernel_abs = convolve(img, boxIntensityDoublerKernel)
	# Blur the image first
	gaussianKernel = getGaussianKernel(5, 0.13301)
	convoluted_img_gaussianKernel, convoluted_img_gaussianKernel_abs = convolve(img, gaussianKernel)
	# Find the difference between Box Intensity Double Kernel of Image and Blurring the image first
	img = np.zeros((convoluted_img_boxIntensityDoublerKernel.shape[0], convoluted_img_boxIntensityDoublerKernel.shape[1]))
	for i in range(0, convoluted_img_boxIntensityDoublerKernel.shape[0]):
		for j in range(0, convoluted_img_boxIntensityDoublerKernel.shape[1]):
			img[i][j]=convoluted_img_boxIntensityDoublerKernel[i][j]-convoluted_img_gaussianKernel[i][j]
	writeImage(img, 'img')
	# Box Intensity Double Kernel of Image
	boxIntensityDoublerKernel = getBoxIntensityDoublerKernel()
	convoluted_img_boxIntensityDoublerKernel, convoluted_img_boxIntensityDoublerKernel_abs = convolve(img, boxIntensityDoublerKernel)
	# Blur the image first
	gaussianKernel = getGaussianKernel(5, 0.13301)
	convoluted_img_gaussianKernel, convoluted_img_gaussianKernel_abs = convolve(img, gaussianKernel)
	# Find the difference between Box Intensity Double Kernel of Image and Blurring the image first
	img = np.zeros((convoluted_img_boxIntensityDoublerKernel.shape[0], convoluted_img_boxIntensityDoublerKernel.shape[1]))
	for i in range(0, convoluted_img_boxIntensityDoublerKernel.shape[0]):
		for j in range(0, convoluted_img_boxIntensityDoublerKernel.shape[1]):
			img[i][j]=convoluted_img_boxIntensityDoublerKernel[i][j]-convoluted_img_gaussianKernel[i][j]
	writeImage(img, 'img')
	# Starting Edge Detection using Sobel X Operator
	sobelX = getSobelX()
	convoluted_img_sobelX, convoluted_img_sobelX_abs = convolve(img, sobelX)
	#No point in writing convoluted_img_sobelX, as it has negative values, which cant be written by imwrite. It just messes everything up.
	#So i have taken a snapshot of this as it runs
	writeImage(convoluted_img_sobelX_abs, 'convoluted_img_sobelX_abs')
	convoluted_img_sobelX_elZerosM1 = eliminateZeros_method1(convoluted_img_sobelX_abs)
	writeImage(convoluted_img_sobelX_elZerosM1, 'convoluted_img_sobelX_elZerosM1')
	convoluted_img_sobelX_elZerosM2 = eliminateZeros_method2(convoluted_img_sobelX_abs)
	writeImage(convoluted_img_sobelX_elZerosM2, 'convoluted_img_sobelX_elZerosM2')
	# Starting Edge Detection using Sobel Y Operator
	sobelY = getSobelY()
	convoluted_img_sobelY, convoluted_img_sobelY_abs = convolve(img, sobelY)
	#No point in writing convoluted_img_sobelY, as it has negative values, which cant be written by imwrite. It just messes everything up.
	#So i have taken a snapshot of this as it runs
	writeImage(convoluted_img_sobelY_abs, 'convoluted_img_sobelY_abs')
	convoluted_img_sobelY_elZerosM1 = eliminateZeros_method1(convoluted_img_sobelY_abs)
	writeImage(convoluted_img_sobelY_elZerosM1, 'convoluted_img_sobelY_elZerosM1')
	convoluted_img_sobelY_elZerosM2 = eliminateZeros_method2(convoluted_img_sobelY_abs)
	writeImage(convoluted_img_sobelY_elZerosM2, 'convoluted_img_sobelY_elZerosM2')
	# Constructing final image by merging Sobel X and Sobel Y's output images
	final_convoluted_img = mergeSobelXYValues(convoluted_img_sobelX_abs, convoluted_img_sobelY_abs)
	writeImage(final_convoluted_img, 'final_convoluted_img')


main()