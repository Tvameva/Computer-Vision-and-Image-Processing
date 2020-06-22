from cv2 import imread, imwrite
from math import sqrt, pi as CONSTANT_PI, e as CONSTANT_E
import numpy as np

from cv2 import imshow, waitKey, destroyAllWindows

imageLocation = './task2.jpg'
gaussianSigma_startingValue = (1/(sqrt(2)))
gaussianSigma_WithinOctaveRatio = sqrt(2) 
gaussianSigma_AcrossOctaveRatio = 2
numberOfOctaves = 4
imagesPerOctave = 5
gaussianKernelSize = 7

def readImage(imageLocation):
    img = imread(imageLocation,0)
    return img


def writeImage(img, outputFileName):
	imwrite('output/Task2_Sol1_'+outputFileName+'.jpg', img)
	return 1


def displayImage(img, imgTitle='No Title'):
    imshow(imgTitle, img)
    waitKey(0)
    destroyAllWindows()


def getGaussianKernel(kernelSize, sigma):
	if (kernelSize%2==0):
		kernelSize=kernelSize+1
	kernelSizeHalf = (int)(kernelSize/2)
	l=-1*kernelSizeHalf
	h=kernelSizeHalf
	kernel = np.zeros((kernelSize, kernelSize))
	sum=0
	for kernel_i in np.arange( l, h+1 ):
		for kernel_j in np.arange( l, h+1 ):
			kernel[kernel_i+kernelSizeHalf][kernel_j+kernelSizeHalf] = (1/( 2*(CONSTANT_PI)*(sigma**2)*( (CONSTANT_E)**( ( ((kernel_i)**2)+((kernel_j)**2) )/( 2*(sigma**2) ) ) ) ))
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


def resize(img, factor=0.5):
	resized_img = np.zeros(( (int)((img.shape[0])*factor) , (int)((img.shape[1])*factor) ))
	for resized_img_i in range(0, resized_img.shape[0]):
		for resized_img_j in range(0, resized_img.shape[1]):
			a = (int)((1/factor)*resized_img_i)
			b = (int)((1/factor)*resized_img_j)
			resized_img[resized_img_i][resized_img_j] = (int)(img[a][b])
	writeImage(resized_img,"sdafsd")
	return resized_img


def main():
	original_img = readImage(imageLocation)
	# Step 1 - Creating Scale Space images
	scaleSpace = []
	for i in range(0, numberOfOctaves):
		temp=[]
		for j in range(0, imagesPerOctave):
			temp.append(None)
		scaleSpace.append(temp)
	octaveSigma_startingValue = gaussianSigma_startingValue / gaussianSigma_AcrossOctaveRatio
	for octaveIndex in range(0, numberOfOctaves):
		octaveSigma_startingValue = octaveSigma_startingValue * gaussianSigma_AcrossOctaveRatio
		sigma = octaveSigma_startingValue / gaussianSigma_WithinOctaveRatio
		for imageIndex in range(0, imagesPerOctave):
			if (imageIndex==0):
				if (octaveIndex==0):
					scaleSpace[0][0]=original_img
				else:
					scaleSpace[octaveIndex][0] = resize(scaleSpace[octaveIndex-1][0], factor=0.5)
			else:
				sigma = sigma * gaussianSigma_WithinOctaveRatio
				gaussianKernel = getGaussianKernel(gaussianKernelSize, sigma)
				convoluted_img, convoluted_img_abs = convolve(scaleSpace[octaveIndex][imageIndex-1], gaussianKernel)
				scaleSpace[octaveIndex][imageIndex] = convoluted_img
	for octaveIndex in range(0, numberOfOctaves):
		for imageIndex in range(0, imagesPerOctave):
			writeImage(scaleSpace[octaveIndex][imageIndex], "ScaleSpace_"+str(octaveIndex)+"_"+str(imageIndex))
	print("Scale Space Created")
	# Step 2 - Creating DOG Space images
	dogSpace = []
	for i in range(0, numberOfOctaves):
		temp=[]
		for j in range(0, imagesPerOctave-1):
			temp.append(None)
		dogSpace.append(temp)
	for octaveIndex in range(0, len(dogSpace)):
		for imageIndex in range(0, len(dogSpace[0])):
			currentDogSpace_l = (scaleSpace[octaveIndex][imageIndex]).shape[0]
			currentDogSpace_b = (scaleSpace[octaveIndex][imageIndex]).shape[1]
			dogSpace[octaveIndex][imageIndex] = np.zeros(( currentDogSpace_l , currentDogSpace_b ))
			for i in range(0, currentDogSpace_l):
				for j in range(0, currentDogSpace_b ):
					dogSpace[octaveIndex][imageIndex][i][j] = scaleSpace[octaveIndex][imageIndex+1][i][j] - scaleSpace[octaveIndex][imageIndex][i][j]
	for octaveIndex in range(0, len(dogSpace)):
		for imageIndex in range(0, len(dogSpace[0])):
			writeImage(dogSpace[octaveIndex][imageIndex], "DOGSpace_"+str(octaveIndex)+"_"+str(imageIndex))
	print("DOG Space Created")
	# Step 3 - Identifying maxima/minima in DOG images
	maxMinDogSpace = []
	for i in range(0, numberOfOctaves):
		temp=[]
		for j in range(0, imagesPerOctave-3):
			temp.append(None)
		maxMinDogSpace.append(temp)
	for octaveIndex in range(0, len(maxMinDogSpace)):
		for imageIndex in range(0, len(maxMinDogSpace[0])):
			maxMinDogSpace[octaveIndex][imageIndex] = np.zeros(( (dogSpace[octaveIndex][0]).shape[0] , (dogSpace[octaveIndex][0]).shape[1] ))
			prevImage = dogSpace[octaveIndex][imageIndex]
			currentImage = dogSpace[octaveIndex][imageIndex+1]
			nextImage = dogSpace[octaveIndex][imageIndex+2]
			for i in range(1, (currentImage.shape[0])-1):
				for j in range(1, (currentImage.shape[1])-1):
					currentPixelVal = currentImage[i][j]
					compArray = []
					k = -1
					l = -1
					while(k<2):
						while(l<2):
							compArray.append(prevImage[i+k][j+l])
							compArray.append(currentImage[i+k][j+l])
							compArray.append(nextImage[i+k][j+l])
							l=l+1
						k=k+1
					minComp = min(compArray)
					maxComp = max(compArray)
					if (minComp!=0 and maxComp!=0 and (minComp==currentPixelVal or maxComp==currentPixelVal)):
						maxMinDogSpace[octaveIndex][imageIndex][i][j] = 255
	for octaveIndex in range(0, len(maxMinDogSpace)):
		for imageIndex in range(0, len(maxMinDogSpace[0])):
			writeImage(maxMinDogSpace[octaveIndex][imageIndex], "maxMinDogSpace_"+str(octaveIndex)+"_"+str(imageIndex))
	print("Max and Min values in DOG Space Found")


main()