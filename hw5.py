import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

#Q2: make plot 
def plotData(filepath):
	X = []
	Y = []
	with open(filepath, mode = 'r') as file:
		heading = next(file) 
		scan = csv.reader(file)
		for line in scan:
			X.append(line[0])
			Y.append(line[1])
	plt.plot(X, Y)
	plt.xlabel("Year")
	plt.ylabel("Number of Frozen Days")
	plt.savefig("plot.jpg")
	return X, Y

#Q3a: compute X
#Q3b: compute Y
def XY(xArr, yArr):
	year = []
	for i in range(len(xArr)):
		x = [1, xArr[i]]
		year.append(np.transpose(x))
	X = np.array(year, dtype = 'int64')
	Y = np.array(yArr, dtype = 'int64')
	return X, Y	
	
#Q3c: compute Xtranspose * X
def xTx(X):
	#X is a n x 2, Xtrans is a 2 x n so when multied Xtrans * X = 2 x 2 matrix
	xtx = np.dot(np.transpose(X), X)
	Z = np.array(xtx, dtype = 'int64')
	return Z	

#Q3d: compute (Xtranspose * X) inverse
def getInverse(xTx):
	#xTx inverse remains a 2 x 2 matrix
	I = np.linalg.inv(xTx)
	return I

#Q3e: compute inverse of (xTx) * Xtranspose
def pseudoInverse(I, Xt):
	#I is supposed to be a 2 x 2 matrix, Xtrans is a 2 x n 
	#so result is a 2 x n matrix
	pI = np.dot(I, Xt)
	return pI

#Q3f: compute betaHat
def bHat(pI, Y):
	bHat = np.dot(pI, Y)
	return bHat

#Q4: compute a prediction for year xTest = 2022 using linear regression model
def predict(bh,xTest):
	yTest = bh[0] + bh[1]*xTest
	return yTest

#Q5a: compute the sign of betaHat1
#Q5b: explain what the sign could mean
def signBeta1(b1):
	if b1 > 0:
		print("Q5a: >")
		print("Q5b: Beta 1 is greater than 0")
	if b1 < 0:
		print("Q5a: <")
		print("Q5b: Beta 1 is less than 0")
	else:
		print("Q5a: =")
		print("Q5b: Beta 1 is equal to 0")

#Q6a: solve the equation 0 = betaHat0 + betaHat1(x*) for x*
def predictZero(betaHat):
	xStar = -(betaHat[0])/(betaHat[1])
	return xStar

#Q6b: discuss whether this x* makes sense given what we see in the data trends


def main():
	filename = sys.argv[1]
	x, y = plotData(filename)
	X, Y = XY(x, y)
	print("Q3a:")
	print(X)
	print("Q3b:")
	print(Y)
	Z = xTx(X)
	print("Q3c:")
	print(Z)
	I = getInverse(Z)
	print("Q3d:")
	print(I)
	PI = pseudoInverse(I, np.transpose(X))
	print("Q3e:")
	print(PI)
	betaHat = bHat(PI, Y)
	print("Q3f:")
	print(betaHat)
	yTest = predict(betaHat, 2022)
	print("Q4: " + str(yTest))
	signBeta1(betaHat[1])
	xStar = predictZero(betaHat)
	print("Q6a: " + str(xStar))
	print("Q6b: This is an accurate prediction as it follows the trends of previous years from the graph formulated.")

if __name__ == '__main__':
	main()
