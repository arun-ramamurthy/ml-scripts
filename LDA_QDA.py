import math
import copy
import scipy as sp
import scipy.io as spi
import numpy as np
import numpy.random as npr
from numpy.linalg import inv
from numpy.linalg import det
from numpy.linalg import eig
from numpy.linalg import slogdet
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.preprocessing import normalize

# Classes
class Dataset:
    def __init__(self, filename, parting = .2, testSet = False, normalize = False):
        self.dataset = self.parse(filename, testSet)
        if not testSet:
            npr.shuffle(self.dataset)
            self.validationSetX, self.validationSetY, self.trainingSetX, self.trainingSetY = self.partition(parting, normalize)
            self.labels = np.unique(self.trainingSetY)
            self.X_c = {label: self.trainingSet()[self.trainingSet()[:, self.F()] == label, :self.F()] for label in self.labels}
        if testSet:
            self.testSetX = self.dataset


    def parse(self, filename, testSet = False):
        rawMat = spi.loadmat(directory + filename)
        if testSet:
            return rawMat["testX"]
        else:
            return rawMat["trainX"]

    def partition(self, parting, normalize = False):
        if parting <= 1:
            C = math.floor(parting*self.N())
        else:
            C = parting
        F = self.F()
        if normalize:
            return normalise(self.dataset[:C, :F]), self.dataset[:C, F], normalise(self.dataset[C:, :F]), self.dataset[C:, F]
        else:
            return self.dataset[:C, :F], self.dataset[:C, F], self.dataset[C:, :F], self.dataset[C:, F]

    def cut(self, n):
        cutDataset = copy.copy(self)
        cutDataset.trainingSetX, cutDataset.trainingSetY = cutDataset.trainingSetX[:n], cutDataset.trainingSetY[:n]
        return cutDataset

    def trainingSet(self):
        return np.hstack([self.trainingSetX, self.trainingSetY.reshape(self.C(), 1)])

    def validationSet(self):
        return np.hstack([self.validationSetX, self.validationSetY])

    def F(self):
        return self.dataset.shape[1]-1

    def N(self):
        return self.dataset.shape[0]

    def C(self):
        return self.trainingSetY.shape[0]

    def C_prime(self):
        return self.validationSetY.shape[0]

    def countLabel(self, label):
        return np.count_nonzero(self.trainingSetY == label)

    def propLabel(self, label):
        return float(self.countLabel(label)) / self.N()

    def meanVector(self, label):
        return np.mean(self.X_c[label], axis = 0)

class SpamDataset(Dataset):
    def parse(self, filename, testSet):
        rawMat = spi.loadmat(directory + filename)
        if testSet:
            return rawMat["test_data"]
        else:
            return np.concatenate([rawMat["training_data"], np.transpose(rawMat["training_labels"])], axis = 1)

class Classifier:
    def __init__(self, trainingSet):
        pass

    def predict(self, testSet):
        return np.ones(testSet.shape[0])

    def computeAccuracy(self, validationSetX, validationSetY):
        z = list(self.predict(validationSetX))
        y = list(validationSetY)
        return sum([i == j for i,j in zip(z,y)]) / len(z)

class AnisotropicGuassian(Classifier):
    def __init__(self, trainingSet):
        self.pi = {label:trainingSet.propLabel(label) for label in trainingSet.labels}
        self.mu_hat = {label:trainingSet.meanVector(label) for label in trainingSet.labels}
        self.d = trainingSet.F()
        self.labels = trainingSet.labels

    def predict(self, testSet):
        return np.array([self.classifyPoint(x) for x in getRowVectors(testSet)])

    def classifyPoint(self, x):
        bestLabel = 0
        bestQ = -np.inf
        for c in self.labels:
            if self.Q_c(x, c) > bestQ:
                bestLabel = c
                bestQ = self.Q_c(x, c)
        return bestLabel

    def Q_c(self, x, c):
        pass

class LDA(AnisotropicGuassian):
    def __init__(self, trainingSet):
        super().__init__(trainingSet)
        self.sigma_hat = sum([np.cov(trainingSet.X_c[label], bias = True, rowvar = False) for label in self.labels]) / len(self.labels)
        self.sigma_hat = kludge(self.sigma_hat)
        self.sigma_hat_inv = inv(self.sigma_hat)
        self.mu_hat_T = {label: (self.mu_hat[label]).reshape(self.d, 1).T for label in self.labels}
        self.w = {label: self.mu_hat_T[label] @ self.sigma_hat_inv for label in self.labels}
        self.alpha = {label: (1/2.0)*self.mu_hat_T[label] @ self.sigma_hat_inv @ self.mu_hat[label] + np.log(self.pi[label]) for label in self.labels}
        print("LDA Constructed")

    def Q_c(self, x, c):
        return self.w[c] @ x - self.alpha[c]

    # def computeAccuracy(self, validationSetX, validationSetY):
    #     z = list(self.predict(validationSetX))
    #     y = list(validationSetY)
        
    #     for i in range(len(getRowVectors(validationSetX))):
    #         print("Y: ", y[i], "Z: ", z[i], [(c, self.Q_c(getRowVectors(validationSetX)[i], c)[0]) for c in self.labels])
        
    #     return sum([i == j for i,j in zip(z,y)]) / len(z)

class QDA(AnisotropicGuassian):
    def __init__(self, trainingSet):
        super().__init__(trainingSet)
        #self.sigma_hat = {label: sum([outer(x_i - self.mu_hat[label], x_i - self.mu_hat[label]) for x_i in getRowVectors(trainingSet.X_c(label))]) / trainingSet.countLabel(label) for label in trainingSet.labels}
        self.sigma_hat = {label: np.cov(trainingSet.X_c[label], bias = True, rowvar = False) for label in self.labels}
        self.sigma_hat = {label: kludge(self.sigma_hat[label]) for label in self.labels}
        self.sigma_hat_inv = {label: inv(self.sigma_hat[label]) for label in self.labels}
        self.sigma_hat_logdet = {label: slogdet(self.sigma_hat[label])[1] for label in self.labels}
        self.term = {label: (1/2)*self.sigma_hat_logdet[label] + np.log(self.pi[label]) for label in self.labels}
        print("QDA Constructed")
        
    def Q_c(self, x, c):
        mu_hat_diff = ((x - self.mu_hat[c]).reshape(self.d,1))
        mu_hat_diff_T = ((x - self.mu_hat[c]).reshape(self.d,1).T)
        return -(1/2)* mu_hat_diff_T @ self.sigma_hat_inv[c] @ mu_hat_diff - self.term[c]
    
    def __str__(self):
        return str(self.sigma_hat[1].max())

    
# Methods
def kludge(M):
    #l_diagonal = 10* smallestEval(M) * np.identity(M.shape[0])
    l_diagonal = 2**-40 * np.identity(M.shape[0])
    return M + l_diagonal

def normalise(M):
    return normalize(M, norm = 'l2')

def smallestEval(M):
    evals = list(eig(M)[0])
    smallest = evals[0]
    for l in evals:
        if np.imag(l) == 0 and l > 0 and l < smallest:
            smallest = l
    return smallest


def plotAccuracies(dataset, nVals = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000], lda = True, subplot = 111):
    print("N ACC STARTED.")
    print("FITTING")
    if lda:
        models = [LDA(dataset.cut(n)) for n in nVals]
    else:
        models = [QDA(dataset.cut(n)) for n in nVals]
    print("COMPUTING ACC")
    accuracies = [model.computeAccuracy(dataset.validationSetX, dataset.validationSetY) for model in models]
    print("N ACC DONE.")
    print(list(zip(nVals, accuracies)))
    plt.subplot(subplot)
    plt.plot(nVals, accuracies)

def getRowVectors(array):
    R, C = array.shape[0], array.shape[1]
    return [x.reshape(C,) for x in np.vsplit(array,R)]


def kFoldCrossvalidation(dataset, classifier, cVals, k = 5, n = 10):
    folds = np.vsplit(dataset.dataset[:n, :], k)
    accuracies = [np.mean([classifier.computeAccuracy(*craftFoldedDataset(folds, i), C = C) for i in range(k)]) for C in cVals]
    print("Chart of Accuracies for each C:\n", np.transpose(np.array([cVals, accuracies])))
    bestC = 1
    bestAcc = 0
    for i in range(len(cVals)):
        if accuracies[i] >= bestAcc:
            bestC = cVals[i]
            bestAcc = accuracies[i]
    print("C is tuned to", bestC, ".")
    return bestC

def craftFoldedDataset(folds, i):
    trainingSet = np.vstack([folds[j] for j in range(len(folds)) if j != i])
    print(i)
    validationSet = folds[i]
    F = trainingSet.shape[1] - 1
    return trainingSet[:, :F], trainingSet[:, F], validationSet[:, :F], validationSet[:, F]

def writeTestResults(testset, model, filename):
    print("PREDICTING")
    testset.testSetY = model.predict(testset.testSetX)
    print("STACKING")
    print(testset.dataset.shape, testset.testSetY.shape)
    kaggleSubmission = np.hstack([np.arange(testset.N()).reshape(testset.N(), 1), testset.testSetY.reshape(testset.N(), 1)])
    print(kaggleSubmission)
    print("WRITING")
    np.savetxt(directory + filename, kaggleSubmission, delimiter = ",", fmt = "%10.5f")
    print("FIN")

def two():
    delta = 0.02
    X, Y = np.meshgrid(np.arange(-5, 5, delta), np.arange(-5, 5, delta))
    Z_a = mlab.bivariate_normal(X, Y, mux = 1, muy = 1, sigmax = np.sqrt(1), sigmay = np.sqrt(2), sigmaxy = np.sqrt(0))
    Z_b = mlab.bivariate_normal(X, Y, mux = -1, muy = 2, sigmax = np.sqrt(2), sigmay = np.sqrt(3), sigmaxy = np.sqrt(1))
    Z_c1 = mlab.bivariate_normal(X, Y, mux = 0, muy = 2, sigmax = np.sqrt(2), sigmay = np.sqrt(1), sigmaxy = np.sqrt(1))
    Z_c2 = mlab.bivariate_normal(X, Y, mux = 2, muy = 0, sigmax = np.sqrt(2), sigmay = np.sqrt(1), sigmaxy = np.sqrt(1))
    Z_d1 = mlab.bivariate_normal(X, Y, mux = 0, muy = 2, sigmax = np.sqrt(2), sigmay = np.sqrt(1), sigmaxy = np.sqrt(1))
    Z_d2 = mlab.bivariate_normal(X, Y, mux = 2, muy = 0, sigmax = np.sqrt(2), sigmay = np.sqrt(3), sigmaxy = np.sqrt(1))
    Z_e1 = mlab.bivariate_normal(X, Y, mux = 1, muy = 1, sigmax = np.sqrt(2), sigmay = np.sqrt(1), sigmaxy = np.sqrt(0))
    Z_e2 = mlab.bivariate_normal(X, Y, mux = -1, muy = -1, sigmax = np.sqrt(2), sigmay = np.sqrt(2), sigmaxy = np.sqrt(1))
    plt.subplot(321)
    plt.clabel(plt.contour(X, Y, Z_a), inline = 1, fontsize = 9)
    plt.title("(a)")
    plt.subplot(322)
    plt.clabel(plt.contour(X, Y, Z_b), inline = 1, fontsize = 9)
    plt.title("(b)")
    plt.subplot(323)
    plt.clabel(plt.contour(X, Y, Z_c1 - Z_c2), inline = 1, fontsize = 9)
    plt.title("(c)")
    plt.subplot(324)
    plt.clabel(plt.contour(X, Y, Z_d1 - Z_d2), inline = 1, fontsize = 9)
    plt.title("(d)")
    plt.subplot(325)
    plt.clabel(plt.contour(X, Y, Z_e1 - Z_e2), inline = 1, fontsize = 9)
    plt.title("(e)")
    plt.show()

def three():
    A = lambda n: npr.normal(loc = 3, scale = 9, size = n)
    B = lambda: npr.normal(loc = 2, scale = 2)

    sample = np.array([(x_1, ((0.5)*x_1 + B())) for x_1 in A(100)])
    sample_mean = np.mean(sample, axis=0)
    print("Mean: ", sample_mean)
    sample_cov = np.cov(sample.T)
    print("Sigma: ",sample_cov)
    evals, evecs = eig(sample_cov)
    print("Eigenvalues: ", evals)
    print("Eigenvectors: ", evecs)
    x,y = tuple(zip(*sample))
    plt.scatter([xp - sample_mean[0] for xp in x],[yp - sample_mean[1] for yp in y])
    ax = plt.axes()
    ax.arrow(0,0,evecs[0,0], evecs[1,0], head_length = 10, fc='k', ec='k')
    ax.arrow(0,0,evecs[0,1], evecs[1,1], head_length = 10, fc='k', ec='k')
    plt.ylim(-15,15)
    plt.xlim(-15,15)
    plt.show()
    sample_prime = (sample - sample_mean) @ evecs
    print(sample_prime)
    x,y = tuple(zip(*sample_prime))
    plt.scatter(x, y)
    plt.show()

def six():
    # Reading, and splitting, the training datasets.
    mnist = Dataset("mnist/train", parting = 1000, normalize = True)
    spam = SpamDataset("spam/spam_data")

    # a: Fitting the Guassian to the data
    a = QDA(mnist)

    # b: Visualizing the covariance matrix
    plt.imshow(a.sigma_hat[0], cmap='hot', interpolation = "nearest")
    plt.show()

    # c: Plotting accuracies
    plotAccuracies(mnist, lda = False, subplot = 212)
    plotAccuracies(mnist, lda = True, subplot = 211)
    plt.show()

    # 5: Final models
    mnistTest = Dataset("mnist/test", testSet = True)
    writeTestResults(mnistTest, LDA(mnist), "mnistTestResults.csv")
    spamTest = SpamDataset("spam/spam_data", testSet = True)
    writeTestResults(spamTest, LDA(spam), "spamTestResults.csv")

# Script
two()
three()
six()

