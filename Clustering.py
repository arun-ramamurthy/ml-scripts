# Imports
import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import scipy.io as spi
from scipy.special import expit
import sklearn.preprocessing as skp

# Classes
class Dataset:
    def __init__(self, file = "", data = None, function = "generic", labelsFirst = False):
        """ 
        Initializes a Dataset object with a particular function, or purpose.
        Pass either a file name or a dataset, as well as a label of the dataset's function.
        """
        if data is not None:
            self.data = data
        else:
            self.data = self.parse(file, function)
        self.function = function
        self.labelsFirst = labelsFirst
        self.n = self.data.shape[0]
        self.f = self.data.shape[1]
        self.X = self.data[:,:]

        if(self.function == "training" or self.function == "validation" or self.function == "generic_labelled"):
            npr.shuffle(self.data)
            self.f -= 1
            if labelsFirst:
                self.y = self.data[:, 0].reshape(self.n, 1)
                self.X = self.data[:, 1:]
            else:
                self.y = self.data[:, self.f].reshape(self.n, 1)
                self.X = self.data[:, :self.f]

    def testify(self):
        return Dataset(data = self.X, function = "test")

    def tiny(self):
        return Dataset(data = self.data[npr.choice(range(self.n))].reshape(1, self.data.shape[1]), function = self.function, labelsFirst = self.labelsFirst) 

    def classes(self):
        return np.unique(self.y.reshape(self.n))

    def classof(self, i):
        if self.function != "testing":
            return self.y[i, 0]

    def ofclass(self, y):
        X = np.array([self.X[i] for i in range(self.n) if self.y[i] == y])
        y = (np.ones(len(X))*y).reshape(len(X), 1).astype(int)
        return Dataset(data = np.hstack((y, X)), function = "generic_labelled", labelsFirst = True)

    def i(self, i, X = True):
        if X:
            return self.X[i]
        else:
            return self.data[i]

    def j(self, j, S = None):
        if not S:
            return self.X[:, j]
        else:
            return self.X[:, j][S]

    def sample(self, n):
        S = npr.choice(range(self.n), n, replace = False)
        return Dataset(data = np.hstack([self.X[S], self.y[S]]), function = self.function, labelsFirst = self.labelsFirst)

    def ij(self, i, j, defaultValue = 0):
        return self.X[i,j]

    def split(self, validationSize = .2):
        """
        Returns the tuple, (training, validation)
        """
        if validationSize <= 1:
            C = np.floor(validationSize*self.n)
        else:
            C = validationSize
        return Dataset(data = self.data[C:], function = "training", labelsFirst = self.labelsFirst), Dataset(data = self.data[:C], function = "validation", labelsFirst = self.labelsFirst)

    def parse(self, file, function = "generic", skipHeader = True):
        with open(file + ".csv") as csvfile:
            reader = csv.reader(csvfile)
            if skipHeader:
                next(reader)
            return np.array([row for row in reader])

    def __str__(self, includeData = True):
        d = ""
        if includeData:
            d +=  "\nData:\n" +self.data.__str__()
        if(self.function == "test"):
            return "\nFunction:\n" + self.function + d
        else:
            return "\nFunction:\n" + self.function + d + "\nClasses:\n" + self.classes().__str__() + "\nX Shape:\n" + self.X.shape.__str__()

class JokeDataset(Dataset):
    def parse(self, file, function = "generic", skipHeader = True):
        rawMat = spi.loadmat(file)
        print(rawMat["__header__"])
        print(rawMat.keys())
        if function == "test":
            return rawMat["test_x"]
        else:
            return rawMat["train"]

class NumbersDataset(Dataset):
    def parse(self, file, function = "generic", skipHeader = True):
        rawMat = spi.loadmat(file)
        return rawMat["images"].reshape(28*28, 60000).T

class KMeans:
    def __init__(self, *args, data, k = 2):
        self.k = k
        self.data = Dataset(data = data.X.copy())
        self.data.y = self.randomizedClusters()
        self.centers = [self.mu(cluster) for cluster in range(k)]

    def optimize(self, iterations = 10):
        def update_mu():
            self.centers = [self.mu(cluster) for cluster in range(self.k)]

        def update_y():
            for i in range(self.data.n):
                point = self.data.i(i)
                currentDistance = lin.norm(point - self.centers[self.data.y[i]])
                for y in range(self.k):
                    potentialDistance = lin.norm(point - self.centers[y])
                    if potentialDistance < currentDistance:
                        self.data.y[i] = y
            
        for iteration in range(iterations):
            print(iteration, self.objFn())
            update_mu()
            update_y()


    def objFn(self):
        def clusterObjFn(y):
            return sum([np.square(lin.norm(point - self.centers[y])) for point in self.cluster(y).X])
        return sum(clusterObjFn(y) for y in range(self.k)) 

    def mu(self, y):
        return np.mean(self.cluster(y).X, axis = 0)

    def cluster(self, y):
        return self.data.ofclass(y)

    def randomizedClusters(self):
        setting = np.trunc(npr.rand(self.data.n)*self.k).astype(int).reshape(self.data.n, 1)
        if np.all(np.in1d(range(self.k), setting)):
            return setting
        return self.randomizedClusters()

class LowRankApproximation:
    def __init__(self, matrix):
        self.matrix = matrix
        self.U, self.s, self.V = lin.svd(matrix)

    def approximate(self, k):
        reducedS = np.zeros((self.U.shape[1], self.V.shape[0]))
        reducedS[:k, :k] = np.diag(self.s[:k])
        return self.U @ reducedS @ self.V

    def mse(self, k):
        return lin.norm(self.matrix.flatten() - self.approximate(k).flatten())