# Imports
import numpy as np
import numpy.random as npr
import scipy.io as spi
from scipy.special import expit
import sklearn.preprocessing as skp

sigmoid = expit
sigmoid_prime = lambda x: (sigmoid(x)*(1-sigmoid(x)))
tanh = lambda x: (np.tanh(x))
tanh_prime = lambda x: (1 - tanh(x)*tanh(x))
s = sigmoid
sp = sigmoid_prime
t = tanh
tp = tanh_prime
scaler = skp.StandardScaler()

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

        if(self.function == "training" or self.function == "validation"):
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

    def j(self, j, S = None):
        if not S:
            return self.X[:, j]
        else:
            return self.X[:, j][S]

    def sample(self, n):
        S = npr.choice(range(self.n), n, replace = False)
        return Dataset(data = np.hstack([self.X[S], self.y[S]]), function = self.function, labelsFirst = self.labelsFirst)

    def batches(self, B):
        S = np.arange(self.n)
        npr.shuffle(S)
        return [Dataset(data = self.data[S_i], function = self.function, labelsFirst = self.labelsFirst) for S_i in np.split(S[:np.floor(self.n/B)*B], np.floor(self.n/B))]

    def possible(self, j, S = None):
        """
        Returns the possible values for the jth feature (optionally, within the datapoints in S)
        """
        return present(np.unique(self.j(j,S)))

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

# letters_data: (['test_x', '__version__', 'train_y', 'train_x'])
class LettersDataset(Dataset):
    def parse(self, file, function = "generic", skipHeader = True):
        rawMat = spi.loadmat(file)
        if function == "test":
            return scaler.transform(rawMat["test_x"])
        else:
            scaler.fit(rawMat["train_x"])
            return np.concatenate([scaler.transform(rawMat["train_x"]), rawMat["train_y"]], axis = 1)

class Model:
    def __init__(self, *params):
        self.params = params

    def train(self, trainingSet):
        self.classes = trainingSet.classes()
        self.temp = trainingSet
        return self

    def predict(self, testSet):
        return np.array([npr.choice(self.classes) for _ in testSet.data])

    def computeAccuracy(self, dataset):
        return sum(self.predict(dataset.testify()) == dataset.y.reshape(dataset.n))/dataset.n

class NeuralNet(Model):
    def __init__(self, *params):
        self.H = 200
        self.I = 30
        self.B = 80
        self.epsilon_V = 0.005
        self.epsilon_W = 0.01
        self.decay_V = 0.5
        self.decay_W = 0.5
        self.decayRate = 8

    def train(self, trainingSet):
        super().train(trainingSet)
        self.k = len(trainingSet.classes())
        self.f = trainingSet.f
        self.V = npr.rand(self.H, self.f + 1)
        self.W = npr.rand(self.k, self.H + 1)
        
        for i in range(self.I):
            batches = trainingSet.batches(self.B)
            for b in range(len(batches)):
                # Forward
                batch = batches[b]
                X = batch.X
                X_ = alphize(X).T
                T = t(self.V @ X_)
                T_ = alphize(T, addCol = False)
                S = s(self.W @ T_)
                Y = onehotencoder(batch.y, k = self.k)
               
                # Backward
                l_T = self.W.T @ (S - Y) # 201.26 @ 26.B = 201.B -> 200.B 
                l_W = (S - Y) @ T_.T # 26.B @ B.201 = 26.201
                l_V = T @ T_.T @ l_T @ X_.T # 200.B @ B.201 @ 201.B @ B.f+1 = 200.f+1 
                
                # Updates
                self.W = self.W - self.epsilon_W*l_W 
                self.V = self.V - self.epsilon_V*l_V 

                if b % 10000000000 == 0:
                    print(i, self.computeAccuracy(self.temp), (-1/self.B)*sum([sum([Y[j, i]*np.log(S[j, i]) + (1 - Y[j, i])*np.log(1- S[j, i]) for j in range(self.k)]) for i in range(self.B)]))
            
            #print(i, self.epsilon_V, self.epsilon_W)
            if (i+1) % self.decayRate == 0:
                self.epsilon_W = self.epsilon_W * self.decay_W
                self.epsilon_V = self.epsilon_V * self.decay_V

        return self

    def predict(self, testSet):
        return onehotdecoder(s(self.W @ alphize(t(self.V @ alphize(testSet.X).T), addCol = False)))

# Helper Methods
def validation(dataset, modelType, HPValues):
    trainingSet, validationSet = dataset.split()
    def tupleWrapper(ele):
        if type(ele) is tuple:
            return ele
        else:
            return (ele,)
    return np.array([(repr(HPV), modelType(*tupleWrapper(HPV)).train(trainingSet).computeAccuracy(validationSet)) for HPV in HPValues])

def onehotdecoder(v):
    """ Takes the k by n matrix y_hat_prime and returns the n by 1 vector y_hat """
    return np.argpartition(-v.T, 0)[:, 0] + 1

def onehotencoder(v, k):
    """ Takes the n by 1 vector y and returns the k by n matrix y_prime """
    return skp.OneHotEncoder(n_values = k, sparse = False).fit_transform((v-1).reshape(len(v), 1)).T

def alphize(A, addCol = True):
    r = A.shape[0]
    c = A.shape[1]
    if addCol:
        return np.hstack([A, np.ones(r).reshape(r, 1)])
    else:
        return np.vstack([A, np.ones(c).reshape(1, c)])