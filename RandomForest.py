# Imports
import numpy as np
import numpy.random as npr
import scipy.io as spi
import csv
import math
sqrt = lambda x: (math.floor(math.sqrt(x)))
lg = lambda x: (math.log(x,2))

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

    def possible(self, j, S = None):
        """
        Returns the possible values for the jth feature (optionally, within the datapoints in S)
        """
        return present(np.unique(self.j(j,S)))

    def ij(self, i, j, defaultValue = 0):
        element = self.X[i,j]
        if element == "?" or element == "":
            return defaultValue
        else:
            return element 

    def split(self, validationSize = .2):
        """
        Returns the tuple, (training, validation)
        """
        if validationSize <= 1:
            C = math.floor(validationSize*self.n)
        else:
            C = validationSize
        return Dataset(data = self.data[C:], function = "training", labelsFirst = self.labelsFirst), Dataset(data = self.data[:C], function = "validation", labelsFirst = self.labelsFirst)

    def parse(self, file, function = "generic", skipHeader = True):
        with open(file + ".csv") as csvfile:
            reader = csv.reader(csvfile)
            if skipHeader:
                next(reader)
            return np.array([row for row in reader])

    def guess(self, j, S):
        return average(present(self.j(j, S).flatten())) or 0

    def clean(self):
        def containsQuestionMark(row):
            for element in row:
                if element == "?" or element == "":
                    return True
            return False
        return Dataset(data = np.array([row for row in self.data if not containsQuestionMark(row)]), function = self.function, labelsFirst = self.labelsFirst)

    def __str__(self, includeData = False):
        d = ""
        if includeData:
            d +=  "\nData:\n" +self.data.__str__()
        if(self.function == "test"):
            return "\nFunction:\n" + self.function + d
        else:
            return "\nFunction:\n" + self.function + d + "\nClasses:\n" + self.classes().__str__() + "\nX Shape:\n" + self.X.shape.__str__()

class SpamDataset(Dataset):
    def parse(self, file, function = "generic", skipHeader = True):
        rawMat = spi.loadmat(file)
        if function == "test":
            return rawMat["test_data"]
        else:
            return np.concatenate([rawMat["training_data"], np.transpose(rawMat["training_labels"])], axis = 1)

class Model:
    def __init__(self, *params):
        self.params = params
        self.stoppingDepth = float("inf")
        self.m = 3
        self.N = 10
        if len(self.params) > 0:
            self.stoppingDepth = self.params[0]
        if len(self.params) > 1:
            self.m = self.params[1]
        if len(self.params) > 2:
            self.N = self.params[2]

    def train(self, trainingSet):
        self.classes = trainingSet.classes()
        return self

    def predict(self, testSet):
        return np.array([npr.choice(self.classes) for _ in testSet.data])

    def computeAccuracy(self, dataset):
        return sum(self.predict(dataset.testify()) == dataset.y.reshape(dataset.n))/dataset.n

class FrequencyDistribution:
    def __init__(self, lst):
        self.elements = lst
        self.categories = present(np.unique([l for l in lst if l is not None]))
        self.counts = {category: sum([element == category for element in self.elements]) for category in self.categories}
        self.props = {category: self.count(category)/len(self.elements) for category in self.categories}

    def count(self, category):
        return self.counts[category]
   
    def p(self, category):
        return self.props[category]

    def mode(self):
        modalCategory = None
        numModalCategory = 0
        for category in self.categories:
            numCategory = self.count(category)
            if numCategory > numModalCategory:
                modalCategory = category
                numModalCategory = numCategory
        return modalCategory

    def entropy(self):
        return - sum([self.p(c)*lg(self.p(c)) for c in self.categories])

    def __str__(self):
        return "\nCategories:\n" + self.categories.__str__() + "\nCounts:\n" + self.counts.__str__()

class Node:
    def __init__(self, *contents, depth = 0, left = None, right = None):
        self.contents = contents
        self.depth = depth
        self.left = left
        self.right = right

    def isLeaf(self):
        return not (left or right)

    def __str__(self):
        printedString = "\t"*self.depth + self.contents.__str__() + "\n"
        if self.left:
            printedString += self.left.__str__()
        if self.right:
            printedString += self.right.__str__()
        return printedString

class DecisionNode(Node):
    def __init__(self, *contents, depth = 0, left = None, right = None):
        """
        If leaf, send one *contents: label (a string like "S")
        If not leaf, send two *contents: splitFeature (an index), splitRule (a string like "> 3")
        """
        super().__init__(*contents, depth = depth, left = left, right = right)
        if len(contents) == 1:
            self.label = self.contents[0]
        else:
            self.splitFeature = self.contents[0]
            self.splitRule = self.contents[1]
            self.defaultValue = self.contents[2]

class DecisionTree(Model):
    def train(self, trainingSet, dropout = False):
        def grow(S = list(range(trainingSet.n)), currDepth=0):
            """
            S is a list of indices
            """
            def findSplit():
                """
                Returns the (splitFeature, splitRule) that maximizes information gain
                """
                features = list(range(trainingSet.f))
                if dropout:
                    features = npr.choice(features, self.m, False)

                bestFeature, bestFeatureEntropy, bestRule, bestLeft, bestRight = 0, float('inf'), "is None", [], S
                for feature in features:
                    print("FEAT", feature)
                    categories = trainingSet.possible(feature, S)
                    defaultValue = trainingSet.guess(feature, S)
                    if(not containsString(categories)):
                        if len(categories) > 10:
                            slide = tenway(np.sort(categories))
                        else:
                            slide = np.sort(categories)
                        slider = 1
                        left = []
                        right = S.copy()
                        while slider < len(slide):
                            pivot = slide[slider]
                            newRight = []
                            for i in right:
                                if trainingSet.ij(i, feature, defaultValue) < pivot:
                                    left += [i]
                                else:
                                    newRight +=[i]
                            right = newRight
                            featureEntropy = entropy(left, right)
                            if featureEntropy < bestFeatureEntropy: 
                                bestFeature = feature
                                bestFeatureEntropy = featureEntropy 
                                bestRule = "< " + repr(pivot)
                                bestLeft = left
                                bestRight = right
                            slider+=1
                    else:
                        for category in categories:
                            rule = "== " + repr(category)
                            left = eval("[i for i in S if trainingSet.ij(i, feature, defaultValue) " + rule + "]", locals())
                            right = eval("[i for i in S if not trainingSet.ij(i, feature, defaultValue) " + rule + "]", locals())
                            featureEntropy = entropy(left, right)
                            if featureEntropy < bestFeatureEntropy: 
                                bestFeature = feature
                                bestFeatureEntropy = featureEntropy
                                bestRule = rule 
                                bestLeft = left
                                bestRight = right
                return bestFeature, bestRule, bestLeft, bestRight, trainingSet.guess(bestFeature, S)
        
            def entropy(left, right):
                return len(left)*FrequencyDistribution(trainingSet.classof(left)).entropy() + len(right)*FrequencyDistribution(trainingSet.classof(right)).entropy()
            def pure(S):
                return len(np.unique(trainingSet.classof(S))) == 1

            if pure(S) or currDepth >= self.stoppingDepth:
                return DecisionNode(FrequencyDistribution(trainingSet.classof(S)).mode(), depth = currDepth)
            else:
                splitFeature, splitRule, S_l, S_r, defaultValue = findSplit()
                return DecisionNode(splitFeature, splitRule, defaultValue, depth = currDepth, left = grow(S = S_l, currDepth = currDepth+1), right = grow(S = S_r, currDepth = currDepth+1))
        super().train(trainingSet)
        self.tree = grow()
        return self

    def predict(self, testSet, printOut = False):
        def predictHelper(testSet, currentNode, i):
            if hasattr(currentNode, "label"):
                if printOut:
                    print("So final prediction is: ", currentNode.label)
                return currentNode.label
            else:
                if eval("testSet.ij(i, currentNode.splitFeature, currentNode.defaultValue) " + currentNode.splitRule, locals()):
                    if printOut:
                        print("Feature ", currentNode.splitFeature, " is ", currentNode.splitRule, " (", testSet.ij(i, currentNode.splitFeature), ")")
                    return predictHelper(testSet, currentNode.left, i)
                else:
                    if printOut:
                        print("Feature ", currentNode.splitFeature, " is not ", currentNode.splitRule, " (", testSet.ij(i, currentNode.splitFeature), ")")
                    return predictHelper(testSet, currentNode.right, i)
        return np.array([predictHelper(testSet, self.tree, i) for i in range(testSet.n)])

    def __str__(self):
        return str(self.tree)

class RandomForest(Model):
    def train(self, trainingSet):
        super().train(trainingSet)
        self.trees = [DecisionTree(*self.params).train(trainingSet, dropout = True) for _ in range(self.N)]
        self.treeAccuracies = [tree.computeAccuracy(trainingSet) for tree in self.trees]
        self.commonRules = FrequencyDistribution([tree.tree.splitFeature.__str__() + ": " + tree.tree.splitRule for tree in self.trees])
        return self

    def predict(self, testSet):
        return average2D(np.hstack([tree.predict(testSet).reshape(testSet.n, 1) for tree in self.trees]))

    def __str__(self):
        return str(self.treeAccuracies)

# Helper Methods
def validation(dataset, modelType, HPValues):
    trainingSet, validationSet = dataset.split()
    def tupleWrapper(ele):
        if type(ele) is tuple:
            return ele
        else:
            return (ele,)
    return np.array([(repr(HPV), modelType(*tupleWrapper(HPV)).train(trainingSet).computeAccuracy(validationSet)) for HPV in HPValues])

def average(lst):
    if len(lst) == 0:
        return 0
    try:
        return sum(lst)/len(lst)
    except:
        return FrequencyDistribution(lst).mode()

def average2D(mat):
    return np.array([average([mat[i,j] for j in range(mat.shape[1])]) for i in range(mat.shape[0])])

def present(lst):
    return [element for element in lst if element != "" and element != "?"]

def containsString(lst):
    for element in lst:
        if not number(element):
            return True
    return False

def number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def tenway(lst):
    return lst[:: math.floor(len(lst)/10)]
