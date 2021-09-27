# Imports
import pandas
import numpy

c = 1  # Parameter of This Function


# Function to Evaluate Absolute Value Of x
def quadRad(x):
    return x / numpy.sqrt(numpy.power(x,2) + numpy.power(c,2))


# Import .csv File
fileName = "../digits.csv"
allData = pandas.read_csv(fileName, header=None)
allData = allData.values

# Defines
nF = 16

# Separating Student Numbers: 0 & 2
num0and2 = numpy.zeros(0)
for i in range(allData.size/(nF+1)):
    if (allData[i][nF] == 0) | (allData[i][nF] == 2):
        if num0and2.size == 0:
            num0and2 = allData[i]
        else:
            num0and2 = numpy.vstack([num0and2, allData[i]])

dataLength = num0and2.size/(nF+1)

# Transforming Labels from (0,2) to (-1,1)
allDataLabeled = num0and2
for i in range(dataLength):
    if num0and2[i][nF] == 2:
        allDataLabeled[i][nF] = 1
    else:
        allDataLabeled[i][nF] = -1

# Separating TrainFeatures, TestFeatures, TrainData, TestData
dataLabel = allDataLabeled[:, nF]
allDataFeatures = allDataLabeled[:, range(nF)]
# add 1 to the end of x : [x1, x2, ..., xn, 1]
o1 = numpy.ones((dataLength, 1))
o1 = o1.astype(int)
allDataFeatures = numpy.hstack([allDataFeatures, o1])

trainLength = int(dataLength*0.8)
trainData = allDataFeatures[range(trainLength)]
testData = allDataFeatures[range(trainLength, dataLength)]

trainLabel = dataLabel[range(trainLength)]
testLabel = dataLabel[range(trainLength, dataLength)]


# Random Initial W
w = numpy.random.random(nF+1)-0.52


# Gradient Descending On Perceptron
e = 0.01
ctr = 0
nr = 1
while nr > 0:
    ctr = ctr + 1
    gred = numpy.zeros(nF+1)
    nr = 0
    wxMatrix = numpy.dot(trainData, numpy.transpose(w))
    wxSignMatrix = numpy.sign(wxMatrix)
    differences = (wxSignMatrix != numpy.transpose(trainLabel))
    for i in range(trainLength):
        if differences[i]:
            nr = nr + 1
            gred = gred + quadRad(wxMatrix[i]) * trainData[i]
    w = w - e * gred

# Validation On Test Data
wxTestMatrix = numpy.dot(testData, numpy.transpose(w))
wxSignTestMatrix = numpy.sign(wxTestMatrix)
testError = (wxSignTestMatrix != numpy.transpose(testLabel))
nErr = 0
for i in range(testError.size):
    if testError[i]:
        nErr = nErr + 1

print "** Perceptron With Sqrt(x2 + c2) Function as Absolute Value **"
print "Number Of Moves: ", ctr
print "Test Error:", nErr, "In", testError.size, "Test Data =", 100*float(nErr)/testError.size, "%"
