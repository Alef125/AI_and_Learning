import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.svm import SVR
import hsic
import pandas as pd


# Function of Generating Y Caused by X
def yGen(x, b, q):
    meanNy = 0
    sigmaNy = 1
    ny = np.random.normal(meanNy, sigmaNy, x.size)
    ny = np.sign(ny) * np.power(np.abs(ny), q)
    y = x + b * np.power(x, 3) + ny
    return y


# Function for Plotting
def stochasticPlot(x, y, kind):
    # Functionality:
    # kind = 0: plots p(x,y) on (x,y)
    # kind = 1: plots p(x) on x
    # kind = 2: plots p(y) on y
    # kind = 3: plots p(y|x) on (x,y)
    # kind = 4: plots p(x|y) on (x,y)
    # kind = 5: plots p(x,y - E(Y|x)) on (x,y)
    # kind = 6: plots p(x - E(X|y),y) on (x,y)
    # kind = 7: plots p(y|x - E(Y|x)) on (x,y)
    # kind = 8: plots p(x|y - E(X|y)) on (x,y)
    bins = 500  # Number Of dXs and dYs in differential plotting
    nTotal, xEdges, yEdges = np.histogram2d(x, y, bins=bins)
    nTotal = nTotal.T
    dX = (xEdges[-1] - xEdges[0]) / bins
    dY = (yEdges[-1] - yEdges[0]) / bins
    DimX = xEdges.size - 1
    DimY = yEdges.size - 1
    nX0 = np.argmin(np.abs(xEdges))
    nY0 = np.argmin(np.abs(yEdges))
    pTotal = (nTotal / np.sum(nTotal)) / (dX * dY)
    xMiddles = (xEdges[0:DimX] + xEdges[1:DimX+1]) / 2
    yMiddles = (yEdges[0:DimY] + yEdges[1:DimY+1]) / 2
    sumX = np.sum(nTotal, axis=0)
    sumY = np.sum(nTotal, axis=1)
    pX = (sumX / np.sum(sumX)) / dX
    pY = (sumY / np.sum(sumY)) / dY
    pyGinenX = (pTotal.T / pX[:, None]).T
    pxGinenY = pTotal / pY[:, None]
    eyGivenX = dY * np.dot(pyGinenX.T, yMiddles)
    exGivenY = dX * np.dot(pxGinenY, xMiddles)
    maxPyOnX = dY *(np.argmax(pyGinenX, axis=0)-nY0)
    maxPxOnY = dX * (np.argmax(pxGinenY, axis=1)-nX0)
    if kind == 0:
        plt.pcolor(xMiddles, yMiddles, pTotal, norm=LogNorm())
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("p(x,y)")
        plt.colorbar()
    elif kind == 1:
        plt.plot(xMiddles, pX)
        plt.xlabel("x")
        plt.ylabel("p(x)")
    elif kind == 2:
        plt.plot(yMiddles, pY)
        plt.xlabel("y")
        plt.ylabel("p(y)")
    elif kind == 3:
        plt.pcolor(xMiddles, yMiddles, pyGinenX, norm=LogNorm())
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("p(y|x)")
        plt.colorbar()
    elif kind == 4:
        plt.pcolor(xMiddles, yMiddles, pxGinenY, norm=LogNorm())
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("p(x|y)")
        plt.colorbar()
    elif kind == 5:
        pNew = pTotal * 0
        nyShift = np.round(eyGivenX / dY)
        for i in range(DimX):
            nShift = -nyShift[i]
            if np.isnan(nShift):
                nShift = 0
            nShift = np.int(nShift)
            nShift = nShift % DimY
            pNew[:, i] = np.roll(pTotal[:, i], nShift)
        plt.pcolor(xMiddles, yMiddles, pNew, norm=LogNorm())
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("p(x,y - E(Y|x))")
        plt.colorbar()
    elif kind == 6:
        pNew = pTotal * 0
        nxShift = np.round(exGivenY / dX)
        for i in range(DimY):
            nShift = -nxShift[i]
            if np.isnan(nShift):
                nShift = 0
            nShift = np.int(nShift)
            nShift = nShift % DimX
            pNew[i, :] = np.roll(pTotal[i, :], nShift)
        plt.pcolor(xMiddles, yMiddles, pNew, norm=LogNorm())
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("p(x - E(X|y),y)")
        plt.colorbar()
    elif kind == 7:
        pNew = pTotal * 0
        nyShift = np.round(eyGivenX / dY)
        for i in range(DimX):
            nShift = -nyShift[i]
            if np.isnan(nShift):
                nShift = 0
            nShift = np.int(nShift)
            nShift = nShift % DimY
            pNew[:, i] = np.roll(pyGinenX[:, i], nShift)
        plt.pcolor(xMiddles, yMiddles, pNew, norm=LogNorm())
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("p(y|x - E(Y|x))")
        plt.colorbar()
    elif kind == 8:
        pNew = pTotal * 0
        nxShift = np.round(exGivenY / dX)
        for i in range(DimY):
            nShift = -nxShift[i]
            if np.isnan(nShift):
                nShift = 0
            nShift = np.int(nShift)
            nShift = nShift % DimX
            pNew[i, :] = np.roll(pxGinenY[i, :], nShift)
        plt.pcolor(xMiddles, yMiddles, pNew, norm=LogNorm())
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("p(x|y - E(X|y))")
        plt.colorbar()
    plt.show()


# Function to Give Y - f(Y|x)
def pureDist(x, y):
    # f = SVR(kernel="poly", C=1, degree=3)
    f=SVR()
    f.fit(x.reshape(-1, 1), y)  # Numpy will find -1 = x.size
    yNoisy = y - f.predict(x.reshape(-1, 1))
    xLine = np.linspace(np.min(x)-1, np.max(x)+1, 1000)
    plt.scatter(x, y, s=1, color='b')
    plt.plot(xLine, f.predict(xLine.reshape(-1, 1)), color='r')
    plt.scatter(x, yNoisy, s=1, color='g')
    return yNoisy


# Confidence
def isIndependent(x, y, cLevel):
    yNoisy = pureDist(x, y)
    testStat, threshold = hsic.hsic_gam(x.reshape(-1, 1), yNoisy.reshape(-1, 1), alph=cLevel)
    isIndep = (testStat <= threshold)
    # dependencyComment = "Dependent"
    # if isIndep:
    #     dependencyComment = "Independent"
    # print "TestStat=", testStat, ", And Threshold=", threshold, " So Variables are: ", dependencyComment
    return isIndep


# Part 1 Session 0, Plotting
def part1Plotting():
    # Generating X Distribution: Nx = N(0, 1)
    meanNX = 0
    sigmaNX = 1
    NX = np.random.normal(meanNX, sigmaNX, 100000)
    # Calculating Y = X + bX^3 + sign(X)*|X|^q
    X = NX
    Y = yGen(X, b=0, q=1)
    stochasticPlot(X, Y, kind=0)

# Part 1 Session 1, X+X^3 Causal Test
def hsicTest():
    # Generating X Distribution: Nx = N(0, 1)
    meanNX = 0
    sigmaNX = 1
    yDependencyOfX = 0
    xDependencyOfY = 0
    for i in range(100):
        NX = np.random.normal(meanNX, sigmaNX, 300)
        # Calculating Y = X + bX^3 + sign(X)*|X|^q
        X = NX
        Y = yGen(X, b=0, q=2)
        if isIndependent(X, Y, cLevel=0.02):
            yDependencyOfX = yDependencyOfX + 1
        if isIndependent(Y, X, cLevel=0.02):
            xDependencyOfY = xDependencyOfY + 1
    print "Y on X Direction Dependencies = ", yDependencyOfX, \
        "ones, And X on Y Direction Dependencies = ", xDependencyOfY, "ones"

# Part 1 Session 2, Eruption Causal Test
def eruptionTest():
    fileName = "eruptions.csv"
    eruptionsData = pd.read_csv(fileName)
    eruptionDuration = np.array(eruptionsData['eruptions'])
    eruptionWaiting = np.array(eruptionsData['waiting'])
    dir1 = isIndependent(eruptionDuration, eruptionWaiting, 0.02)
    dir2 = isIndependent(eruptionWaiting, eruptionDuration, 0.02)
    if dir1 & ~dir2:
        print "Waiting is Cause of Duration"
    elif ~dir1 & dir2:
        print "Duration is Cause of Waiting"
    else:
        print "Undefined Causality"


# Part 1 Session 3, Abalone Causal Test
def abaloneTest():
    fileName = "abalone.csv"
    abaloneColumns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                      'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    abaloneData = pd.read_csv(fileName, names=abaloneColumns)
    abaloneLength = np.array(abaloneData['Length'])
    abaloneRings = np.array(abaloneData['Rings'])
    dir1 = isIndependent(abaloneLength, abaloneRings, 0.02)
    dir2 = isIndependent(abaloneRings, abaloneLength, 0.02)
    if dir1 & ~dir2:
        print "Length is Cause of Rings"
    elif ~dir1 & dir2:
        print "Rings is Cause of Length"
    else:
        print "Undefined Causality"

# Part 2, Detecting DAG
def directingDAG():
    fileName = "dag.csv"
    dagData = pd.read_csv(fileName)
    w = np.array(dagData['w'])
    x = np.array(dagData['x'])
    y = np.array(dagData['y'])
    z = np.array(dagData['z'])
    # plt.scatter(y, z, s=1, color='y')
    # plt.xlabel("Y")
    # plt.ylabel("Z")
    # plt.show()
    alpha = 0.02
    print "is W-->X ? :", isIndependent(w, x, cLevel=alpha)
    print "is X-->W ? :", isIndependent(x, w, cLevel=alpha)
    print "is W-->Y ? :", isIndependent(w, y, cLevel=alpha)
    print "is Y-->W ? :", isIndependent(y, w, cLevel=alpha)
    print "is X-->Z ? :", isIndependent(x, z, cLevel=alpha)
    print "is Z-->X ? :", isIndependent(z, x, cLevel=alpha)
    print "is Y-->Z ? :", isIndependent(y, z, cLevel=alpha)
    print "is Z-->Y ? :", isIndependent(z, y, cLevel=alpha)


# Main :
# part1Plotting()
# hsicTest()
# eruptionTest()
# abaloneTest()
directingDAG()
