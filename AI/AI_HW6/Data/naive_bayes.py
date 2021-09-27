import numpy as np


# DO NOT TOUCH THIS #
def loadmat(filename):
    flat_array = np.load(filename)
    nr, nc = flat_array[0], flat_array[1]
    data = np.zeros((nr.astype(int), nc.astype(int)))
    for val in flat_array[2:]:
        r, c = int(val // nc), int(val % nc)
        data[r, c] = 1.
    return data


def NB_XGivenY(XTrain, yTrain):
    """This function computes the Conditional Distribution of the words
    given the class labels.

        N: Number of data points
        D: Number of Features

    Arguments:
        XTrain (np.ndarray): The Training Data matrix of shape N x D
            Each row is an article, with X_{i,j} = 1 -> ith article contains
            jth word.
        yTrain (np.ndarray): The Label matrix of shape N x 1
            0 -> The article comes from The Economist
            1 -> The article comes from The Onion
    Returns:
        D (np.ndarray): The log conditional probability distribution of shape
            2 x D, where for any word index w \in {1, .. V}, and class
            y \in {0, 1} the entry D(y, w) is the MAP estimate of
            \theta_{y,w} = \log\left[P(X_{w} = 1 | Y=y)\right]
            (We work in log space)

    """
    raise NotImplementedError("TODO")


def NB_YPrior(yTrain):
    """
        This function computes the MLE Prior for P(Y = 1)

    Arguments:
        (np.ndarray): The Label matrix of shape N x 1
            0 -> The article comes from The Economist
            1 -> The article comes from The Onion

    Returns:
        logp (float): The prior (\log\left[P(Y=1)\right])

    """
    raise NotImplementedError("TODO")


def NB_Classify(X, D, logp):
    """Given the training data, the estimate of the log conditional
    probabilites, and the log priors, compute the predictions for each data
    point

        N: Number of data points
        D: Number of Features

    Arguments:
        X (np.ndarray): The Data matrix of shape N x D
            Each row is an article, with X_{i,j} = 1 -> ith article contains
            jth word.
        D (np.ndarray): The log conditional probability distribution of shape
            2 x D
        logp (float): The prior (\log\left[P(Y=1)\right])
    Returns:
        y_hat (np.ndarray): The predicted label matrix of shape N x 1
    """
    raise NotImplementedError("TODO")


def ClassificationError(y_pred, y_gold):
    """Computes the accuracy, given predictions and gold labels

    Arguments:
        y_pred (np.ndarray): The predicitons of shape N x 1, with each
            element being 0 or 1
        y_gold (np.ndarray): The gold labels of shape N x 1, with each
            element being 0 or 1

    Return:
        accuracy (float): The accuracy
    """
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    pass
