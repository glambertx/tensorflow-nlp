
import numpy as np
import random
from cs224d.data_utils import *
from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad


def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length

    ### YOUR CODE HERE
    x = x/ np.sqrt(np.sum(x**2,axis=1,keepdims=True))
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ""



def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, assuming the softmax prediction function and cross
    # entropy loss.

    # Inputs:
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word
    # - outputVectors: "output" vectors (as rows) for all tokens
    # - dataset: needed for negative sampling, unused here.

    # Outputs:
    # - cost: cross entropy cost for the softmax word prediction
    # - gradPred: the gradient with respect to the predicted word
    #        vector
    # - grad: the gradient with respect to all the other word
    #        vectors

    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    """
    # Calculate the gradient for r_hat
    r = predicted
    # Make the prediction
    pred_denom = np.sum(np.exp(np.dot(outputVectors, r)))
    pred_num = np.exp(np.dot(outputVectors, r))
    predictions = pred_num/pred_denom
    # Calculate the cost
    cost = -np.log(predictions[target])
    # Calculate deltas
    z = predictions.copy()
    z[target] -= 1.
    gradPred = np.dot(outputVectors.T, z)
    grad = np.outer(z, r)
    """


    ## softmax and loss for target word vector
    # y(1xV), wp(1xV), cost(1x1)
    vc = predicted
    u = outputVectors
    y = np.dot(u, vc)
    wp = softmax(y)
    cost = -np.log(wp[target])
    delta = wp
    delta[target] -= 1

    ## gradients
    gradPred = np.dot(u.T, delta)
    grad = np.outer(delta, vc)

    return cost, gradPred, grad


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    """ Negative sampling cost function for word2vec models """

    ## set-up some variables
    sample=[dataset.sampleTokenIdx() for i in range(K)]
    vc = predicted
    u = outputVectors
    gradPred = np.zeros_like(vc)
    nsample = dataset

    #initial cost
    y1 = np.dot(u[target], vc.T)
    sigma1 = sigmoid(y1)
    cost = -np.log(sigma1)
    # gradients

    grad = np.zeros((outputVectors.shape[0], outputVectors.shape[1]))
    gradPred = -u[target]*(1-sigma1)

    #update loop

    for i in sample:
        y2 = np.dot(u[i],vc)
        sigma2 = sigmoid(y2)
        cost -= np.log(1-sigma2)
        grad[i] += vc*sigma2
        gradPred += u[i]*sigma2

    grad[target] += -vc*(1-sigma1)

    return cost, gradPred, grad



def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    cost=0.0
    gradIn=np.zeros_like(inputVectors)
    gradOut=np.zeros(outputVectors.shape)
    current_word_index = tokens[currentWord]
    r=inputVectors[tokens[currentWord]]

    for i in contextWords:
       target=tokens[i]
       cost_0, gradIn_0, gradOut_0 =word2vecCostAndGradient(r, target,outputVectors,dataset)
       cost +=cost_0
       gradIn[current_word_index]+=gradIn_0
       gradOut+=gradOut_0

    N = len(contextWords)
    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    N=len(contextWords)
    return cost/N, gradIn/N, gradOut/N #here divide by N


def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)]            for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    #print "\n==== Gradient check for CBOW      ===="
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)
    #print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    #print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
