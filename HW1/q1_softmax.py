import numpy as np
import random

def softmax(x):

    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the
    written assignment!
    """

    ### YOUR CODE HERE
    if len(np.shape(x)) >= 2:
        grt = np.max(x, axis=1)
        grt = grt.reshape(grt.shape[0],1)
        exp = np.exp(x - grt)
        x = exp/np.sum(exp, axis=1,keepdims=True)
        return x
    else:
        grt = np.max(x)
        exp = np.exp(x - grt)
        x = exp/np.sum(exp)
        return x
    ### END YOUR CODE


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print "You should verify these results!\n"


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE

    test4 = softmax(np.array([[12,11,4,3],[12,11,4,3]]))
    test5 = softmax(np.array([12,11,4,3]))
    test6 = softmax(np.array([0,100]))

    # does the matrix version of softmax corretly calculate each row
    # both rows should be the same
    print test4

    # check that probabilities for each row sum to 1
    print np.sum(test4, axis=1)
    print np.sum(test5)


    #check for acurate probabilities compare to logistic function

    ### END YOUR CODE



if __name__ == "__main__":
    test_softmax_basic();
    test_softmax()
