import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
# Added by me
import numpy.matlib as matlib
import time


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data
    #### ^ 10000. Training set contains 60000 examples. So divide it into two. 50000 for training set, 10000 for validation.
    
    
    #Your code here
    train0 = mat.get('train0')
    train1 = mat.get('train1')
    train2 = mat.get('train2')
    train3 = mat.get('train3')
    train4 = mat.get('train4')
    train5 = mat.get('train5')
    train6 = mat.get('train6')
    train7 = mat.get('train7')
    train8 = mat.get('train8')
    train9 = mat.get('train9')
    train_data = np.concatenate((train0, train1, train2, train3, train4, train5, train6, train7, train8, train9))
    train_data = train_data * 1.0   # convert data to double
    train_data = train_data / 255.0 # normalize data

    #### NEED TO DO FEATURE SELECTION ON TRAIN_DATA

    # THIS IS A 784 LENGTH ARRAY WITH VALUES TRUE OR FALSE: TRUE IF THE COLUMN CAN BE REMOVED (ALL PIXEL VALUES ARE THE SAME)
    # I checked how many "True" columns are in train_data and it is 67, the same # people have mentioned on Piazza, so I think
    # I am doing this part correctly.
    check_columns = np.all(train_data == train_data[0,:], axis=0)
    # Now I need to remove the columns marked True from the train_data.
    count = 0
    elim = []  # array of indexes of columns to be removed
    for i in check_columns:
        if i == True:
            elim.append(count)
        count += 1

    # Final train_data with feature selection
    train_data = np.delete(train_data, elim, 1)


    label0 = np.empty(len(train0))
    label0.fill(0)
    label1 = np.empty(len(train1))
    label1.fill(1)
    label2 = np.empty(len(train2))
    label2.fill(2)
    label3 = np.empty(len(train3))
    label3.fill(3)
    label4 = np.empty(len(train4))
    label4.fill(4)
    label5 = np.empty(len(train5))
    label5.fill(5)
    label6 = np.empty(len(train6))
    label6.fill(6)
    label7 = np.empty(len(train7))
    label7.fill(7)
    label8 = np.empty(len(train8))
    label8.fill(8)
    label9 = np.empty(len(train9))
    label9.fill(9)

    # label0 = matlib.repmat([1,0,0,0,0,0,0,0,0,0], len(train0), 1)
    # label1 = matlib.repmat([0,1,0,0,0,0,0,0,0,0], len(train1), 1)
    # label2 = matlib.repmat([0,0,1,0,0,0,0,0,0,0], len(train2), 1)
    # label3 = matlib.repmat([0,0,0,1,0,0,0,0,0,0], len(train3), 1)
    # label4 = matlib.repmat([0,0,0,0,1,0,0,0,0,0], len(train4), 1)
    # label5 = matlib.repmat([0,0,0,0,0,1,0,0,0,0], len(train5), 1)
    # label6 = matlib.repmat([0,0,0,0,0,0,1,0,0,0], len(train6), 1)
    # label7 = matlib.repmat([0,0,0,0,0,0,0,1,0,0], len(train7), 1)
    # label8 = matlib.repmat([0,0,0,0,0,0,0,0,1,0], len(train8), 1)
    # label9 = matlib.repmat([0,0,0,0,0,0,0,0,0,1], len(train9), 1)
    train_label = np.concatenate((label0, label1, label2, label3, label4, label5, label6, label7, label8, label9))

    perm = np.random.permutation(range(len(train_data)))
    new_train_data = train_data[perm[0:50000],:]
    validation_data = train_data[perm[50000:],:]
    train_data = new_train_data

    new_train_label = train_label[perm[0:50000]]
    validation_label = train_label[perm[50000:]]
    train_label = new_train_label

    test0 = mat.get('test0')
    test1 = mat.get('test1')
    test2 = mat.get('test2')
    test3 = mat.get('test3')
    test4 = mat.get('test4')
    test5 = mat.get('test5')
    test6 = mat.get('test6')
    test7 = mat.get('test7')
    test8 = mat.get('test8')
    test9 = mat.get('test9')
    test_data = np.concatenate((test0, test1, test2, test3, test4, test5, test6, test7, test8, test9))
    test_data = test_data * 1.0     # convert data to double
    test_data = test_data / 255.0   # normalize data

    #### NEED TO DO FEATURE SELECTION ON TEST_DATA

    # Final test_data with feature selection (reuse elim from feature selection above).
    test_data = np.delete(test_data, elim, 1)

    test_label0 = np.empty(len(test0))
    test_label0.fill(0)
    test_label1 = np.empty(len(test1))
    test_label1.fill(1)
    test_label2 = np.empty(len(test2))
    test_label2.fill(2)
    test_label3 = np.empty(len(test3))
    test_label3.fill(3)
    test_label4 = np.empty(len(test4))
    test_label4.fill(4)
    test_label5 = np.empty(len(test5))
    test_label5.fill(5)
    test_label6 = np.empty(len(test6))
    test_label6.fill(6)
    test_label7 = np.empty(len(test7))
    test_label7.fill(7)
    test_label8 = np.empty(len(test8))
    test_label8.fill(8)
    test_label9 = np.empty(len(test9))
    test_label9.fill(9)

    # test_label0 = matlib.repmat([1,0,0,0,0,0,0,0,0,0], len(test0), 1)
    # test_label1 = matlib.repmat([0,1,0,0,0,0,0,0,0,0], len(test1), 1)
    # test_label2 = matlib.repmat([0,0,1,0,0,0,0,0,0,0], len(test2), 1)
    # test_label3 = matlib.repmat([0,0,0,1,0,0,0,0,0,0], len(test3), 1)
    # test_label4 = matlib.repmat([0,0,0,0,1,0,0,0,0,0], len(test4), 1)
    # test_label5 = matlib.repmat([0,0,0,0,0,1,0,0,0,0], len(test5), 1)
    # test_label6 = matlib.repmat([0,0,0,0,0,0,1,0,0,0], len(test6), 1)
    # test_label7 = matlib.repmat([0,0,0,0,0,0,0,1,0,0], len(test7), 1)
    # test_label8 = matlib.repmat([0,0,0,0,0,0,0,0,1,0], len(test8), 1)
    # test_label9 = matlib.repmat([0,0,0,0,0,0,0,0,0,1], len(test9), 1)
    test_label = np.concatenate((test_label0, test_label1, test_label2, test_label3, test_label4, test_label5, test_label6, 
        test_label7, test_label8, test_label9))
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    print 'nnObjFunction'
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    inputBiases = np.ones((len(training_data),1))
    training_data = np.append(training_data, inputBiases, axis=1)

    training_label = training_label.astype(np.int64)
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    
    #Your code here
    label_encoding = [[1,0,0,0,0,0,0,0,0,0], #0
                      [0,1,0,0,0,0,0,0,0,0], #1
                      [0,0,1,0,0,0,0,0,0,0], #2
                      [0,0,0,1,0,0,0,0,0,0], #3
                      [0,0,0,0,1,0,0,0,0,0], #4
                      [0,0,0,0,0,1,0,0,0,0], #5
                      [0,0,0,0,0,0,1,0,0,0], #6
                      [0,0,0,0,0,0,0,1,0,0], #7
                      [0,0,0,0,0,0,0,0,1,0], #8
                      [0,0,0,0,0,0,0,0,0,1]] #9


    n = len(training_data)
    J = 0 #cumulative error

    w1_gradient = np.zeros((n_hidden, n_input + 1))
    w2_gradient = np.zeros((n_class, n_hidden + 1))

    # for each training example
    for i in range(n):
        if i%1000 == 0: print i
        # FEED FORWARD
        # get outputs of hidden nodes for ith example
        output_hidden_i = np.zeros(n_hidden) # z
        for j in range(n_hidden):
            output_j = 0
            for x in range(n_input + 1):
                output_j += w1[j][x] * training_data[i,x]       ######

            output_hidden_i[j] = sigmoid(output_j)

        # Append bias node to hidden nodes
        output_hidden_i = np.append(output_hidden_i, [1], axis=1)


        # get outputs of output nodes for ith example
        output_out_i = np.zeros(n_class) # o
        for l in range(n_class):
            output_l = 0
            for j in range(n_hidden + 1):
                output_l += w2[l][j] * output_hidden_i[j]

            output_out_i[l] = sigmoid(output_l)


        # ERROR FUNCTION CALCULATION
        # Calculate Ji
        Ji = 0
        yi = label_encoding[training_label[i]]
        for l in range(n_class):
            Ji += yi[l] * np.log(output_out_i[l]) + (1 - yi[l]) * np.log(1 - output_out_i[l])
        Ji = -1 * Ji
        # Add Ji to cumulative J
        J += Ji


        # function 8/9 from Asst1 pdf
        sigma_i = np.zeros(n_class)
        for j in range(n_hidden + 1):
            for l in range(n_class):
                sigma_i[l] = output_out_i[l] - yi[l]
                w2_gradient[l][j] += sigma_i[l] * output_hidden_i[j]     # derivative from function 8/9 


        # function 10/11/12
        for d in range(n_input + 1):
            for j in range(n_hidden):
                sum_x = 0
                for l in range(n_class):
                    sum_x += sigma_i[l] * w2[l][j]

                w1_gradient[j][d] += (1 - output_hidden_i[j]) * output_hidden_i[j] * sum_x * training_data[i][d]


    J = J / len(training_data)  # Is this obj_val?
    

    # REGULARIZATION
    reg_J = 0
    w1_sum = 0
    w2_sum = 0
    for m in range(n_hidden):
        for d in range(n_input + 1):
            w1_sum += w1[m][d] * w1[m][d]
    for l in range(n_class):
        for m in range(n_hidden + 1):
            w2_sum += w2[l][m] * w2[l][m]

    reg_J = J + ((lambdaval / (2 * n)) * (w1_sum + w2_sum))

    obj_val = reg_J

    w1_gradient = (w1_gradient + (lambdaval * w1)) / n
    w2_gradient = (w2_gradient + (lambdaval * w2)) / n
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((w1_gradient.flatten(), w2_gradient.flatten()),0)
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % labels: a column vector of predicted labels""" 
    print 'predicting...'
    n = len(data)    
    labels = np.zeros(n)

    for i in range(n):
        output_i = np.zeros(n_class)

        testAr = np.array([1])

        test_train = np.concatenate((training_data[i], testAr))

        input_vectors_1 = np.dot(w1, test_train)

        for m in range(input_vectors_1.shape[0]):
            input_vectors_1[m] = sigmoid(input_vectors_1[m])

        test_train2 = np.concatenate((input_vectors_1, testAr))
        input_vectors_2 = np.dot(w2, test_train2)

        for l in range(n_class):
            output_i[l] = sigmoid(input_vectors_2[l])

        top_l = 0
        test = 0
        for l in range(n_class):
            if output_i[l] > top_l:
                top_l = output_i[l]
                test = l
        labels[i] = float(test)
    
    return labels
    



"""**************Neural Network Script Starts here********************************"""
start_time = time.time()

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess();

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 4;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print '\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%'

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print '\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%'


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print '\n Test set Accuracy:' + + str(100*np.mean((predicted_label == test_label).astype(float))) + '%'

end_time = (time.time() - start) / 60
print end_time # minutes