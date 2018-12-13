import os
import struct
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot

# Read the data from the datset
def read(dataset = "training", path = "/home/nidhi/Downloads/MNIST"):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        _, _ = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        _, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows* cols)

    return (img, lbl)

# function to display image
def show(image):
    fig = pyplot.figure()
    image = image.reshape(28, 28)
    imgplot = pyplot.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    pyplot.show()

#sigmoid function to predict accuracy
def sigmoid1(z):
    s = 1/ (1.0 + np.exp(-z))
    return s

# sigmoid modfied for gradient ascent for Multiple classes
def sigmoid(a,a_sum,case):
    if case ==9:
        return 1/(1+a_sum)
    else:
        return (np.exp(a) / (1 + a_sum))

# function to find gradient, cost, and probability
def gradient(w, X, Y,case,w_sum):
    m = X.shape[1]
    z = np.dot(w.T,X)
    A = sigmoid(z,w_sum,case)
    Y1 = (Y == case)
    Y1.astype(np.int)

    cost = 1.0 / m * np.sum((Y1 * np.log(A)) - (1.0 - Y1) * np.log(1.0 - A))
    grad = 1.0 / m * np.dot(X, (Y1-A).T)

    return grad, cost, A

# For predicting test label
def predict(w, X):
    m = X.shape[1]
    A = sigmoid1(np.dot(w.T, X))
    return A

def model(train_img, train_lbl, test_img, test_lbl, num_iterations=10, learning_rate=0.001):

    dim=784
    w1 = np.zeros((dim, 1))
    w2 = np.zeros((dim, 1))
    w3 = np.zeros((dim, 1))
    w4 = np.zeros((dim, 1))
    w5 = np.zeros((dim, 1))
    w6 = np.zeros((dim, 1))
    w7 = np.zeros((dim, 1))
    w8 = np.zeros((dim, 1))
    w9 = np.zeros((dim, 1))
    w0 = np.zeros((dim, 1))

    for j in range(10):
        costs = []
        print("Computing for label = %i " % (j))
        for i in range(num_iterations):
            z_sum = np.exp((np.dot(w0.T, train_img))) + np.exp((np.dot(w1.T, train_img))) + np.exp((np.dot(w2.T, train_img))) \
                    + np.exp((np.dot(w3.T, train_img))) + np.exp((np.dot(w4.T, train_img))) + np.exp((np.dot(w5.T, train_img))) \
                    + np.exp((np.dot(w6.T, train_img))) + np.exp((np.dot(w7.T, train_img))) + np.exp((np.dot(w8.T, train_img)))

            if j == 0:
                grad, cost, Y_train0 = gradient(w0, train_img, train_lbl, j, z_sum)
                w0 = w0 + learning_rate * grad
            elif j == 1:
                grad, cost,Y_train1 = gradient(w1, train_img, train_lbl, j, z_sum)
                w1 = w1 + learning_rate * grad
            elif j == 2:
                grad, cost,Y_train2 = gradient(w2, train_img, train_lbl, j, z_sum)
                w2 = w2 + learning_rate * grad
            elif j == 3:
                grad, cost,Y_train3 = gradient(w3, train_img, train_lbl, j, z_sum)
                w3 = w3 + learning_rate * grad
            elif j == 4:
                grad, cost,Y_train4 = gradient(w4, train_img, train_lbl, j, z_sum)
                w4 = w4 + learning_rate * grad
            elif j == 5:
                grad, cost,Y_train5 = gradient(w5, train_img, train_lbl, j, z_sum)
                w5 = w5 + learning_rate * grad
            elif j == 6:
                grad, cost,Y_train6 = gradient(w6, train_img, train_lbl, j, z_sum)
                w6 = w6 + learning_rate * grad
            elif j == 7:
                grad, cost,Y_train7 = gradient(w7, train_img, train_lbl, j, z_sum)
                w7 = w7 + learning_rate * grad
            elif j == 8:
                grad, cost,Y_train8 = gradient(w8, train_img, train_lbl, j, z_sum)
                w8 = w8 + learning_rate * grad
            elif j == 9:
                grad, cost,Y_train9 = gradient(w9, train_img, train_lbl, j, z_sum)
                w9 = w9 + learning_rate * grad


            costs.append(cost)
            # print("Cost (iteration %i) = %f" % (i, cost))

    # Using maximize w method
    w=np.concatenate((w0,w1,w2,w3,w4,w5,w6,w7,w8,w9),axis=1)
    # argmax of probability
    Y_train = np.concatenate((Y_train0, Y_train1, Y_train2, Y_train3, Y_train4, Y_train5, Y_train6, Y_train7, Y_train8, Y_train9),axis=0)

    #print('After predicting')
    # Y_train = predict(w,train_img)
    Y_test = predict(w, test_img)


    # print('After argmax')
    Y_prediction_test = np.argmax(Y_test, axis=0)
    Y_prediction_train = np.argmax(Y_train, axis=0)


    #print('After reshape')
    Y_prediction_train = Y_prediction_train.reshape((1, 60000))
    Y_prediction_test = Y_prediction_test.reshape((1, 10000))
    # print(Y_prediction_train.shape)
    # print(Y_prediction_test.shape)


    error = ((Y_prediction_train - train_lbl))
    result = np.where(error != 0, 1, 0)
    error_count = np.count_nonzero(result)
    mean_error = (error_count / 60000) * 100
    train_accuracy = 100 - mean_error

    error = ((Y_prediction_test - test_lbl))
    result = np.where(error != 0, 1, 0)
    error_count = np.count_nonzero(result)
    mean_error = (error_count / 10000) * 100
    test_accuracy = 100 - mean_error

    print('Number of iterations:', num_iterations)
    print("Accuarcy Test: ", test_accuracy)
    #print("Accuracy Train: ", train_accuracy)

# Read data
train_img, train_lbl=read(dataset = "training", path = "/home/nidhi/Downloads/MNIST")
test_img, test_lbl=read(dataset = "testing", path = "/home/nidhi/Downloads/MNIST")

# Normalise the data
train_img_normalised = (train_img-train_img.mean())/255.0
test_img_normalised = (test_img-test_img.mean())/255.0

#  we need features along the rows, and training cases along the columns so take transpose
train_img_tr = train_img_normalised.transpose()
train_lbl_tr = train_lbl.reshape(1,train_lbl.shape[0])
test_img_tr = test_img_normalised.transpose()
test_lbl_tr = test_lbl.reshape(1,test_lbl.shape[0])

print('Training data image shape:')
print(train_img_tr.T.shape)
print('Training data label shape:')
print(train_lbl_tr.T.shape)
print('Test data image shape:')
print(test_img_tr.T.shape)
print('Test data label shape:')
print(test_lbl_tr.T.shape)

# train the data
model (train_img_tr,train_lbl_tr,test_img_tr,test_lbl_tr,num_iterations = 80,learning_rate = 0.003)

