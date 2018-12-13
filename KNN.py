import os
import struct
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot
np.seterr(over='ignore')

# Function to read data
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

# This function will display an image
def show(image):

    fig = pyplot.figure()
    image = image.reshape(28, 28)
    imgplot = pyplot.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    pyplot.show()

#Implementation of knn algorithm
def knn(train_label, test_label, euclidian_dist, k):
    err = 0
    test_size = test_label.shape[0]
    k_nearest_label = np.zeros(k, dtype=int)
    for l in range(0, test_size):
        index_array = np.argsort(euclidian_dist[l, :])  # find the indices of maximum label
        for j in range(0, k):
            k_nearest_label[j] = train_label[index_array[j]] # store same index from train label dataset
        pred_label = np.bincount(k_nearest_label).argmax()  # Find the maximum occurrence of a value in label dataset to count votes
        if pred_label != test_label[l]: # If values are unequal it's an error
            err = err + 1
    return err


# Define the path for Dataset.
train_img, train_lbl=read(dataset = "training", path = "/home/nidhi/Downloads/MNIST")
test_img, test_lbl=read(dataset = "testing", path = "/home/nidhi/Downloads/MNIST")

# we can select how many samples we want to test and train
training_examples = 60000
testing_examples = 10000
total_examples = 60000
test_examples = 10000
class_labels = 10

# Shuffle the dataset if we are testing on partial dataset. Shuffle is for training data ans shuffle1 is for test data
shuffle = np.arange(total_examples)
shuffle1 = np.arange(test_examples)
np.random.shuffle(shuffle)
np.random.shuffle(shuffle1)

# Initialize zeroes in the dataset and then select shuffled data, set datatype as int32 to store correct value of distance
training_img = np.zeros((training_examples, 784), dtype=np.int32)
training_img[:] = train_img[shuffle[:training_examples]]
training_label = train_lbl[shuffle[:training_examples]]

testing_img = np.zeros((testing_examples, 784), dtype=np.int32)
testing_img[:] = test_img[shuffle1[: testing_examples]]
testing_label = test_lbl[shuffle1[: testing_examples]]


train_img=training_img
test_img=testing_img
train_lbl=training_label
test_lbl=testing_label

print("The training dataset has dimensions equal to")
print(train_img.shape)
print(train_lbl.shape)
print("The test set has dimensions equal to")
print(test_img.shape)
print(test_lbl.shape)


#define dataset with values of k
k_val = [1,3,5,10,30,50,70,80,90,100]

error = np.zeros(len(k_val))
accuracy = np.zeros(len(k_val))
dists = np.zeros((test_img.shape[0], train_img.shape[0]))

#calculate common distance matrix
dists = -2 * np.dot(test_img, train_img.T) + np.sum(train_img**2, axis=1) + np.sum(test_img**2, axis=1)[:, np.newaxis]

#Calculate neighbours and error for all values of K
for m in range(0, len(k_val)):
    error[m] = knn(train_lbl, test_lbl, dists, k_val[m])
    error[m] = error[m] / test_img.shape[0]
    accuracy[m] = (1 - error[m])*100

# Print the result
print(k_val)
print(error)
print(accuracy)

#create graph for error
pyplot.plot(k_val, error, label="Error")
pyplot.ylabel('Error')
pyplot.xlabel('K Values')
pyplot.show()

#create graph for accuracy
pyplot.plot(k_val, accuracy, label="Accuracy vs K values for KNN")
pyplot.ylabel('Accuracy')
pyplot.xlabel('K Values')
pyplot.show()
pyplot.savefig('knn_accuracy.png')

