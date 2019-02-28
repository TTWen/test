#cs231n:K-近邻实现cifar-10图像分类

import numpy as np
import pickle

# NN
class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred

#数据预处理和调用
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def load_CIFAR10(file):
    dataTrain = []
    labelTrain = []
    for i in range(1, 6):
        dic = unpickle(r"C:\Users\TTWen\Desktop\DesktopFile\Study\py\cifar-10-batches-py\data_batch_" + str(i))
        for item in dic["data"]:
            dataTrain.append(item)
        for item in dic["labels"]:
            labelTrain.append(item)

    # get the test data
    dataTest = []
    labelTest = []
    dic = unpickle(r"C:\Users\TTWen\Desktop\DesktopFile\Study\py\cifar-10-batches-py\test_batch")
    for item in dic["data"]:
        dataTest.append(item)
    for item in dic["labels"]:
        labelTest.append(item)
    return (dataTrain,labelTrain,dataTest,labelTest)

datatr, labeltr, datate, labelte = load_CIFAR10(r"C:\Users\TTWen\Desktop\DesktopFile\Study\py\cifar-10-batches-py")

Xtr = np.asarray(datatr)
Xte = np.asarray(datate)
Ytr = np.asarray(labeltr) #（5000*1）
Yte = np.asarray(labelte)
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
print ("Xtr.shape = ",Xtr.shape)
print ("Xte.shape = ",Xte.shape)
print ("Ytr.shape = ",Ytr.shape)
print ("Yte.shape",Yte.shape)
print ("type(Xtr) = ",type(Xtr))
#  dataTr.shape = (50000, 3072) 训练集50000张32x32的图片3原色，32x32x3=3072,获取到训练集了

nn = NearestNeighbor()  # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr)  # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows)  # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print ('accuracy: %f' % ( np.mean(Yte_predict == Yte) ))

