
"""
softmax
"""
import  numpy as np
import pickle
import matplotlib as plt
from cs231n.data_utils import load_CIFAR10
from cs231n.gradient_check import grad_check_sparse
import time

# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0,8.0)     #设置图像的初试大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#加载cifar10数据集并进行预处理
def get_CIFAR10_data(num_training = 49000, num_validation = 1000, num_test = 1000, num_dev = 500):
    #加载原始cifar10数据
    X_train, y_train, X_test, y_test = load_CIFAR10(r"C:\Users\Michael-School\PycharmProjects\datasets\cifar-10-batches-py")
    # print("X_train:",np.shape(X_train)) #(50000, 32, 32, 3)
    # print("y_train:",np.shape(y_train)) #(50000,)
    # print("X_test:",np.shape(X_test))   #X_test: (10000, 32, 32, 3)
    # print("y_test",np.shape(y_test))    #y_test (10000,)

    #对数据二次采样：从数据集中取数据子集用于后面的训练
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    #数据预处理：将一幅图像变成一行存在相应的矩阵里
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    #标准化数据：先求平均图像，再将每个图像都减去平均图像，这样预处理可以加速后期优化过程中权重参数的收敛性
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    #增加偏置的维度：在原矩阵后加一个全是1的列
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

#本地测试
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
# print('Train data shape: ', np.shape(X_train))  #(49000, 3073)
# print('Train labels shape: ', np.shape(y_train))    #(49000,)
# print('Validation data shape: ', np.shape(X_val))   #(1000, 3073)
# print('Validation labels shape: ', np.shape(y_val)) #(1000,)
# print('Test data shape: ', np.shape(X_test))    #(1000, 3073)
# print('Test labels shape: ', np.shape(y_test))  #(1000,)
# print('dev data shape: ', np.shape(X_dev))  #(500, 3073)
# print('dev labels shape: ', np.shape(y_dev))    (500,)


#softmax分类器循环实现
def softmax_loss_naive(W,X,y,reg):

    """
    输入有D维，C类，在minibatch上运行N个例子

    :param W:维度总数，种类，包含权重的numpy(D,C)=(3073,10)
    :param X:包含minibatch的numpy(N,D)=(200,3073)
    :param y:包含训练标签的numpy=200，y[i]=c来表示x[i]有标签c
    :param reg:正规化的强度(需要多次尝试选出最优值)
    :return:返回一个元组作为单一浮动损失，和W同样维度的W的梯度
    """
    loss = 0.0  #初始化损失值为0
    dW = np.zeros_like(W)   #初始化W的梯度为0

    #计算loss和gradient
    num_classes = W.shape[1]    #分类列数10
    num_train = X.shape[0]  #训练数据的行数200

    scores = np.dot(X,W)

    #为每一批计算损失和梯度
    for i in range(num_train):
        current_scores = scores[i,:]   #取每一行的全部
        # 通过从得分向量中减去最大值来修复数值稳定性
        shift_scores = current_scores - np.max(current_scores)
        #计算损失值
        loss_i = -shift_scores[y[i]] + np.log(np.sum(np.exp(shift_scores)))   #计算交叉熵损失
        loss += loss_i

        for j in range(num_classes):
            softmax_score = np.exp(shift_scores[j]) / np.sum(np.exp(shift_scores))     #计算softmax分数

            #计算梯度(求导)
            if j ==  y[i]:
                dW[:,j] += (-1 + softmax_score) * X[i]
            else:
                dW[:,j] += softmax_score * X[i]
    #在批上求平均，添加正则化系数项
    loss /= num_train
    loss += 2*reg*W

    return loss,dW


#softmax分类器向量化实现
def softmax_loss_vectorized(W,X,y,reg):
    loss = 0.0  #初始化损失值为0
    dW = np.zeros_like(W)

    num_train = X.shape[0]

    #计算得分修正数值
    scores = np.dot(X,W)
    shift_scores = scores - np.max(scores,axis=1)[..., np.newaxis]

    #计算softmax得分
    softmax_scores = np.exp(shift_scores)/np.sum(np.exp(shift_scores),axis=1)[...,np.newaxis]

    #计算dScore
    dScore = softmax_scores
    dScore[range(num_train),y] = dScore[range(num_train),y] - 1

    dW = np.dot(X.T,dScore)
    dW /= num_train
    dW += 2*reg*W

    #计算交叉熵损失函数
    correct_class_scores = np.choose(y, shift_scores.T)
    loss = -correct_class_scores + np.log(np.sum(np.exp(shift_scores),axis=1))
    loss = np.sum(loss)

    #平均损失函数值并且正则化
    loss /= num_train
    loss += reg *np.sum(W*W)

    return loss,dW

#生成一个权重矩阵
W = np.random.randn(3073, 10) * 0.0001
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

# As a rough sanity check, our loss should be something close to -log(0.1).
print('loss: %f' % loss)
print('sanity check: %f' % (-np.log(0.1)))


loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 5e1)
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)














