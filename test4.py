#使用SVM处理手写体图片数据集

#导入手写体数字加载器
from sklearn.datasets import load_digits

#获得手写体数字的数码图像
digits = load_digits()

print(digits.data.shape)  #(1797, 64) 1797条数码图像数据，每张图像像素8x8

#数据分割
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)
# print(y_train.shape)  #1347
# print(y_test.shape)  #450

# 从sklearn.svm里导入基于线性假设的支持向量机分类器LinearSVC。
from sklearn.svm import LinearSVC

from sklearn.preprocessing import StandardScaler  #StandardScaler数据标准化预处理
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lsvc = LinearSVC()
lsvc.fit(X_train,y_train)
y_predict = lsvc.predict(X_test)

#评估
print('lsvc accuracy:',lsvc.score(X_test,y_test))  #0.9533333333333334
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names =digits.target_names.astype(str)))


