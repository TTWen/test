#K近邻分类：对生物物种“鸢尾”进行分类

# iris数据集中有3个分类，每个分类有50个样本，总150条数据，每条数据包含4个属性

from sklearn.datasets import load_iris #导入iris数据加载器
iris = load_iris()
# print("iris shape :",iris.data.shape)  #iris shape : (150, 4)
# print(iris.DESCR)  #数据说明
from sklearn.cross_validation import train_test_split #数据分割
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)

# 数据标准化模块
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#使用 K近邻
from sklearn.neighbors import  KNeighborsClassifier
knc = KNeighborsClassifier()
knc.fit(X_train,y_train)
y_predict = knc.predict(X_test)

print("accuracy:",knc.score(X_test,y_test))  #0.8947368421052632
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names=iris.target_names))


