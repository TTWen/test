#朴素贝叶斯：实现新闻分类
from sklearn.datasets import fetch_20newsgroups #导入新闻数据抓取器，抓取器需要从网上下载资源
news = fetch_20newsgroups(subset="all")
print(len(news.data)) #有18846条新闻，这些数据没有设定特征和数字化度量
# print(news.data[0])

#对数据进行分割
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33)

#将文本转化为特征向量，利用朴树贝叶斯从训练数据中估计参数利用这些参数对测试新闻样本进行预测
from sklearn.feature_extraction.text import CountVectorizer #导入文本特征向量转化模块
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)
from sklearn.naive_bayes import MultinomialNB #导入贝叶斯模型
mnb = MultinomialNB() #默认初始化
mnb.fit(X_train,y_train)
y_predict = mnb.predict(X_test)
#评估部分
from sklearn.metrics import classification_report
print("the accuracy is ",mnb.score(X_test,y_test)) #0.8397707979626485
print(classification_report(y_test,y_predict,target_names=news.target_names))














