#cs231n:批量归一化
import  numpy as np
#定义一个双层网络随机初始化
def init_two_layer_model(input_size,hidden_size,output_size):
    model = {}
    model['W1'] = 0.0001 * np.random.randn(input_size,hidden_size)
    model['b1'] = np.zeros(hidden_size)
    model['W2'] = 0.0001 * np.random.randn(hidden_size,output_size)
    model['b2'] = np.zeros(output_size)
    return model

model = init_two_layer_model(32*32*3,50,10)
loss,grade = two_layer_net(X_train,model,y_train,0.0)  #这里的0是零正则化项，loss大约是2.3，启动正则化赋值为1e3时，loss大约是3.06
print("loss:",loss)


















