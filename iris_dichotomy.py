#import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime



#导入数据
#Set working directory and load data
os.chdir('C:\\Users\\rohan\\Documents\\Analytics\\Data')
iris = pd.read_csv('iris.csv')
#Create numeric classes for species (0,1,2)
iris.loc[iris['Name']=='virginica','species']=0
iris.loc[iris['Name']=='versicolor','species']=1
iris.loc[iris['Name']=='setosa','species'] = 2
iris = iris[iris['species']!=2]
#Create Input and Output columns
X = iris[['PetalLength', 'PetalWidth']].values.T
Y = iris[['species']].values.T
Y = Y.astype('uint8')
#Make a scatter plot
plt.scatter(X[0, :], X[1, :], c=Y[0,:], s=40, cmap=plt.cm.Spectral);
plt.title("IRIS DATA | Blue - Versicolor, Red - Virginica ")
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()


#初始化参数（权重和偏置）
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)  # we set up a seed so that our output matches ours although the initialization is random.

    W1 = np.random.randn(n_h, n_x) * 0.01  # weight matrix of shape (n_h, n_x)
    b1 = np.zeros(shape=(n_h, 1))  # bias vector of shape (n_h, 1)
    W2 = np.random.randn(n_y, n_h) * 0.01  # weight matrix of shape (n_y, n_h)
    b2 = np.zeros(shape=(n_y, 1))  # bias vector of shape (n_y, 1)

    # store parameters into a dictionary
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


#前向传播（forward propagation）
def forward_propagation(X, parameters):
    # retrieve intialized parameters from dictionary
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Implement Forward Propagation to calculate A2 (probability)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)  # tanh activation function
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))  # sigmoid activation function

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

# 计算代价函数（cost function）
def compute_cost(A2, Y, parameters):
    m = Y.shape[1]  # number of training examples

    # Retrieve W1 and W2 from parameters
    W1 = parameters['W1']
    W2 = parameters['W2']

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m

    return cost


#反向传播（back propagation）
def backward_propagation(parameters, cache, X, Y):
    # Number of training examples
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".


    W1 = parameters['W1']
    W2 = parameters['W2']
    ### END CODE HERE ###

    # Retrieve A1 and A2 from dictionary "cache".
    A1 = cache['A1']
    A2 = cache['A2']

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
         "db1": db1,
         "dW2": dW2,
         "db2": db2}

    return grads


# 更新参数
def update_parameters(parameters, grads, learning_rate=1.2):

    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# 建立神经网络
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)

        ### END CODE HERE ###

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return parameters, n_h



def plot_decision_boundary(model, X, y):
        # Set min and max values and give it some padding
        x_min, x_max = X[0, :].min() - 0.25, X[0, :].max() + 0.25
        y_min, y_max = X[1, :].min() - 0.25, X[1, :].max() + 0.25
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0,:])
    plt.title("Decision Boundary for hidden layer size " + str(6))
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')




# 6.模型评估
def predict(parameters, x_test, y_test):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    z1 = np.dot(w1, x_test) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))

    # 结果的维度
    n_rows = y_test.shape[0]
    n_cols = y_test.shape[1]

    # 预测值结果存储
    output = np.empty(shape=(n_rows, n_cols), dtype=int)

    for i in range(n_rows):
        for j in range(n_cols):
            if a2[i][j] > 0.5:
                output[i][j] = 1
            else:
                output[i][j] = 0

    print('预测结果：', output)
    print('真实结果：', y_test)

    count = 0
    for k in range(0, n_cols):
        if output[0][k] == y_test[0][k] and output[1][k] == y_test[1][k] and output[2][k] == y_test[2][k]:
            count = count + 1
        else:
            print('错误分类样本的序号：', k + 1)

    acc = count / int(y_test.shape[1]) * 100
    print('准确率：%.2f%%' % acc)

    return output


# 7.结果可视化
# 特征有4个维度，类别有1个维度，一共5个维度，故采用了RadViz图
def result_visualization(x_test, y_test, result):
    cols = y_test.shape[1]
    y = []
    pre = []

    # 反转换类别的独热编码
    for i in range(cols):
        if y_test[0][i] == 0 and y_test[1][i] == 0 and y_test[2][i] == 1:
            y.append('setosa')
        elif y_test[0][i] == 0 and y_test[1][i] == 1 and y_test[2][i] == 0:
            y.append('versicolor')
        elif y_test[0][i] == 1 and y_test[1][i] == 0 and y_test[2][i] == 0:
            y.append('virginica')

    for j in range(cols):
        if result[0][j] == 0 and result[1][j] == 0 and result[2][j] == 1:
            pre.append('setosa')
        elif result[0][j] == 0 and result[1][j] == 1 and result[2][j] == 0:
            pre.append('versicolor')
        elif result[0][j] == 1 and result[1][j] == 0 and result[2][j] == 0:
            pre.append('virginica')
        else:
            pre.append('unknown')

    # 将特征和类别矩阵拼接起来
    real = np.column_stack((x_test.T, y))
    prediction = np.column_stack((x_test.T, pre))

    # 转换成DataFrame类型，并添加columns
    df_real = pd.DataFrame(real, index=None, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species'])
    df_prediction = pd.DataFrame(prediction, index=None, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species'])

    # 将特征列转换为float类型，否则radviz会报错
    df_real[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']] = df_real[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].astype(float)
    df_prediction[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']] = df_prediction[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].astype(float)

    # 绘图
    plt.figure('真实分类')
    radviz(df_real, 'Species', color=['blue', 'green', 'red', 'yellow'])
    plt.figure('预测分类')
    radviz(df_prediction, 'Species', color=['blue', 'green', 'red', 'yellow'])
    plt.show()


if __name__ == "__main__":
    # 读取数据
    data_set = pd.read_csv('D:\\rjaz\Pycharm\\code\\bp\\bpnn_V2数据集\\iris_training.csv', header=None)

    # 第1种取数据方法：
    X = data_set.iloc[:, 0:4].values.T          # 前四列是特征，T表示转置
    Y = data_set.iloc[:, 4:].values.T           # 后三列是标签

    # 第2种取数据方法：
    # X = data_set.ix[:, 0:3].values.T
    # Y = data_set.ix[:, 4:6].values.T

    # 第3种取数据方法：
    # X = data_set.loc[:, 0:3].values.T
    # Y = data_set.loc[:, 4:6].values.T

    # 第4种取数据方法：
    # X = data_set[data_set.columns[0:4]].values.T
    # Y = data_set[data_set.columns[4:7]].values.T
    Y = Y.astype('uint8')

    # 开始训练
    start_time = datetime.datetime.now()
    # 输入4个节点，隐层10个节点，输出3个节点，迭代10000次
    parameters = nn_model(X, Y, n_h=10, n_input=4, n_output=3, num_iterations=10000, print_cost=True)
    end_time = datetime.datetime.now()
    print("用时：" + str((end_time - start_time).seconds) + 's' + str(round((end_time - start_time).microseconds / 1000)) + 'ms')

    # 对模型进行测试
    data_test = pd.read_csv('D:\\rjaz\\Pycharm\\code\\bp\\bpnn_V2数据集\\iris_test.csv', header=None)
    x_test = data_test.iloc[:, 0:4].values.T
    y_test = data_test.iloc[:, 4:].values.T
    y_test = y_test.astype('uint8')

    result = predict(parameters, x_test, y_test)

    # 分类结果可视化
    result_visualization(x_test, y_test, result)

