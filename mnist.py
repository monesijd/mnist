import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

class Mnist_MLP:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.w = []
        self.b = []
        self.activation_func = []

    def add_layer(self, input_shape, output_shape, activation_func):
        self.w.append(np.random.randn(input_shape, output_shape))
        self.b.append(np.random.randn(output_shape))
        self.activation_func.append(activation_func)

    def train(self, x_train, y_train, epochs, batch_size, learning_rate):
        for epoch in range(epochs):
            for i in range(len(x_train) // batch_size):
                x = x_train[i*batch_size : (i+1)*batch_size]
                y = y_train[i*batch_size : (i+1)*batch_size]
                activation = self.frontpropagation(x, batch_size)
                self.backpropagation(activation, y, learning_rate)

    def predict(self):
        accuracy = 0
        for i in range(len(x_test)):
            prediction = self.predict_evaluate(x_test[i].reshape(1, 784))
            if y_test[i][prediction] == 1:
                accuracy += 1
        print("Accuracy:", accuracy/len(x_test))

    def predict_evaluate(self, x):
        x = x.reshape(1, 784)
        activation = data.frontpropagation(x, batch_size=1)
        predicted_label = np.argmax(activation[-1], axis=1)
        return predicted_label

    def frontpropagation(self, x, batch_size):
        activation = [x]
        for i in range(len(self.w)):
            z = np.dot(x, self.w[i]) + self.b[i]
            if self.activation_func[i] == 'relu':
                activation.append(self.relu(z))
            elif self.activation_func [i] == 'sigmoid':
                activation.append(self.sigmoid(z))
            elif self.activation_func[i] == 'softmax':
                activation.append(self.softmax(z))
            else:
                print('error')
            x = z
        # print(activation[0].shape, activation[1].shape, activation[2].shape, activation[3].shape)
        return activation

    def backpropagation(self, A, y, learning_rate):
        m = y.shape[0]
        L = len(self.w)
        dW = [0] * L
        db = [0] * L
        dA = [0] * (L+1)

        # print((A[-1] - y).shape)
        dA[L] = A[-1] - y

        # 從最後一層開始反向計算
        for l in range(L-1, -1, -1):
            dZ = dA[l+1]  # 誤差
            dW[l] = np.dot(A[l].T, dZ) / m  # 計算權重梯度
            db[l] = np.sum(dZ, axis=0, keepdims=True) / m  # 計算偏差梯度

            # 若不是第一層，計算上一層的誤差
            if l > 0:
                if self.activation_func[l-1] == 'relu':
                    dA[l] = np.dot(dZ, self.w[l].T) * (A[l] >= 0)  # ReLU 梯度
                elif self.activation_func[l-1] == 'sigmoid':
                    dA[l] = np.dot(dZ, self.w[l].T) * A[l] * (1 - A[l])  # Sigmoid 梯度

        # 更新權重和偏置
        for l in range(L-1, -1, -1):
            self.w[l] -= learning_rate * dW[l]
            self.b[l] -= learning_rate * np.squeeze(db[l])


    def cost(self, x, y, w, b):
        y_pred = np.dot(x, w) + b
        cost = np.mean((y - y_pred) ** 2)
        return cost

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        x -= np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True) 


(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], 784) / 255
x_test = x_test.reshape(x_test.shape[0], 784) / 255
one_hot_encoder = OneHotEncoder()
y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test = one_hot_encoder.fit_transform(y_test.reshape(-1, 1)).toarray()

data = Mnist_MLP(x_train, y_train)
data.add_layer(784, 128, 'relu')
data.add_layer(128, 10, 'sigmoid')
data.train(x_train, y_train, epochs=10, batch_size=64, learning_rate=0.01)

data.predict()
