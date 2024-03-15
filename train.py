import sys
import numpy as np
import math
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def dSigmoid(x):
    return np.multiply(sigmoid(x),np.subtract(1, sigmoid(x)))

def cross_entropy(y, op):
    return -np.sum(np.add(y,np.log(op)))

def gFunction(x):
    if a == "sigmoid":
        return dSigmoid(x)

def activationFunction(x):
    if a == "sigmoid":
        return sigmoid(x)

def computeLoss(y, op):
    if l == "cross_entropy":
        return cross_entropy(y, op)

def outputFunction(x):
    exp_x = np.exp(x)
    return np.divide(exp_x,(1 + exp_x))

def computeAccuracy(y, op):
    return np.mean(np.all(y == op, axis=1)) * 100

# There are total L layers (does not include input layer) and L-1 hidden layers.
# Each neuron has pre-activation "a" and activation "h".
# If the size of layer is n then we have batch x n size vector of "a" and "h" for that layer
# size(w1) = size(L0) x size(L1)
# size(b1) = size(l1)

class Model:
    def __init__(self):
        input_row_size, input_col_size = data_input.shape
        output_row_size, output_col_size = data_output.shape
        self.nodes = []
        for i in range(nhl+2):
            if i == 0:
                self.nodes.append(input_col_size)
            elif i == nhl + 1:
                self.nodes.append(output_col_size)
            else:
                self.nodes.append(sz)

        if w_i == "random":
            init_range = 0.1
        else:
            init_range = math.sqrt(6/(input_col_size + output_col_size))

        self.weight = []
        for i in range(nhl+1):
            self.weight.append(np.random.uniform(-init_range, init_range, size = (self.nodes[i+1], self.nodes[i])))

        self.bias = []
        for i in range(nhl+1):
            self.bias.append(np.random.uniform(-init_range, init_range, size = (self.nodes[i+1])))


    def feedForward(self, x, y, wgt, bi):
        A =[]
        h =[]

        # first layer
        h.append(x)
        for j in range(1, nhl + 1):
            A.append(np.add(np.matmul(h[len(h) - 1], wgt[j-1]), bi[j-1]))
            h.append(activationFunction(A[len(A)-1]))

        A.append(np.add(np.matmul(h[len(h) - 1], wgt[len(wgt)-1]), bi[len(bi)-1]))
        op = outputFunction(A[len(A)-1])
        print("loss = " + str(computeLoss(y, op)))
        print("accuracy = " + str(computeAccuracy(y, op)))
        return A,h,op


    def backProp(self, A, h, y, op):
        del_a = []
        del_w = []
        del_b = []
        del_h = []

        
        del_a.append(np.multiply(np.subtract(y, op), -1))
        for i in range(nhl + 1, 0, -1):
            del_w.append(np.matmul(h[i-1].transpose(), del_a[len(del_a)-1]))
            print(del_a[len(del_a) - 1].shape)
            del_b.append(np.sum(del_a[len(del_a)-1], axis=0)) 
            if i - 2 >= 0:
                print(del_a[len(del_a) - 1].shape, self.weight[i-1].transpose().shape)
                del_h.append(np.matmul(del_a[len(del_a) - 1], self.weight[i-1].transpose()))
                print(del_h[len(del_h)-1].shape)
                print(del_h[len(del_h) - 1].shape, gFunction(A[i-2]).shape)
                del_a.append(np.multiply(del_h[len(del_h) - 1], gFunction(A[i-2])))
                print("===================")
        del_w.reverse() 
        del_b.reverse()
        return del_w, del_b

    def sgd(self):
        data_input_row, data_input_col = data_input.shape
        for epoch in range(e):
            for start in range(0, data_input_row, b):
                end = start + b
                if end > data_input_row:
                    end = data_input_row

                x = data_input[start:end, : ]
                y = data_output[start:end, : ]

                A, h, op = self.feedForward(x, y, self.weight, self.bias)
                del_w, del_b = self.backProp(A, h, y, op)


                for i in range(nhl+1):
                    # print(del_w[i])
                    # print("--------------")
                    print("yeeee")
                    print(del_b[i].shape)
                    self.weight[i] = np.subtract(self.weight[i], np.multiply(del_w[i], lr))
                    self.bias[i] = np.subtract(self.bias[i], np.multiply(del_b[i], lr))

    def momentum(self):
        data_input_row, data_input_col = data_input.shape
        prev_uw = []
        prev_ub = []

        for i in range(nhl + 1):
            prev_uw.append(np.zeros((self.nodes[i], self.nodes[i+1])))
            prev_ub.append(np.zeros((self.nodes[i+1])))


        for epoch in range(e):
            total_del_w = []
            total_del_b = []
            for i in range(nhl + 1):
                total_del_w.append(np.zeros((self.nodes[i], self.nodes[i+1])))
                total_del_b.append(np.zeros((self.nodes[i+1])))
            for start in range(0, data_input_row, b):
                end = start + b
                if end > data_input_row:
                    end = data_input_row

                x = data_input[start:end, : ]
                y = data_output[start:end, : ]

                A, h, op = self.feedForward(x, y, self.weight, self.bias)
                del_w, del_b = self.backProp(A, h, y, op)
                for i in range(nhl + 1):
                    total_del_w[i] = np.add(total_del_w[i], del_w[i])
                    total_del_b[i] = np.add(total_del_b[i], del_b[i])

            uw = []
            ub = []
            for i in range(nhl+1):
                uw.append(np.add(np.multiply(prev_uw[i], m),np.multiply(total_del_w[i], lr)))
                ub.append(np.add(np.multiply(prev_ub[i], m),np.multiply(total_del_b[i], lr)))
                self.weight[i] = np.subtract(self.weight[i], uw[i])
                self.bias[i] = np.subtract(self.bias[i], ub[i])

            prev_uw = uw
            prev_ub = ub


    def nag(self):
        data_input_row, data_input_col = data_input.shape
        prev_vw = []
        prev_vb = []
        for i in range(nhl + 1):
            prev_vw.append(np.zeros((self.nodes[i], self.nodes[i+1])))
            prev_vb.append(np.zeros((self.nodes[i+1])))

        for epoch in range(e):
            v_w = []
            v_b = []
            total_del_w = []
            total_del_b = []
            nag_w = []
            nag_b = []
            for i in range(nhl + 1):
                total_del_w.append(np.zeros((self.nodes[i], self.nodes[i+1])))
                total_del_b.append(np.zeros((self.nodes[i+1])))
                v_w.append(np.multiply(prev_vw[i], m))
                v_b.append(np.multiply(prev_vb[i], m))
                nag_w.append(np.subtract(self.weight[i], v_w[i]))
                nag_b.append(np.subtract(self.bias[i], v_b[i]))

            for start in range(0, data_input_row, b):
                end = start + b
                if end > data_input_row:
                    end = data_input_row

                x = data_input[start:end, : ]
                y = data_output[start:end, : ]

                A, h, op = self.feedForward(x, y, nag_w, nag_b)

                del_w, del_b = self.backProp(A, h, y, op)
                for i in range(nhl + 1):
                    total_del_w[i] = np.add(total_del_w[i], del_w[i])
                    total_del_b[i] = np.add(total_del_b[i], del_b[i])

            vw = []
            vb = []
            for i in range(nhl + 1):
                vw.append(np.add(np.multiply(prev_vw[i], m), np.multiply(total_del_w[i], lr)))
                vb.append(np.add(np.multiply(prev_vb[i], m), np.multiply(total_del_b[i], lr)))
                self.weight[i] = np.subtract(self.weight[i], vw[i])
                self.bias[i] = np.subtract(self.bias[i], vb[i])

            prev_vw = vw 
            prev_vb = vb

    def rmsprop(self):
        data_input_row, data_input_col = data_input.shape
        v_w = []
        v_b = []
        for i in range(nhl + 1):
            v_w.append(np.zeros((self.nodes[i], self.nodes[i+1])))
            v_b.append(np.zeros((self.nodes[i+1])))

        for epoch in range(e):
            total_del_w = []
            total_del_b = []
    
            for i in range(nhl + 1):
                total_del_w.append(np.zeros((self.nodes[i], self.nodes[i+1])))
                total_del_b.append(np.zeros((self.nodes[i+1])))

            for start in range(0, data_input_row, b):
                end = start + b
                if end > data_input_row:
                    end = data_input_row

                x = data_input[start:end, : ]
                y = data_output[start:end, : ]

                A, h, op = self.feedForward(x, y, self.weight, self.bias)

                del_w, del_b = self.backProp(A, h, y, op)

                for i in range(nhl + 1):
                    total_del_w[i] = np.add(total_del_w[i], del_w[i])
                    total_del_b[i] = np.add(total_del_b[i], del_b[i])


            for i in range(nhl + 1):
                v_w[i] = np.add(np.multiply(v_w[i], beta), np.multiply(np.power(total_del_w[i], 2), 1 - beta))
                v_b[i] = np.add(np.multiply(v_b[i], beta), np.multiply(np.power(total_del_b[i], 2), 1 - beta))

                self.weight[i] = np.subtract(self.weight[i], np.multiply(np.divide(total_del_w[i], np.add(np.sqrt(v_w[i]), eps)), lr))
                self.bias[i] = np.subtract(self.bias[i], np.multiply(np.divide(total_del_b[i], np.add(np.sqrt(v_b[i]), eps)), lr))


    def adam(self):
        data_input_row, data_input_col = data_input.shape
        m_w = []
        m_b = []
        v_w = []
        v_b = []
        for i in range(nhl + 1):
            m_w.append(np.zeros((self.nodes[i], self.nodes[i+1])))
            m_b.append(np.zeros((self.nodes[i+1])))
            v_w.append(np.zeros((self.nodes[i], self.nodes[i+1])))
            v_b.append(np.zeros((self.nodes[i+1])))

        for epoch in range(e):
            total_del_w = []
            total_del_b = []
    
            for i in range(nhl + 1):
                total_del_w.append(np.zeros((self.nodes[i], self.nodes[i+1])))
                total_del_b.append(np.zeros((self.nodes[i+1])))

            for start in range(0, data_input_row, b):
                end = start + b
                if end > data_input_row:
                    end = data_input_row

                x = data_input[start:end, : ]
                y = data_output[start:end, : ]

                A, h, op = self.feedForward(x, y, self.weight, self.bias)

                del_w, del_b = self.backProp(A, h, y, op)

                for i in range(nhl + 1):
                    total_del_w[i] = np.add(total_del_w[i], del_w[i])
                    total_del_b[i] = np.add(total_del_b[i], del_b[i])


            m_w_hat = []
            m_b_hat = []
            v_w_hat = []
            v_b_hat = []
            for i in range(nhl + 1):
                m_w[i] = np.add(np.multiply(m_w[i], beta1), np.multiply(total_del_w[i], 1 - beta1))
                m_b[i] = np.add(np.multiply(m_b[i], beta1), np.multiply(total_del_b[i], 1 - beta1))
                v_w[i] = np.add(np.multiply(v_w[i], beta2), np.multiply(np.power(total_del_w[i], 2), 1 - beta2))
                v_b[i] = np.add(np.multiply(v_b[i], beta2), np.multiply(np.power(total_del_b[i], 2), 1 - beta2))

                m_w_hat.append(np.divide(m_w[i], 1 - np.power(beta1, i+1)))
                m_b_hat.append(np.divide(m_b[i], 1 - np.power(beta1, i+1)))
                v_w_hat.append(np.divide(v_w[i], 1 - np.power(beta2, i+1)))
                v_b_hat.append(np.divide(v_b[i], 1 - np.power(beta2, i+1)))

                self.weight[i] = np.subtract(self.weight[i], np.multiply(np.divide(m_w_hat[i], np.add(np.sqrt(v_w_hat[i]), eps)), lr))
                self.bias[i] = np.subtract(self.bias[i], np.multiply(np.divide(m_b_hat[i], np.add(np.sqrt(v_b_hat[i]), eps)), lr))


    def nadam(self):
        data_input_row, data_input_col = data_input.shape
        m_w = []
        m_b = []
        v_w = []
        v_b = []
        for i in range(nhl + 1):
            m_w.append(np.zeros((self.nodes[i], self.nodes[i+1])))
            m_b.append(np.zeros((self.nodes[i+1])))
            v_w.append(np.zeros((self.nodes[i], self.nodes[i+1])))
            v_b.append(np.zeros((self.nodes[i+1])))

        for epoch in range(e):
            total_del_w = []
            total_del_b = []
    
            for i in range(nhl + 1):
                total_del_w.append(np.zeros((self.nodes[i], self.nodes[i+1])))
                total_del_b.append(np.zeros((self.nodes[i+1])))

            for start in range(0, data_input_row, b):
                end = start + b
                if end > data_input_row:
                    end = data_input_row

                x = data_input[start:end, : ]
                y = data_output[start:end, : ]

                A, h, op = self.feedForward(x, y, self.weight, self.bias)

                del_w, del_b = self.backProp(A, h, y, op)

                for i in range(nhl + 1):
                    total_del_w[i] = np.add(total_del_w[i], del_w[i])
                    total_del_b[i] = np.add(total_del_b[i], del_b[i])


            m_w_hat = []
            m_b_hat = []
            v_w_hat = []
            v_b_hat = []
            for i in range(nhl + 1):
                m_w[i] = np.add(np.multiply(m_w[i], beta1), np.multiply(total_del_w[i], 1 - beta1))
                m_b[i] = np.add(np.multiply(m_b[i], beta1), np.multiply(total_del_b[i], 1 - beta1))
                v_w[i] = np.add(np.multiply(v_w[i], beta2), np.multiply(np.power(total_del_w[i], 2), 1 - beta2))
                v_b[i] = np.add(np.multiply(v_b[i], beta2), np.multiply(np.power(total_del_b[i], 2), 1 - beta2))

                m_w_hat.append(np.divide(m_w[i], 1 - np.power(beta1, i+1)))
                m_b_hat.append(np.divide(m_b[i], 1 - np.power(beta1, i+1)))
                v_w_hat.append(np.divide(v_w[i], 1 - np.power(beta2, i+1)))
                v_b_hat.append(np.divide(v_b[i], 1 - np.power(beta2, i+1)))

                self.weight[i] = np.subtract(self.weight[i], np.multiply(np.divide(lr, np.sqrt(np.add(v_w_hat[i], eps))), np.multiply(beta1, m_w_hat[i]) + np.divide(np.multiply(total_del_w[i],(1 - beta1)), 1 - np.power(beta1, i+1))))
                self.bias[i] = np.subtract(self.bias[i], np.multiply(np.divide(lr, np.sqrt(np.add(v_b_hat[i], eps))), np.multiply(beta1, m_b_hat[i]) + np.divide(np.multiply(total_del_b[i],(1 - beta1)), 1 - np.power(beta1, i+1))))



    def train(self):
        if o == "sgd":
            self.sgd()
        elif o == "momentum":
            self.momentum()
        elif o == "nag":
            self.nag()
        elif o == "rmsprop":
            self.rmsprop()
        elif o == "adam":
            self.adam()
        elif o == "nadam":
            self.nadam()




# # command_line parameters
wp = "myprojectname"
we = "myname"
d = "fashion_mnist"
e = 5
b = 16
l = "cross_entropy"
o = "sgd"
lr = 0.1
m = 0.5
beta = 0.5
beta1 = 0.5
beta2 = 0.5
eps = 0.000001
w_d = 0.0
w_i = "random"
nhl = 3
sz = 64
a = "sigmoid"


label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# f, p = pyplot.subplots(2, 5, figsize=(10, 5))
# p = p.ravel()
# for i in range(len(label)):
#     index = np.where(y_train == i)[0][0]
#     p[i].imshow(x_train[index], cmap='gray')
#     p[i].set_title(label[i])
#     p[i].axis('off')
# pyplot.tight_layout()
# pyplot.show()

x_train,x_val, y_train, y_val=train_test_split(x_train,y_train, test_size=0.1,shuffle=True,random_state=42)
x_train = x_train.reshape((-1, 28 * 28))
x_test = x_train.reshape((-1, 28 * 28))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
data_input = x_train
data_output = y_train
data_input = np.divide(data_input, 255)
model = Model()
model.train()







# n = len(sys.argv)
# for i in range(1, n, 2):
#     if sys.argv[i] == "-wp" or "--wandb_project":
#         wp = sys.argv[i+1]
#     elif sys.argv[i] == "-we" or "--wandb_entity":
#         we = sys.argv[i+1]
#     elif sys.argv[i] == "-d" or "--dataset":
#         d = sys.argv[i+1]
#     elif sys.argv[i] == "-e" or "--epochs":
#         e = int(sys.argv[i+1])
#     elif sys.argv[i] == "-b" or "--batch_size":
#         b = int(sys.argv[i+1])
#     elif sys.argv[i] == "-l" or "--loss":
#         l = sys.argv[i+1]
#     elif sys.argv[i] == "-o" or "--optimizer":
#         o = sys.argv[i+1]
#     elif sys.argv[i] == "-lr" or "--learning_rate":
#         lr = double(sys.argv[i+1])
#     elif sys.argv[i] == "-m" or "--momentum":
#         m = double(sys.argv[i+1])
#     elif sys.argv[i] == "-beta" or "--beta":
#         beta = double(sys.argv[i+1])
#     elif sys.argv[i] == "-beta1" or "--beta1":
#         beta1 = double(sys.argv[i+1])
#     elif sys.argv[i] == "-beta2" or "--beta2":
#         beta2 = double(sys.argv[i+1])
#     elif sys.argv[i] == "-eps" or "--epsilon":
#         eps = double(sys.argv[i+1])
#     elif sys.argv[i] == "-w_d" or "--weight_decay":
#         w_d = double(sys.argv[i+1])
#     elif sys.argv[i] == "-w_i" or "--weight_init":
#         w_i = sys.argv[i+1]
#     elif sys.argv[i] == "-nhl" or "--num_layers":
#         nhl = int(sys.argv[i+1])
#     elif sys.argv[i] == "-sz" or "--hidden_size":
#         sz = int(sys.argv[i+1])
#     elif sys.argv[i] == "-a" or "--activation":
#         a = sys.argv[i+1]














