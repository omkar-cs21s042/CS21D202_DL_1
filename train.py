import sys
import numpy as np

# command_line parameters
wp = "myprojectname"
we = "myname"
d = "fashion_mnist"
e = 1
b = 4
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
nhl = 1
sz = 4
a = "sigmoid"


n = len(sys.argv)
for i in range(1, n, 2):
    if sys.argv[i] == "-wp" or "--wandb_project":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-we" or "--wandb_entity":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-d" or "--dataset":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-e" or "--epochs":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-b" or "--batch_size":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-l" or "--loss":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-o" or "--optimizer":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-lr" or "--learning_rate":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-m" or "--momentum":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-beta" or "--beta":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-beta1" or "--beta1":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-beta2" or "--beta2":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-eps" or "--epsilon":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-w_d" or "--weight_decay":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-w_i" or "--weight_init":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-nhl" or "--num_layers":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-sz" or "--hidden_size":
        wp = sys.argv[i+1]
    elif sys.argv[i] == "-a" or "--activation":
        wp = sys.argv[i+1]


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def dSigmoid(x):
    return np.multiply(sigmoid(x),np.multiply(-1, np.subtract(sigmoid(x), 1)))

def cross_entropy(y, op):
    return -np.sum(np.add(y,np.log(op)))

def gFunction(x):
    if a == "sigmoid":
        return dSigmoid(x)

def activationFunction(x):
    if a == "sigmoid":
        return sigmoid(x)

def outputFunction(y, op):
    if l == "cross_entropy":
        return cross_entropy(y, op)


# There are total L layers (does not include input layer) and L-1 hidden layers.
# Each neuron has pre-activation "a" and activation "h".
# If the size of layer is n then we have batch x n size vector of "a" and "h" for that layer
# size(w1) = size(L0) x size(L1)
# size(b1) = size(l1)

class model:
    def __init__(self):
        input_row_size, input_col_size = data_input.shape
        output_row_size, output_col_size = data_output.shape
        self.nodes = []
        for i in range(nhl+3):
            if i == 0:
                self.nodes.append(input_nodes)
            elif i == nhl + 2:
                self.nodes.append(output_nodes)
            else:
                self.nodes.append(sz)

        self.weight = []
        for i in range(nhl+2):
            self.weight.append(np.zeros(nodes[i], nodes[i+1]))

        self.bias = []
        for i in range(nhl+2):
            self.bias.append(np.zeros(1))

        self.a_batch = []
        self.h_batch = []

    def feedForward(self):
        for i in range(0, input_size, b):
            A =[]
            h =[]
            y = []
            start = i
            end = start + b
            input_row_size, input_col_size = data_input.shape
            if end > input_row_size:
                end = input_row_size

            x = data_input[start:end, :]
            y = data_output[start:end,:]

            # first layer
            A.append(np.add(np.matmul(x, self.weight[0]),self.bias[0]))
            h.append(activationFunction(A[0]))
            for j in range(1, nhl):
                A.append(np.add(np.matmul(h[len(h) - 1], self.weight[j]), self.bias[j]))
                h.append(activationFunction(A[len(A)-1]))
            A.append(np.add(np.matmul(h[len(h) - 1], self.weight[len(self.weight)-1]), self.bias[len(self.bias)-1]))
            h.append(outputFunction(y, A[len(A)-1]))

            self.a_batch.append(A)
            self.h_batch.append(h)

    def backProp(self):
        for batch_no in range(len(self.a_batch)):
            A = self.a_batch[batch_no]
            h = self.h_batch[batch_no]
            start = batch_no * b
            end = start + b
            input_row_size, input_col_size = data_input.shape
            if end > input_row_size:
                end = input_row_size

            y = data_output[start:end,:]

            del_a = []
            del_w = []
            del_b = []
            del_h = []
            del_a.append(np.multiply(np.subtract(y, h[len(h)-1]), -1))
            for i in range(nhl + 1, -1, -1):
                del_w.append(np.matmul(h[i-1].transpose(), del_a[len(del_a)-1]))
                del_b.append(np.sum(del_a[len(del_a)-1]))  # doubt regarding dimension
                del_h.append(np.matmul(del_a[len(del_a) - 1], self.weight[i].transpose()))
                del_a.append(np.multiply(del_h[len(del_h) - 1], gFunction(A[i-1])))







