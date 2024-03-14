import sys
import numpy as np
import math

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
        we = sys.argv[i+1]
    elif sys.argv[i] == "-d" or "--dataset":
        d = sys.argv[i+1]
    elif sys.argv[i] == "-e" or "--epochs":
        e = int(sys.argv[i+1])
    elif sys.argv[i] == "-b" or "--batch_size":
        b = int(sys.argv[i+1])
    elif sys.argv[i] == "-l" or "--loss":
        l = sys.argv[i+1]
    elif sys.argv[i] == "-o" or "--optimizer":
        o = sys.argv[i+1]
    elif sys.argv[i] == "-lr" or "--learning_rate":
        lr = double(sys.argv[i+1])
    elif sys.argv[i] == "-m" or "--momentum":
        m = double(sys.argv[i+1])
    elif sys.argv[i] == "-beta" or "--beta":
        beta = double(sys.argv[i+1])
    elif sys.argv[i] == "-beta1" or "--beta1":
        beta1 = double(sys.argv[i+1])
    elif sys.argv[i] == "-beta2" or "--beta2":
        beta2 = double(sys.argv[i+1])
    elif sys.argv[i] == "-eps" or "--epsilon":
        eps = double(sys.argv[i+1])
    elif sys.argv[i] == "-w_d" or "--weight_decay":
        w_d = double(sys.argv[i+1])
    elif sys.argv[i] == "-w_i" or "--weight_init":
        w_i = sys.argv[i+1]
    elif sys.argv[i] == "-nhl" or "--num_layers":
        nhl = int(sys.argv[i+1])
    elif sys.argv[i] == "-sz" or "--hidden_size":
        sz = int(sys.argv[i+1])
    elif sys.argv[i] == "-a" or "--activation":
        a = sys.argv[i+1]





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
                self.nodes.append(input_col_size)
            elif i == nhl + 2:
                self.nodes.append(output_col_size)
            else:
                self.nodes.append(sz)

        if w_i == "random":
            init_range = 0.1
        else:
            init_range = math.sqrt(6/(input_col_size + output_col_size))
        self.weight = []
        for i in range(nhl+2):
            self.weight.append(np.random.uniform(-init_range, init_range, size = (self.nodes[i], self.nodes[i+1])))

        self.bias = []
        for i in range(nhl+2):
            self.bias.append(np.random.uniform(-init_range, init_range, size = (self.nodes[i+1])))


    def feedForward(self, x, y):
        A =[]
        h =[]

        # first layer
        A.append(np.add(np.matmul(x, self.weight[0]),self.bias[0]))
        h.append(activationFunction(A[0]))
        for j in range(1, nhl):
            A.append(np.add(np.matmul(h[len(h) - 1], self.weight[j]), self.bias[j]))
            h.append(activationFunction(A[len(A)-1]))
        A.append(np.add(np.matmul(h[len(h) - 1], self.weight[len(self.weight)-1]), self.bias[len(self.bias)-1]))
        h.append(outputFunction(y, A[len(A)-1]))

        return A,h


    def backProp(self, A, h, y):
        del_a = []
        del_w = []
        del_b = []
        del_h = []
        del_a.append(np.multiply(np.subtract(y, h[len(h)-1]), -1))
        for i in range(nhl + 1, -1, -1):
            del_w.append(np.matmul(h[i-1].transpose(), del_a[len(del_a)-1]))
            del_b.append(np.sum(del_a[len(del_a)-1]), axis=0)  # doubt regarding dimension
            del_h.append(np.matmul(del_a[len(del_a) - 1], self.weight[i].transpose()))
            del_a.append(np.multiply(del_h[len(del_h) - 1], gFunction(A[i-1])))
        return del_w.reverse(), del_b.reverse()

    def sgd(self):
        data_input_row, data_input_col = data_input.shape
        for start in range(0, data_input_row, b):
            end = start + b
            if end > data_input_row:
                end = data_input_row

            x = data_input[start:end, : ]
            y = data_output[start:end, : ]

            A, h = self.feedForward(x, y)
            del_w, del_b = self.backProp(A, h, y)

    def momentum(self):
        data_input_row, data_input_col = data_input.shape
        prev_uw = []
        prev_ub = []
        for i in range(nhl + 2):
            prev_uw.append(np.zeros(self.nodes[i], self.nodes[i+1]))
            prev_ub.append(np.zeros(self.nodes[i+1]))

        for epoch in range(e):
            total_del_w = []
            total_del_b = []
            for i in range(nhl + 2):
                total_del_w.append(np.zeros(self.nodes[i], self.nodes[i+1]))
                total_del_b.append(np.zeros(self.nodes[i+1]))
            for start in range(0, data_input_row, b):
                end = start + b
                if end > data_input_row:
                    end = data_input_row

                x = data_input[start:end, : ]
                y = data_output[start:end, : ]

                A, h = self.feedForward(x, y)
                del_w, del_b = self.backProp(A, h, y)
                for i in range(len(total_del_w)):
                    total_del_w[i] = np.sum(total_del_w[i], del_w[i])
                    total_del_b[i] = np.sum(total_del_b[i], del_b[i])

            uw = []
            ub = []
            for i in range(nhl+2):
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
        for i in range(nhl + 2):
            prev_vw.append(np.zeros(self.nodes[i], self.nodes[i+1]))
            prev_vb.append(np.zeros(self.nodes[i+1]))

        for epoch in range(e):
            v_w = []
            v_b = []
            total_del_w = []
            total_del_b = []
            for i in range(nhl + 2):
                total_del_w.append(np.zeros(self.nodes[i], self.nodes[i+1]))
                total_del_b.append(np.zeros(self.nodes[i+1]))
                v_w.append(np.multiply(prev_vw[i], m))
                v_b.append(np.multiply(prev_vb[i], m))

            for start in range(0, data_input_row, b):
                end = start + b
                if end > data_input_row:
                    end = data_input_row

                x = data_input[start:end, : ]
                y = data_output[start:end, : ]

                A, h = self.feedForward(x, y)

                for i in range(len(A)):
                    A[i] = np.subtract(A[i], v_w[i])
                    h[i] = np.subtract(h[i], v_b[i])

                del_w, del_b = self.backProp(A, h, y)
                for i in range(nhl + 2):
                    total_del_w[i] = np.sum(total_del_w[i], del_w[i])
                    total_del_b[i] = np.sum(total_del_b[i], del_b[i])

            vw = []
            vb = []
            for i in range(nhl + 2):
                vw.append(np.add(np.multiply(prev_vw[i], m), np.multiply(total_del_w, lr)))
                vb.append(np.add(np.multiply(prev_vb[i], m), np.multiply(total_del_b, lr)))
                self.weight[i] = np.subtract(self.weight[i], vw[i])
                self.bias[i] = np.subtract(self.bias[i], vb[i])

            prev_vw = vw 
            prev_vb = vb








