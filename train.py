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



# There are total L layers (does not include input layer) and L-1 hidden layers.
# Each neuron has pre-activation "a" and activation "h".
# If the size of layer is n then we have batch x n size vector of "a" and "h" for that layer
# size(w1) = size(L1) x size(L0)
# size(b1) = size(l1)

class model:
    def __init__(self, input_nodes, output_nodes):
        nodes = []
        for i in range(h+2):
            if i == 0:
                nodes.append(input_nodes)
            elif i == h + 1:
                nodes.append(output_nodes)
            else:
                nodes.append(sz)

        weight = []
        for i in range(1, h+2):
            weight.append(np.zeros(nodes[i], nodes[i-1]))

        bias = []
        for i in range(1, h+2):
            bias.append(nodes[i])

