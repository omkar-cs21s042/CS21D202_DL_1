import sys
import numpy as np
import math
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import tensorflow as tf
import wandb

def tanh(x):
  return np.tanh(x)

def dtanh(x):
    return 1 - np.square(np.tanh(x))

def relu(x):
  return np.where(np.asarray(x) > 0, x, 0)

def drelu(x):
    return np.where(x <= 0, 0, 1)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def dSigmoid(x):
    return np.multiply(sigmoid(x),np.subtract(1, sigmoid(x)))

def cross_entropy(y, op):
    return -np.sum(np.add(y.transpose(),np.log(op)))

def squared_error(y, op):
  return np.sum(np.square(y - op))

def gFunction(x):
    if a == "sigmoid":
        return dSigmoid(x)
    elif a == "tanh":
        return dtanh(x)
    elif a == "relu":
        return drelu(x)

def activationFunction(x):
    if a == "sigmoid":
        return sigmoid(x)
    elif a == "tanh":
        return tanh(x)
    elif a == "relu":
        return relu(x)

def computeLoss(y, op):
    if l == "cross_entropy":
        return cross_entropy(y, op)
    elif l == "squared_error":
        return squared_error(y, op)

def outputFunction(x):
    f = np.exp(x - np.max(x))  
    return f / f.sum(axis=0)


def computeAccuracy(y, op):
    op = op.argmax(axis=0)
    op = to_categorical(op, num_classes=10) 
    op = op.transpose()
    return (np.mean(np.equal(y.transpose(),op))-op_) * 100

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
            self.weight.append(np.random.uniform(-init_range, init_range, size = (self.nodes[i+1], self.nodes[i])).astype(np.longdouble))

        self.bias = []
        for i in range(nhl+1):
            self.bias.append(np.random.uniform(-init_range, init_range, size = (self.nodes[i+1],1)).astype(np.longdouble))


    def feedForward(self, x, wgt, bi):
    
        A =[]
        h =[]

        # first layer
        h.append(x.transpose())
        for j in range(1, nhl + 1):
            A.append(np.add(np.matmul(wgt[j-1], h[len(h) - 1]), bi[j-1]))
            h.append(activationFunction(A[len(A)-1]))
        A.append(np.add(np.matmul(wgt[len(wgt)-1], h[len(h) - 1]), bi[len(bi)-1]))
        op = outputFunction(A[len(A)-1])

        return A,h,op


    def backProp(self, A, h, y, op):
        del_w = []
        del_b = []
        del_a = []
        del_h = []
        
        
        del_a.append(np.multiply(np.subtract(y.transpose(), op), -1))
        for i in range(nhl + 1, 0, -1):
            del_w.append(np.matmul(del_a[len(del_a)-1], h[i-1].transpose()))

            temp = np.sum(del_a[len(del_a)-1], axis=1)
            temp_r = temp.shape
            temp = temp.reshape((temp_r[0], 1))

            del_b.append(temp)

            if i - 2 >= 0:
                del_h.append(np.matmul(self.weight[i-1].transpose(), del_a[len(del_a) - 1]))
                del_a.append(np.multiply(del_h[len(del_h) - 1], gFunction(A[i-2])))
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


                A, h, op = self.feedForward(x, self.weight, self.bias)

                del_w, del_b = self.backProp(A, h, y, op)
                


                for i in range(nhl+1):
                    # print(del_w[i])
                    # print("===============")
                    self.weight[i] = np.subtract(self.weight[i], np.multiply(del_w[i], lr))
                    # print(self.weight[i])
                    # print(del_b)
                    self.bias[i] = np.subtract(self.bias[i], np.multiply(del_b[i], lr))
                    # bub()

            tr, tc = x_val.shape
            A, h, op = self.feedForward(x_val, self.weight, self.bias)

            print("epoch = " + str(epoch))
            print("ValAccuracy = " + str(computeAccuracy(y_val, op)))
            wandb.log({"epoch":epoch, "ValAccuracy" : computeAccuracy(y_val, op)})

        A, h, op = self.feedForward(x_test, self.weight, self.bias)

        wandb.log({"TestAccuracy": computeAccuracy(y_test, op)})



    def momentum(self):
        data_input_row, data_input_col = data_input.shape
        prev_uw = []
        prev_ub = []

        for i in range(nhl + 1):
            prev_uw.append(np.zeros((self.nodes[i+1], self.nodes[i])))
            prev_ub.append(np.zeros((self.nodes[i+1])))


        for epoch in range(e):
            for start in range(0, data_input_row, b):
                end = start + b
                if end > data_input_row:
                    end = data_input_row

                x = data_input[start:end, : ]
                y = data_output[start:end, : ]

                A, h, op = self.feedForward(x, self.weight, self.bias)
                del_w, del_b = self.backProp(A, h, y, op)
    

                uw = []
                ub = []
                for i in range(nhl+1):
                    uw.append(np.add(np.multiply(prev_uw[i], m),np.multiply(del_w[i], lr)))
                    temp = np.sum(np.add(np.multiply(prev_ub[i], m),np.multiply(del_b[i], lr)), axis=1)
                    temp_r = temp.shape
                    temp = temp.reshape((temp_r[0], 1))
                    ub.append(temp)
                    self.weight[i] = np.subtract(self.weight[i], uw[i])
                    self.bias[i] = np.subtract(self.bias[i], ub[i])
                

                prev_uw = uw
                prev_ub = ub

            tr, tc = x_val.shape
            A, h, op = self.feedForward(x_val, self.weight, self.bias)

            print("epoch = " + str(epoch))
            print("ValAccuracy = " + str(computeAccuracy(y_val, op)))
            wandb.log({"epoch":epoch, "ValAccuracy" : computeAccuracy(y_val, op)})
        A, h, op = self.feedForward(x_test, self.weight, self.bias)

        wandb.log({"TestAccuracy": computeAccuracy(y_test, op)})


    def nag(self):
        data_input_row, data_input_col = data_input.shape
        prev_vw = []
        prev_vb = []
        for i in range(nhl + 1):
            prev_vw.append(np.zeros((self.nodes[i], self.nodes[i+1])))
            prev_vb.append(np.zeros((self.nodes[i+1])))

        for epoch in range(e):            

            for start in range(0, data_input_row, b):
                v_w = []
                v_b = []
                nag_w = []
                nag_b = []
                for i in range(nhl + 1):
                    v_w.append(np.multiply(prev_vw[i], m))
                    v_b.append(np.multiply(prev_vb[i], m))
                    nag_w.append(np.subtract(self.weight[i], v_w[i].transpose()))
                    nag_b.append(np.subtract(self.bias[i], v_b[i].transpose()))

                end = start + b
                if end > data_input_row:
                    end = data_input_row

                x = data_input[start:end, : ]
                y = data_output[start:end, : ]

                A, h, op = self.feedForward(x, self.weight, self.bias)

                del_w, del_b = self.backProp(A, h, y, op)
               
                vw = []
                vb = []
                for i in range(nhl + 1):
                    vw.append(np.add(np.multiply(prev_vw[i], m), np.multiply(del_w[i], lr).transpose()))
                    vb.append(np.add(np.multiply(prev_vb[i], m), np.multiply(del_b[i], lr).transpose()))
                    self.weight[i] = np.subtract(self.weight[i], vw[i].transpose())
                    self.bias[i] = np.subtract(self.bias[i], vb[i].transpose())

                prev_vw = vw 
                prev_vb = vb


            tr, tc = x_val.shape
            A, h, op = self.feedForward(x_val, self.weight, self.bias)

            print("epoch = " + str(epoch))
            print("ValAccuracy = " + str(computeAccuracy(y_val, op)))
            wandb.log({"epoch":epoch, "ValAccuracy" : computeAccuracy(y_val, op)})

        A, h, op = self.feedForward(x_test, self.weight, self.bias)

        wandb.log({"TestAccuracy": computeAccuracy(y_test, op)})


    def rmsprop(self):
        data_input_row, data_input_col = data_input.shape
        v_w = []
        v_b = []
        for i in range(nhl + 1):
            v_w.append(np.zeros((self.nodes[i], self.nodes[i+1])))
            v_b.append(np.zeros((self.nodes[i+1])))

        for epoch in range(e):    

            for start in range(0, data_input_row, b):
                end = start + b
                if end > data_input_row:
                    end = data_input_row

                x = data_input[start:end, : ]
                y = data_output[start:end, : ]

                A, h, op = self.feedForward(x, self.weight, self.bias)

                del_w, del_b = self.backProp(A, h, y, op)


                for i in range(nhl + 1):
                    v_w[i] = np.add(np.multiply(v_w[i], beta), np.multiply(np.power(del_w[i], 2), 1 - beta).transpose())
                    v_b[i] = np.add(np.multiply(v_b[i], beta), np.multiply(np.power(del_b[i], 2), 1 - beta).transpose())

                    self.weight[i] = np.subtract(self.weight[i], np.multiply(np.divide(del_w[i], np.add(np.sqrt(v_w[i]), eps).transpose()), lr))
                    self.bias[i] = np.subtract(self.bias[i], np.multiply(np.divide(del_b[i], np.add(np.sqrt(v_b[i]), eps).transpose()), lr))

            tr, tc = x_val.shape
            A, h, op = self.feedForward(x_val, self.weight, self.bias)

            print("epoch = " + str(epoch))
            print("ValAccuracy = " + str(computeAccuracy(y_val, op)))
            wandb.log({"epoch":epoch, "ValAccuracy" : computeAccuracy(y_val, op)})

        A, h, op = self.feedForward(x_test, self.weight, self.bias)

        wandb.log({"TestAccuracy": computeAccuracy(y_test, op)})



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

            for start in range(0, data_input_row, b):
                end = start + b
                if end > data_input_row:
                    end = data_input_row

                x = data_input[start:end, : ]
                y = data_output[start:end, : ]

                A, h, op = self.feedForward(x, self.weight, self.bias)

                del_w, del_b = self.backProp(A, h, y, op)


                m_w_hat = []
                m_b_hat = []
                v_w_hat = []
                v_b_hat = []
                for i in range(nhl + 1):
                    m_w[i] = np.add(np.multiply(m_w[i], beta1), np.multiply(del_w[i], 1 - beta1).transpose())
                    m_b[i] = np.add(np.multiply(m_b[i], beta1), np.multiply(del_b[i], 1 - beta1).transpose())
                    v_w[i] = np.add(np.multiply(v_w[i], beta2), np.multiply(np.power(del_w[i], 2), 1 - beta2).transpose())
                    v_b[i] = np.add(np.multiply(v_b[i], beta2), np.multiply(np.power(del_b[i], 2), 1 - beta2).transpose())

                    m_w_hat.append(np.divide(m_w[i], 1 - np.power(beta1, i+1)))
                    m_b_hat.append(np.divide(m_b[i], 1 - np.power(beta1, i+1)))
                    v_w_hat.append(np.divide(v_w[i], 1 - np.power(beta2, i+1)))
                    v_b_hat.append(np.divide(v_b[i], 1 - np.power(beta2, i+1)))

                    self.weight[i] = np.subtract(self.weight[i], np.multiply(np.divide(m_w_hat[i], np.add(np.sqrt(v_w_hat[i]), eps)), lr).transpose())
                    self.bias[i] = np.subtract(self.bias[i], np.multiply(np.divide(m_b_hat[i], np.add(np.sqrt(v_b_hat[i]), eps)), lr).transpose())

            tr, tc = x_val.shape
            A, h, op = self.feedForward(x_val, self.weight, self.bias)

            print("epoch = " + str(epoch))
            print("ValAccuracy = " + str(computeAccuracy(y_val, op)))
            wandb.log({"epoch":epoch, "ValAccuracy" : computeAccuracy(y_val, op)})

        A, h, op = self.feedForward(x_test, self.weight, self.bias)

        wandb.log({"TestAccuracy": computeAccuracy(y_test, op)})


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
            

            for start in range(0, data_input_row, b):
                end = start + b
                if end > data_input_row:
                    end = data_input_row

                x = data_input[start:end, : ]
                y = data_output[start:end, : ]

                A, h, op = self.feedForward(x, self.weight, self.bias)

                del_w, del_b = self.backProp(A, h, y, op)




                m_w_hat = []
                m_b_hat = []
                v_w_hat = []
                v_b_hat = []
                for i in range(nhl + 1):
                    m_w[i] = np.add(np.multiply(m_w[i], beta1), np.multiply(del_w[i], 1 - beta1).transpose())
                    m_b[i] = np.add(np.multiply(m_b[i], beta1), np.multiply(del_b[i], 1 - beta1).transpose())
                    v_w[i] = np.add(np.multiply(v_w[i], beta2), np.multiply(np.power(del_w[i], 2), 1 - beta2).transpose())
                    v_b[i] = np.add(np.multiply(v_b[i], beta2), np.multiply(np.power(del_b[i], 2), 1 - beta2).transpose())

                    m_w_hat.append(np.divide(m_w[i], 1 - np.power(beta1, i+1)))
                    m_b_hat.append(np.divide(m_b[i], 1 - np.power(beta1, i+1)))
                    v_w_hat.append(np.divide(v_w[i], 1 - np.power(beta2, i+1)))
                    v_b_hat.append(np.divide(v_b[i], 1 - np.power(beta2, i+1)))


                    self.weight[i] = np.subtract(self.weight[i], np.multiply(np.divide(lr, np.sqrt(np.add(v_w_hat[i].transpose(), eps))), np.multiply(beta1, m_w_hat[i].transpose()) + np.divide(np.multiply(del_w[i],(1 - beta1)), 1 - np.power(beta1, i+1))))
                    self.bias[i] = np.subtract(self.bias[i], np.multiply(np.divide(lr, np.sqrt(np.add(v_b_hat[i].transpose(), eps))), np.multiply(beta1, m_b_hat[i].transpose()) + np.divide(np.multiply(del_b[i],(1 - beta1)), 1 - np.power(beta1, i+1))))

            tr, tc = x_val.shape
            A, h, op = self.feedForward(x_val, self.weight, self.bias)

            print("epoch = " + str(epoch))
            print("ValAccuracy = " + str(computeAccuracy(y_val, op)))
            wandb.log({"epoch":epoch, "ValAccuracy" : computeAccuracy(y_val, op)})

        A, h, op = self.feedForward(x_test, self.weight, self.bias)

        wandb.log({"TestAccuracy": computeAccuracy(y_test, op)})



# # command_line parameters
wp = "myprojectname"
we = "myname"
d = "fashion_mnist"
e = 10
b = 16
l = "cross_entropy"
o = "sgd"
lr = 0.001
m = 0.5
beta = 0.5
beta1 = 0.5
beta2 = 0.5
eps = 0.000001
w_d = 0.0
w_i = "xavier"
nhl = 5
sz = 128
op_ = 0.1
a = "relu"
np.seterr(divide = 'ignore') 

label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

y_train_n = y_train

x_train,x_val, y_train, y_val=train_test_split(x_train,y_train, test_size=0.1,shuffle=True,random_state=42)

x_train = x_train.reshape((-1, 28 * 28))
x_val = x_val.reshape((-1, 28 * 28))
x_test = x_test.reshape((-1, 28 * 28))
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)
# x_train = np.divide(x_train, 255)
data_input = x_train
data_output = y_train


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




# model = Model()
# model.train()

# A, h, op = model.feedForward(x_test, model.weight, model.bias)

# print("accuracy = " + str(computeAccuracy(y_test, op)))

sweep_config = {
    "method":"bayes"
}
metric = {
    "name" : "val_accuracy",
    "goal" : "maximize"
}

sweep_config['metric']=metric

parameter_dict = {
      "epochs" : {
      "values" : [e]
      },
      "learning_rate" : {
      "values" : [lr]
      },
      "h_layers" : {
      "values" : [nhl]
      },
      "neurons" : {
      "values" : [sz]
      },
      "optimizer" : {
      "values" : [o]
      },
      "batch_size" : {
      "values" : [b]
      },
      "activation" : {
      "values" : [a]
      }
  }

sweep_config['parameters']=parameter_dict

wandb.login(key='')



sweep_id = wandb.sweep(sweep=sweep_config, project=wp)



def model_train_wandb():

  with wandb.init() as run:
    config=wandb.config
    run_name="-ac_"+wandb.config.activation+"-hs_"+str(wandb.config.h_layers)+"-hl_"+str(wandb.config.h_layers)+"-lr_"+str(wandb.config.learning_rate)+"-bs_"+str(wandb.config.batch_size)+"-e_"+str(wandb.config.epochs)
    wandb.run.name = run_name

    model = Model()
    model.train()

    A, h, op = model.feedForward(x_test, model.weight, model.bias)

    wandb.log({"TestAccuracy": computeAccuracy(y_test, op)})


wandb.agent(sweep_id, model_train_wandb, count=1)


"""###Predictions and accuracy using validation data and test data
(Using best identified model)
"""


wandb.init()

activation = 'relu'
batch_size = 64
epochs = 10
h_layers = 5
learning_rate = 0.001
neurons = 128
optimizer = 'nesterov'


"""###Confusion Matrix with best model:"""

model = Model()
model.train()

A, h, op = model.feedForward(x_test, model.weight, model.bias)

wandb.log({"TestAccuracy": computeAccuracy(y_test, op)})

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=y_test, preds=y_test,
                        class_names=labels)})

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
img_list = []
idx_list = []
count = 0
for i in range(y_train_n.shape[0]):
    if y_train_n[i] in idx_list:
      continue
    else:
      idx_list.append(y_train_n[i])
      img_list.append(x_train_n[i])
    if len(idx_list) == 10:
      break

wandb_list = []
for j in range(10):
  plt.imshow(img_list[j], cmap='gray')
  plt.title("class: "+ str(idx_list[j]) + " ("+labels[idx_list[j]]+")")
  #wandb.log({"img": [wandb.Image(plt.imshow(img_list[j], cmap='gray'), caption="class: "+ str(idx_list[j]) + " ("+labels[idx_list[j]]+")")]})
  plt.show()

wandb.log({"class: "+ str(idx_list[0]) + " ("+labels[idx_list[0]]+")": [wandb.Image(plt.imshow(img_list[0], cmap='gray'))],
              "class: "+ str(idx_list[1]) + " ("+labels[idx_list[1]]+")": [wandb.Image(plt.imshow(img_list[1], cmap='gray'))],
             "class: "+ str(idx_list[2]) + " ("+labels[idx_list[2]]+")": [wandb.Image(plt.imshow(img_list[2], cmap='gray'))],
             "class: "+ str(idx_list[3]) + " ("+labels[idx_list[3]]+")": [wandb.Image(plt.imshow(img_list[3], cmap='gray'))],
             "class: "+ str(idx_list[4]) + " ("+labels[idx_list[4]]+")": [wandb.Image(plt.imshow(img_list[4], cmap='gray'))],
             "class: "+ str(idx_list[5]) + " ("+labels[idx_list[5]]+")": [wandb.Image(plt.imshow(img_list[5], cmap='gray'))],
             "class: "+ str(idx_list[6]) + " ("+labels[idx_list[6]]+")": [wandb.Image(plt.imshow(img_list[6], cmap='gray'))],
             "class: "+ str(idx_list[7]) + " ("+labels[idx_list[7]]+")": [wandb.Image(plt.imshow(img_list[7], cmap='gray'))],
             "class: "+ str(idx_list[8]) + " ("+labels[idx_list[8]]+")": [wandb.Image(plt.imshow(img_list[8], cmap='gray'))],
             "class: "+ str(idx_list[9]) + " ("+labels[idx_list[9]]+")": [wandb.Image(plt.imshow(img_list[9], cmap='gray'))]})

# images = wandb.Image(img_list, caption="class", Output, Bottom, Input)
# wandb.log({"example": [wandb.Image(img) for img in img_list]})

wandb.finish()













