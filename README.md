# Assignment 1

This directory contains solution to [assignment 1]() of Fundamentals of Deep Learning (CS6910), Spring 2022(implementation of a **Feed Forward Neural Network**) which can be trained to work on numerical data.

The solution report with results can be found [here]().

I have used object-oriented approach to implement the assignment.
Model is the main class which maintains attributes weights and bias for ach layer. I also maintain another list to keep track of nodes in each layer.
Model.train() function has a switch case which invokes the optimizer based on the command line arguments passed by the user.
Similarly switches are also maintained for activation function and loss function.


## Contents of the File
The main content of this file is the class ```class Model``` which implements model of neural network. It initializes the weight and biases of each layer. The model also initializes the weights and biases using zxavier method or random method based on user's input.
```python
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
```

### The activation function implemented in the assignment:  
**activation functions** : ```sigmoid```, ```tanh``` and ```relu```  




### The functions implemented for the algorithms: 
#### **optimizer** : ```sgd```, ```momentum```, ```nesterov```, ```rmsprop```, ```adam```, ```nadam``` 
 
The loss functions implemented
#### **loss** = {'cross_entropy_loss', 'squared_error'}


### The function ```feedForward``` has been implemented:
Takes dataset ```x```, ```weights``` and ```bias``` as input.
Returns pre-activation and  activation for all layers and output for final layer. 


### The function ```backProp``` has been implemented:
Takes activation ```h```, pre-activation ```A``` and  output ```op``` as input.
Compute and update the gradients of each layer returs the list ```dw``` and ```db``` which contain gradients for all layers.



# Training the model 
```
The model can be trained by running train.py
The script can take following parameters as command line arguments -
-wp, --wandb_project    default=myprojectname	             //Project name used to track experiments in Weights & Biases dashboard
-we, --wandb_entity     default=myname	                   //Wandb Entity used to track experiments in the Weights & Biases dashboard.
-d, --dataset	        default=fashion_mnist	             //choices: ["mnist", "fashion_mnist"]
-e, --epochs	        default=1	                         //Number of epochs to train neural network.
-b, --batch_size	default=4	                         //Batch size used to train neural network.
-l, --loss	        default=cross_entropy	             //choices: ["mean_squared_error", "cross_entropy"]
-o, --optimizer	        default=sgd	                       //choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
-lr, --learning_rate	default=0.1	                       //Learning rate used to optimize model parameters
-m, --momentum	        default=0.5	                       //Momentum used by momentum and nag optimizers.
-beta, --beta	        default=0.5	                       //Beta used by rmsprop optimizer
-beta1, --beta1	        default=0.5	                       //Beta1 used by adam and nadam optimizers.
-beta2, --beta2	        default=0.5	                       //Beta2 used by adam and nadam optimizers.
-eps, --epsilon	        default=0.000001	                 //Epsilon used by optimizers.
-w_d, --weight_decay	default=.0	                       //Weight decay used by optimizers.
-w_i, --weight_init	default=random	                   //choices: ["random", "Xavier"]
-nhl, --num_layers	default=1	                         //Number of hidden layers used in feedforward neural network.
-sz, --hidden_size	default=4	                         //Number of hidden neurons in a feedforward layer.
-a, --activation	default=sigmoid	                   //choices: ["identity", "sigmoid", "tanh", "ReLU"]
```
