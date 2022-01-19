Julia Adamczyk
Project 3
Submission date: 11/10/2021

Neural network file contains 4 functions
1. softmax(x) - calculates softmax activation function of the output layer, layer x represents the numbers for each output class. What softmax does is it takes those numbers and for each class it calculates probability of each class given that sample. 
2. calculate loss - it performs forward propagation, maps labels into 2D vector according to the rule: label 0: [1,0], label 1: [0, 1] and calculates the loss given by the formula. 
3. predict - does forward propagation for a single sample, then determines the lable by returning the maximum of probablilities returned by the softmax function.
4. build model - trains weights and biases for each model. Firstly, the learning rate, initial weights and initial bias are initialized. The model is initialized with the initial values of weights and biases. Then the loop trains for num_passes. For each pass through the network, following activities are performed:
a. forward propagation
b. label is fit for broadcasting 
c. backpropagation calculations are performed given formulas in the document
note: biases are summed over all samples since the forward/back pass is done in a batch training manner
d. the model's current parameters are updated using learning rate and gradients
e. the trained model is returned after all passes are done

Notes: my graph for N_hiddims = 2 looks much different, the loss function returns 0.25 after last iteration which seems pretty reasonable
I might be doing something wrong, but why are n_hid_dims 3 and 4 returning similar result?