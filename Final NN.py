# Feed Forward NN
# Eduardo Saldana
# 6612626

import numpy as np
import re
import matplotlib.pyplot as plt

# Opens the file and reads the first two numbers to know
# the number of inputs
# To use a different output change the name of the line below
file = open("L30fft16.out")
first = file.readline()
numbers = re.findall(r'\d+', first)
for i in range(len(numbers)):
    numbers[i] = int(numbers[i])

result = []
evidence = []

# Iterates through the file and appends the result on the result array
# and the evidence on the evidence array
for x in range(numbers[0]):
    temp = file.readline()
    temp = re.findall(r'\d+\.*\d*',temp)
    for i in range(len(temp)):
        temp[i] = float(temp[i])
    result_obtained = [temp[0]]
    result.append(result_obtained)
    temp.pop(0)
    for i in range(len(temp)):
        temp[i] = float(temp[i])
    evidence.append(temp)

#Normalizes the data
evidence_array = np.array(evidence)
evidence_array = evidence_array / np.amax(evidence_array)
result_array = np.array(result)

# The following code splits the data into different sets to be used as either training or testing
A_evidence = []
B_evidence = []
C_evidence = []

A_result = []
B_result = []
C_result = []

for i in range(len(result)):
    if i%3 == 0:
        A_evidence.append(evidence_array[i])
        A_result.append(result_array[i])
    elif i%3 == 1:
        B_evidence.append(evidence_array[i])
        B_result.append(result_array[i])
    else:
        C_evidence.append(evidence_array[i])
        C_result.append(result_array[i])

A_evidence = np.array(A_evidence)
B_evidence = np.array(B_evidence)
C_evidence = np.array(C_evidence)

A_result = np.array(A_result)
B_result = np.array(B_result)
C_result = np.array(C_result)

#Sigmoid function and derivative function
def sigmoid(x):
    x = -x
    return 1/(1+np.exp(x))

def derivative(x):
    return (1-x)*x

# Neural Network Class
class FeedForward:

    # Create the initial weights for the input and hidden layer
    # Its dimensions are based on the number of inputs and number of choosen number of hidden nodes
    W = []
    W.append(np.random.rand(evidence_array.shape[1], 10))
    W.append(np.random.rand(10, 1))

    # Initializes the layers to be used
    first = None
    second = None
    third = None

    layers = [
        first, second, third
    ]

    # This is where the current predicted result is stored
    predicted_result = None

    def __init__(self):
        pass
    
    def Feed_Forward(self, input):
        # First Layer is simply the input from the data set
        self.layers[0] = input
        # Second Layer is simply the multiplication of the input and the input layer through the sigmoid function
        self.layers[1] = sigmoid(np.dot(self.layers[0], self.W[0]))
        # Third layer is the multiplication of the previosu result multiplied by the hidden layer through the sigmoid function
        self.layers[2] = sigmoid(np.dot(self.layers[1], self.W[1]))
        # Out predicted result will be this last third layer
        self.predicted_result = self.layers[2]
        # It retures this predicted result
        return self.predicted_result
    

    def back_propagation(self, expected_result, obtaind_result):
        # Obtains error simply by subtracting the expected result from the one that it predicted
        error = expected_result - obtaind_result
        # Obtains the derrivative by using the error obtained and multiplying it by the derivative of the predicted classification
        delta = derivative(self.layers[2])
        delta = delta * error
        # Updates the hidden layer weights by adding/subracting the dot multiplication of itself and the delta result, then multiplying it by the alpha
        new_weights = np.dot(self.layers[1].T, delta)
        self.W[1] = self.W[1] + new_weights*alpha
        # Calculates the new error by doing a dot multiplication of the delta and the weights of the hidden layer
        error = np.dot(delta, self.W[1].T)
        # Gets the new delta by multiplying the new error by the derivative of the hidden layer
        delta = derivative(self.layers[1])
        delta = delta * error
        # Updates the input layer by doing the same process done with the previous layer but now using the new delta values
        new_weights = np.dot(self.layers[0].T, delta)
        self.W[0] = self.W[0] + new_weights*alpha
    
    
    # Feeds the input forward and then takes the predicted classification and calls for backpropagation
    def train_NN(self, input, expected_result):
        self.Feed_Forward(input)
        self.backpropagate(self.predicted_result, expected_result)
    
    # Uses the predicted classification and the actual needed one to backpropagtate and update the weights
    def backpropagate(self, result_obtained, expected_result):
        self.back_propagation(expected_result, result_obtained)

    
    # Given the input it will classify it with a value between 0 and 1
    def classify(self, input):
        answer = self.Feed_Forward(input)
        return answer



        




# Defines alpha (learning rate) and iterations (epochs)
iterations = 20000
alpha = 0.1

# Plot points to be included in the graph
plot_points = int(iterations/20)

# Create the neural networks with their names indicating the evidence they are using
NN = FeedForward()
AB = FeedForward()
AC = FeedForward()
BC = FeedForward()
Just_A = FeedForward()
Just_B = FeedForward()
Just_C = FeedForward()

# Records result in an array based on the Neural Network used
Squared_loss = []
AB_loss = []
AC_loss = []
BC_loss = []
Just_A_loss = []
Just_B_loss = []
Just_C_loss = []

# Creates the evidence and result arrays for each separate testing and training set
AB_evidence = np.concatenate((A_evidence, B_evidence))
AC_evidence = np.concatenate((A_evidence, C_evidence))
BC_evidence = np.concatenate((B_evidence, C_evidence))


AB_result = np.concatenate((A_result, B_result))
AC_result = np.concatenate((A_result, C_result))
BC_result = np.concatenate((B_result, C_result))

# training
for i in range(iterations+1):
    AB.train_NN(AB_evidence, AB_result)
    AC.train_NN(AC_evidence, AC_result)
    BC.train_NN(BC_evidence, BC_result)
    Just_A.train_NN(A_evidence, A_result)
    Just_B.train_NN(B_evidence, B_result)
    Just_C.train_NN(C_evidence, C_result)
    
    # Records the MSE of each Neural Network
    if i%plot_points==0:
        #alpha += 0.05
        AB_loss.append(np.mean(np.square(AB_result - AB.predicted_result)))
        AC_loss.append(np.mean(np.square(AC_result - AC.predicted_result)))
        BC_loss.append(np.mean(np.square(BC_result - BC.predicted_result)))
        Just_A_loss.append(np.mean(np.square(A_result - Just_A.predicted_result)))
        Just_B_loss.append(np.mean(np.square(B_result - Just_B.predicted_result)))
        Just_C_loss.append(np.mean(np.square(C_result - Just_C.predicted_result)))


print("MSE:")
print("AB MSE: " + str(np.mean(np.square(C_result - AB.classify(C_evidence)))))
print("BC MSE: " +str(np.mean(np.square(A_result - BC.classify(A_evidence)))))
print("AC MSE: " +str(np.mean(np.square(B_result - AC.classify(B_evidence)))))
print("Just A MSE: " +str(np.mean(np.square(BC_result - Just_A.classify(BC_evidence)))))
print("Just B MSE: " +str(np.mean(np.square(AC_result - Just_B.classify(AC_evidence)))))
print("Just C MSE: " +str(np.mean(np.square(AB_result - Just_C.classify(AB_evidence)))))



plt.plot(AB_loss, label="AB")
plt.plot(AC_loss, label="AC")
plt.plot(BC_loss, label="BC")
plt.plot(Just_A_loss, label="Just A")
plt.plot(Just_B_loss, label="Just B")
plt.plot(Just_C_loss, label="Just C")
plt.legend()
#label x and y axis
plt.xlabel('Epochs in thousands')
plt.ylabel('MSE')
plt.show()