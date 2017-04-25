import numpy as np

#=====================Activation Functions=======================
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh(x):
    return np.tanh(x)
#================================================================

#===================Derivative of Activation Function============
def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh_prime(x):
    return 1.0 - x**2
#================================================================


class NeuralNetwork:

    def __init__(self, layers, activation='sigmoid'): # Default activation function set to Sigmoid
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self.weights = []

        # layers = [2,2,1] - number of uits in input layer, number of units in hidden layer 1, number of units in output layer
        # Another example : layers = [2, 2, 3, 1] - number of uits in input layer, number of units in hidden layer 1, number of units in hidden layer 2, number of units in output layer
        # range of weight values (-1,1)
        # input and hidden layers for layers [2, 2, 1] - random((2+1, 2+1)) : 3 x 3, 1 is added for the bias units in each layer
        # Assigning random initial weights from input layer to hidden layer 1, hidden layer 1 to hidden layer 2 till "last" hidden layer 
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            #r = np.zeros((layers[i-1] + 1, layers[i] + 1))-1
            self.weights.append(r)
        # output layer for layers [2, 2, 1] - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1 # Assigning weights for the "last" hidden layer to the output unit
        #r = np.zeros((layers[i] + 1, layers[i+1]))
        self.weights.append(r)
        

    def fit(self, X, y, learning_rate=0.2, epochs=100000): # Can play with learning_rate values and epochs values as well
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        #print X.shape[0],X.shape[1]
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            
            a = [X[i]] # Taking a random input from the list of input given by generating a random number i 

            for l in range(len(self.weights)):
##                print a[l]
##                a1 = map(lambda x: float(x),a[l])
##                print a1
##                print self.weights[l]
##                b1 = map(lambda x: float(x),self.weights[l])
##                dot_value = np.dot(a1,b1)
                #print a[l].shape[0], self.weights[l].shape[0]
                a1 = map(lambda x: float(x),a[l])
                dot_value = np.dot(a1, self.weights[l])
                activation = self.activation(dot_value)
                a.append(activation)
            # output layer
            
            error = y[i] - a[-1] # a[-1] gives the last value of a which is essentially the output
            deltas = [error * self.activation_prime(a[-1])] 

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0: print 'epochs:', k

    def predict(self, x):
        #print x
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)      
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

class Data:

    def __init__(self): # Default activation function set to Sigmoid
        print 'Reading Data'

    def readData(self, fileName, trainTest): # Can play with learning_rate values and epochs values as well
        if trainTest == 'train':
            f = open(fileName, 'r')
            i = 0
            resultX = []
            resultY = []
            for line in f:
                if i == 0:
                    i = i + 1
                else:
                    words = line.strip().split(',')
                    list = []
                    listy = []
                    listy.append(float(words[0]))
                    resultY.append(listy)
                    for i in range(1, len(words)):
                        list.append(float(words[i]))
                    resultX.append(list)
            answer = []
            answer.append(resultX)
            answer.append(resultY)
            return answer

        if trainTest == 'test':
            f = open(fileName, 'r')
            i = 0
            resultX = []
            for line in f:
                if i == 0:
                    i = i + 1
                else:
                    words = line.strip().split(',')
                    list = []
                    for i in range(0, len(words)):
                        list.append(float(words[i]))
                    resultX.append(list)
            
##            answer = []
##            answer.append(resultX)
##            answer.append(resultY)
            return resultX
            f.close()

    def accuracy(self, x, y):
        f1 = open(x,'r')
        f2 = open(y,'r')
        l1 = []
        l2 = []
        i = 0
        for l in f1:
            if i==0:
                i = i + 1
            else:
                #print l1.strip().split(',')[1]
                pre = float(l.strip().split(',')[1])
                l1.append(pre)
        i = 0
        for l in f2:
            if i == 0:
                i = i+1
            else:
                
                pre = float(l.strip().split(',')[1])
                l2.append(pre)
        c = 0
        for i in range(len(l1)):
            if l1[i] < 0.5 and l2[i] < 0.5:
                c = c + 1
            if l1[i] > 0.5 and l2[i] > 0.5:
                c = c + 1
        print c , len(l1)
        print float((c*1.0)/len(l1))
        
if __name__ == '__main__':

    data = Data()
    (Train_X, Train_Y) = data.readData('train.csv','train')
    #print len(Train_X)
    nn = NeuralNetwork([22, 15, 10, 10,10, 10,  1])
    # you can play with the hidden layers here for example - nn = NeuralNetwork([3,4,3,4, 3]) three hidden layers with 4,3,4 units in each
    #list_input = 
    #[[0,0],[0,1],[1,0],[1,1]]
    #[[0,1],[1,0],[1,1],[0,0]]
    X = np.array(Train_X)
    y = np.array(Train_Y)
    #print X[0]
        
    nn.fit(X, y)

    # Testing the values on the test set. Here you can take any test value 
##    for e in X:
##        print(e,nn.predict(e))
    f = open('prediction.csv','w')
    line = 'Id,Choice'
    f.write(line)
    f.write('\n')
    Test_X = data.readData('test.csv','test')
    i = 1
    for list in Test_X:
        line = ''
        predict = nn.predict(list)
        for pre in predict:
            line = ''+str(i)+','+str(pre)
        f.write(line)
        i = i + 1
        f.write('\n')
    f.close()
    data.accuracy('prediction.csv','sample_predictions.csv')
    
##    print([1,1,1],nn.predict([1,1,1]))
##    print([0,0,0],nn.predict([0,0,0]))
    
