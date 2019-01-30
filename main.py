import numpy as np

class Perceptron(object):


    def __init__(self,bias=0,epochs=100,learning_rate=.01):
        self.bias=bias
        self.learning_rate=learning_rate
        self.epochs=epochs 


    def train(self,X_train,y_train):
        self.w=np.zeros(X_train.shape[1])

        for _ in range(self.epochs):
            for xis,yi in zip(X_train,y_train):
                y= self.predict(xis)
                error=yi-y
                self.bias+=self.learning_rate * error
                self.w+=self.learning_rate * error * xis

        # print(self.w,self.bias)

        print(self.w,self.bias)
        # return w_sum
            

    def predict(self,_x):
        w_sum=np.dot(_x,self.w) + self.bias
        return 1 if w_sum > 0 else 0
    

# Inputs
X_train=np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
])

y=np.array([0,0,0,1])



p=Perceptron()
p.train(X_train,y)
# print(p.predict([1,1]))
