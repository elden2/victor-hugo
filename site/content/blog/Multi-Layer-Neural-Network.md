---
author: "Zachary S"
date: 2018-03-19
title: Creating a Multi-Layer Neural Network from Scratch
keywords:
  - Neural Network
  - Back-Propagation
  - Python
  - ML
---

*Creating a very simple neural network for binary classification*

Every time I try to build an ML algorithm from scratch, it gives me a far greater understanding of what is actually going on.  This is a pretty straightforward neural network with one hidden layer trained with backpropagation.  Although before I tried to build this, I generally understood the theory of how backpropagation worked, fumbling through all the trouble aligning matrices correctly, adding bias units, and implementing the idea really solidified that understanding.

The attached code, which you can find at [github](https://github.com/zswarth/MNIST-Classification/blob/master/MultiLayerFinal.py) will allow you to train a neural network with 1 hidden layer (you can choose the number of neurons in this layer).  This [Jupyter Script](https://github.com/zswarth/MNIST-Classification/blob/master/Playing_with_Hidden_Layers.ipynb) may also be useful.

There is a lot more that I want to eventually do with this little example code - that will be the basis for my next project.  This includes implementing batch gradient decent, allowing this to automatically work with an arbitrary number of hidden layers, trying different activation functions, and, most importantly, allowing for multi-class classification.


First I created a simple dataset to play with.

![sin data](/img/sin.png)

This is clearly a non linear boundary, so my last project, a single perceptron, won't do much good.


Lets just see what it would do anyway:

```python
from perceptron_object import Perceptron

Per = Perceptron(data[:,0:2], data[:,2:3], alpha = .001, iteration = 200, test_percentage = .1, binary = True)
```

```python
def plot_weights(w):
    x = np.arange(-1,11,.01)
    y = (-w[0]-w[1]*x)/w[2]
    plt.plot(x,y)
plot_weights(Per.weights[1,:])
    
for i in range(0,500):
    if data[i,2] == 1:
        plt.scatter(data[i,0], data[i,1], color = 'blue')
    else:
        plt.scatter(data[i,0], data[i,1], color = 'red')
plt.show()
```
![sin data](/img/line.png)

I won't go as far as to say this is useless, but it certainly isn't great.

We can also try to fit a quadratic curve through the data.  To do this, I created new data with the following dimensions: x, y, x^2, y^2, and xy

```python
def quadratic(data):
    return np.transpose(np.vstack((data[:,0], data[:, 1], data[:,0]*data[:,1], data[:,0]**2, data[:,1]**2)))
```

What happens:

```python
new_data = quadratic(data[:, 0:2])
Per_q = Perceptron(new_data, data[:,2:3], alpha = 1.05, iteration = 3000, test_percentage = .2, binary = True)

def quad(a,b,c):
    dis = ((abs(b**2-4*a*c))**.5)
    return (-b+dis)/float(2*a)
    

def y(x, w):
    a = w[5]
    b = x*w[3]+w[2]
    c = w[0]+w[1]*x+w[4]*x*x
    return quad(a,b,c)


xs = np.arange(1,10,.1)
ys = y(xs, Per_q.binary_weigths)


for i in range(0,500):
	if data[i,2] == 1:
		plt.scatter(data[i,0], data[i,1], color = 'blue')
	else:
		plt.scatter(data[i,0], data[i,1], color = 'red')

plt.plot(xs, ys)
plt.show()
```

![Quadratic](/img/quad.png)

Certainly not great, but a lot better.  If I spent some time playing with the parameters, I could spruce this up a bit, but it'll never be perfect (as the boundary isn't quadratic)


A neural network should do better.

```python
import numpy as np
import random
```

These are the only libraries I'm going to use.

```python
class NN(object):

    def __init__ (self, data, labels, hidden_layer_size = 3):
            self.data = data
            self.labels = labels
            self.size = hidden_layer_size
            self.wh, self.wo, self.bh, self.bo = self.create_weights()

```
I need to research the specifics of how to chose an activation function.  I'm using a sigmoid one here for no other reason than it is what I learned in Andrew Ng's Coursera course.  I'm going to come back and play with this later. 


```python
	def sigmoid(self, x):
		return 1/(1+np.exp(-x))

	def sig_prime(self, sig):
		return sig*(1-sig)
		#just to be clear, instead of computing sigmoid 3 times, i can just store the value, and then use it again for the derivative
```


```python
	def create_weights(self):
		a,b = np.shape(self.data)
		wh = np.random.randn(b, self.size)
		wo = np.random.randn(self.size,1)
		bh = np.random.randn(1, self.size)
		bo = np.random.randn(1, 1)
		return wh, wo, bh, bo
```

I'll come back to the weights also in a bit.  Right now, you can't adjust the number of hidden layers, just the size of the hidden layer.  But I want to change this to add any arbitrary number of layers.



```python
    def feed_foward(self, data):
        output1 = self.sigmoid(np.dot(data, self.wh)+self.bh)
        output2 = self.sigmoid(np.dot(output1, self.wo)+self.bo)

        return output1, output2


    def back_prop(self, output1, output2, alpha):
        error = self.labels - output2
        delta_output = error*self.sig_prime(output2)
        error_hidden = delta_output.dot(self.wo.T)
        delta_hidden = error_hidden*self.sig_prime(output1)
        self.wo +=  alpha*output1.T.dot(delta_output)
        self.wh += alpha*self.data.T.dot(delta_hidden)
        self.bo += alpha*np.sum(delta_output, axis = 0)
        self.bh += alpha*np.sum(delta_hidden, axis = 0)
        total_error = np.sum(abs(error))
        return total_error
 ```
 A full training cycle involves  one feed forward step, followed by propagating the errors created in this feed forward prediction backwards though the network.


```python
    def train(self, data = None, alpha = 100, iterations = 120000):
        data = self.data
        error = []
        for i in range(0, iterations):
            output1, output2 = self.feed_foward(data)
            total_error = self.back_prop(output1, output2, alpha)
            if i%1000 == 0:
                if i%10000:
                    print total_error, i//1000, 'Thousand Iterations'
                error.append(total_error)
        return total_error
 ```

```python
    def predict(self, x):
        out1, out2 = self.feed_foward(x)
        results = []
        for i in out2:
            if i >=.5:
                results.append(1)
            else:
                results.append(0)
        return np.asarray([results]).T

    def test(self, x, labels):
        results =  self.predict(x)
        correct = (results==self.labels)*1
        return correct, results
```


And lets see what happens:

```python

a = NN(x, y, hidden_layer_size = 2)

a.train(alpha = .06, iterations = 3000)

for i in range(0,20):
    it = i/2.0
    for j in range(0,20):
        jt = j/2.0
        b = a.predict([it,jt])
        if b[0]>=.5:
            plt.scatter(it, jt, color = 'blue')
        else: plt.scatter(it, jt, color = 'red')
plt.show()
```
![2 Layers](/img/2l.png)

Two neurons just aren't enough to make this work well.

![Layers](/img/l1l2.png)

We can see above the output of each of the two neurons and how they are contributing to the classification, but the boundary is too complex for 2 to do well with.


6 Neurons:


```python
new = NN(x, y, hidden_layer_size = 6)
errors = new.train(x, alpha = .001, iterations = 30000)

for i in range(0,20):
    it = i/2.0
    for j in range(0,20):
        jt = j/2.0
        b = new.predict([it,jt])
        if b[0]>=.5:
            plt.scatter(it, jt, color = 'blue')
        else: plt.scatter(it, jt, color = 'red')
plt.show()
```

![6 hidden units](/img/6n.png)

This is starting to look promising

```
for i in range(1,7):
    plt.subplot(2,3,i)
    for k in range(0,20):
        it = k/2.0
        for j in range(0,20):
            jt = j/2.0
            r = new.feed_foward([it, jt])
            if r[0][0][i-1]>=.5:
                plt.scatter(it, jt, color = 'blue')
            else: plt.scatter(it, jt, color = 'red')
plt.show()
```

![Each Neuron](/img/6ns.png).  We can see what each neuron is doing.

If I divide up my data into half training and half testing, 6 neurons is enough to pretty close to perfectly classify my data (granted, this data has no noise - so overfitting is not going to be a problem).


This is a fun little toy to play with, and a great exercise  to help me understand how neural networks work.  Next I need to add the ability to work with multi-class classification, multiple hidden layers, and batch gradient decent, and then try and run it on the MNIST data set.  Of course using a library like tensorflow would be far more efficient for any actual project, but where's the fun in that?


















