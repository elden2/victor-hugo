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

Every time I try to build an ML algorithm from scratch, it gives me a far greater understanding of what is actually going on.  This is a pretty straightforward  neural network with one hidden layer trained with backpropagation.  Although before I tried to build this, I generally understood the theory of how backpropagation worked, fumbling through all the trouble aligning matrices  correctly, adding bias units, and implementing the idea really solidified that understanding.

The attached code, which you can find at [github](MNIST-Classification/MultiLayerFinal.py) will allow you to train a neural network with 1 hidden layer with a self selecting number of neurons on a binary labeled data set.  This [Jupyter Script](MNIST-Classification/Playing_with_Hidden_Layers.ipynb) may also be useful.

There is a lot more that can be done with this little example code, which will be the basis for my next project.  That includes implementing batch gradient decent, making an arbitrary number of hidden layers, trying different activation functions, and allowing for multi-class classification.



First I created a simple dataset to play with.

![sin data](/img/sin.png)

This is clearly a non linear boundary, so my last project, a single perceptron, won't do much good.



```python
from MultiLayerFinal import NN
import numpy as np
import random
from scipy.special import expit
from perceptron_object import Perceptron
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = np.loadtxt('sin.csv', delimiter = ',')

Per = Perceptron(data[:,0:2], data[:,2:3], alpha = .001, iteration = 200, test_percentage = .1, binary = True)
```

Lets just see what it would do anyway:

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

Certainly not great, but a lot better.  If I spent some time playing with the parameters, I could spruce this up a bit, but it'll never be perfect.


It's time to create my neural network to see if it does better.



```python
import numpy as np
import random
```

These are the only libraries I'm going to use.

```python
class NN(object):

	def __init__ (self, data, labels, hidden_layer_size = 3, alpha = .1, batch = 20):
		self.data = data
		self.alpha = alpha
		#self.batch = batch
		#I'm going to keep this in here to implement batch gradient decent later
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
 	def feed_foward(self, data, wh, wo, bh, bo):

 		output1 = self.sigmoid(np.dot(data, wh)+bh)
 		output2 = self.sigmoid(np.dot(output1, wo)+bo)

 		return output1, output2



 	def back_prop(self, data, labels, output1, output2, wh, wo, bh, bo, alpha = 1):
 		error = labels - output2
 		delta_output = error*self.sig_prime(output2)
 		error_hidden = delta_output.dot(wo.T)
 		delta_hidden = error_hidden*self.sig_prime(output1)
 		wo +=  alpha*output1.T.dot(delta_output)
 		wh += alpha*data.T.dot(delta_hidden)
 		bo += alpha*np.sum(delta_output, axis = 0)
 		bh += alpha*np.sum(delta_hidden, axis = 0)
 		total_error = np.sum(abs(error))
 		return wo, wh, total_error
 ```
 A full training cycle involves  one feed foward step, followed by propagating the errors created in this feed forward prediction backwards though the network.


```python
 	def train(self, data, labels, wh, wo, bh, bo, alpha = 100, iter = 120000):
 		error = []
 		for i in range(0, iter):
 			output1, output2 = self.feed_foward(data,wh, wo, bh, bo)
 			wo, wh, total_error = self.back_prop(data, labels, output1, output2, wh, wo, bh, bo, alpha = alpha)
 			if i%1000 == 0:
 				if i%10000:
 					print total_error, i//1000, 'Thousand Iterations'
 				error.append(total_error)
 		return wh, wo, bh, bo, total_error
 ```

```python
 	def test(self, data, labels, wh, wo, bh, bo):
 		out1, out2 = self.feed_foward(data,wh, wo, bh, bo)
 		def pr(x):
 			output = []
 			for i in x:
 				if i >=.5:
 					output.append(1)
 				else:
 					output.append(0)
 			return np.asarray([output]).T
 		correct = pr(out2)==labels
		return correct*1, pr(out2), out2
```


And lets see what happens:

```python

a = NN(x, y, hidden_layer_size = 2)
c,d = a.feed_foward(x, a.wh, a.wo, a.bh, a.bo)


# In[40]:


w1,w2, bh, bo, errors = a.train(x, y,a.wh, a.wo, a.bh, a.bo, alpha = .06, iter = 30000)


# In[41]:


for i in range(0,20):
    it = i/2.0
    itr,b,c = a.test(x, y, a.wh, a.wo, a.bh, a.bo)
    for j in range(0,20):
        jt = j/2.0
        b = a.feed_foward([it, jt], a.wh, a.wo, a.bh, a.bo)
        if b[1]>=.5:
            plt.scatter(it, jt, color = 'blue')
        else: plt.scatter(it, jt, color = 'red')
plt.show()
```
![2 Layers](/img/2l.png)

Two neurons just isn't enough to make this work well.

![Layers](/img/l1l2.png)

We can see above the output of each of the two neurons and how they are contributing to the classification, but the boundary is too complex for 2 to do well with.


6 Neurons:



```python
new = NN(x, y, hidden_layer_size = 6)
c,d = new.feed_foward(x, new.wh, new.wo, new.bh, new.bo)


# In[77]:


w1,w2, bh, bo, errors = new.train(x, y,new.wh, new.wo, new.bh, new.bo, alpha = .0001, iter = 30000)


# In[78]:


for i in range(0,20):
    it = i/2.0
    itr,b,c = a.test(x, y, new.wh, new.wo, new.bh, new.bo)
    for j in range(0,20):
        jt = j/2.0
        b = new.feed_foward([it, jt], new.wh, new.wo, new.bh, new.bo)
        if b[1]>=.5:
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
        itr,q,c = new.test(x, y, new.wh, new.wo, new.bh, new.bo)
        for j in range(0,20):
            jt = j/2.0
            r = new.feed_foward([it, jt], new.wh, new.wo, new.bh, new.bo)
            if r[0][0][i-1]>=.5:
                plt.scatter(it, jt, color = 'blue')
            else: plt.scatter(it, jt, color = 'red')
plt.show()
```

![Each Neuron](/img/6ns.png).  We can see what each neuron is doing.

If I divide up my data into half training and half testing, 6 neurons is enough to pretty close to perfectly classify my data.  With 10 neurons, it doesn't take too much work to get a prefect accuracy with this data.


This is a fun little toy to play with, and a great exercise  to help me understand how neural networks work.  Next I need to add the ability to work with multi-class classification, multiple  hidden layers, and batch gradient decent, and then try and run it on the MNIST data set.  Of course using a library like tensorflow would be far more efficient for any actual project, but where's the fun in that?


















