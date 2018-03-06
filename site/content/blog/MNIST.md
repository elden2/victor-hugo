---
author: "Zachary S"
date: 2018-03-01
title: Creating a Perceptron Classifier for MNIST Dataset from Scratch in Python
keywords:
  - MNIST
  - Digit Classification
  - Python
  - ML
---

*Learn how to create a simple perceptron classifier is python to recognize handwritten digits in the MNIST dataset.*


![MNIST Dataset](/img/mnist.png)
The MNIST dataset is great little dataset to start exploring image recognition.  It's a series of 60,000 28 x 28 pixel images, each representing one of the digits between 0 and 9.

We're going to try to classify handwritten digits using a single layer perceptron classifier.  This is by no means the most accurate way of doing this, but it gives us a very nice jumping off point to explore more complex methods (most notably, deeper neural networks), which I'll explore later.

The data is easily found online, in a few forms.  I also have a modified version on github * INCERT LINK HERE * along with the code for this project.

```python
import numpy as np
import matplotlib.pyplot as plt
import random as rd
data = np.loadtxt('train_MNIST.csv', dtype = str, delimiter = ',')

y = np.asarray(data[1:, 0:1], dtype='float')
X = np.asarray(data[1:,1:], dtype='float')
```
We're only making one modification to the data itself for now, and that is to add a first column of ones to our data.  This will make our classifier easier to train later.

```python
def add_ones(x):
 	a, b = np.shape(x)
	c = np.ones((a , 1))   
	return np.hstack((c, x))
```

We're also going to want to split up our data into test and training sets.  Since this dataset is so large, this also allows us to make a much smaller batch to work with.  We can always increase this later.  (This will be the only time we're using Scikit Learn as I want to build this classifier myself)


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)
```

Since our data is in a row vector, we are going to need to resize it if we want to view it.



```python
def img(row, data):
	image = np.zeros((28,28))
	for i in range(0,28):
		for j in range(0,28):
			pix = 28*i+j
			image[i,j] = data[row, pix]
	plt.imshow(image, cmap = 'gray')
	plt.show()
	print data[row,0]
```

![MNIST Dataseta](/img/mnist1.png)

We're going to look at two parts of the perceptron algorithm: use and the training.

For a feed-forward binary perceptron, for any example, we take each of our dimensions, and multiply it by a predetermined weight, and add the results together.  If the totality is above a threshold, then we classify the example as positive; otherwise negative.

![MNIST Dataset](/img/per.png)


Lets start by creating a matrix of random weights.  Since the images as 28x28 pixels, each has 784 dimensions, and we are going to therefore need 784 weights.


```python
def create_weights(data):
	a, b = np.shape(data)
	weights = np.random.rand(b,1)
	return weights

weights = create_weights(X_train)

To write our feed forward prediction, we could write this as a loop, but it would be painfuly slow.  Numpy allows us to do this as a matrix multiplication, which will be far far faster.

```python
def predict(data_point, weights):
	b = np.dot(data_point, weights)
	a = b>0
	return a*1
````
This returns an either/or prediction for each row of our data.  As we can see, this is only good for determining  if something is or is not what we're looking for.  There are more efficient ways to do this, but we are going to cycle through our data and train it in an either or way for each individual digit.

To do this, we first need to write a function to change our data to 'is or is not' a number 


```python
def one_number(labels, number):
	return (labels == number)*1
```

Now comes actually training our neuron.
The perceptron algorithm is fairly straightforward:
1) For any individual  piece of data, determine if the weight correctly or incorrectly classify the data
2) If it is correct, do nothing and move on to the next piece of data
3) If the prediction is incorrect, rotate the plane of our classifier incrementally toward the correct answer.
4) Repeat for all data points, and until we have a classifier which is acceptable enough.

Alpha is our learning rate (the amount we are rotating our plane).  


```python
def update(weights, data_point, labels, alpha=.1):
	predicted = predict(data_point, weights)
	weight_temp = np.zeros(np.shape(weights))
	weight_temp[:,0] = alpha*(labels-predicted)*data_point
	return weight_temp+weights
```

```python
def train_perceptron(data, labels, weights, alpha = .1, iterations = 100):
	for j in range(0, iterations):
		for i in range(0, len(data)):
			weights = update(weights, data[i], labels[i], alpha)
	return weights
```

I've written a few functions  to test how well this program works before we move on.

```python
def test_perceptron_f(data, labels, weights):
    a,b = np.shape(data)
    predicted = predict(data, weights)
    correct = (predicted==labels)*1==1
    true_pos = np.sum((labels==1)*(correct))
    true_neg = np.sum((labels==0)*(correct))
    tp_p = true_pos/float(np.sum(labels))
    print np.sum(labels)
    tn_p = true_neg/float(a- np.sum(labels))
    return true_pos, true_neg, tp_p, tn_p, a
```
It's important here to look at what is working well and poorly in a bit more detail than just ''accuracy'.
This data is pretty evenly split between all 10 digits.  So when training on any one digit, we only have 10% of our data that are positive outcomes.  If my entire code was just "predict digit is 0", I'd have a 90% accuracy for 9 of our digits.

![MNIST Dataseta](/img/bar.png)

Looks like we're doing pretty well here for a start.
With just a few iterations and only a small subset of our data, we're classifying 73% of our positive results correctly, and 99% of our negative results correctly.  We'll talk about fine tuning these results later, but for now we have something that seems to be working.


Now lets create a function that will allow us to classify all our numbers.

```python
def all_numbers(data,labels):
	c,d = np.shape(data)
	w = create_weights(data)
	weights = []
	for i in range(0, 10):
		z = one_number(labels, i)
		a = train_perceptron(data, z, w, .1, 4)
		weights.append(a[:,0])
	return np.asarray(weights)
```
All this is doing is training 10 different neurons to give us 10 sets of weights that will classify each of the 10 digits.

```python
def one_all(data, weights):
	a = np.dot(data,np.transpose(weights))
	b = len(np.shape(data))
	if b == 1:
		return np.argmax(a)
	return np.argmax(a, axis=1)
```
Since our classifier is far from perfect, several different weights might predcit that a number belongs to that class.  For instance, a poorly drawn 3 might activate our perceptron from both 3 and 8.  In order to account for this, the last function will classify our numbers based on the weights that return the highest output.

Since at this point our data is fairly symmetrical (it's almost perfectly divided into 10% for each number), we can just look at overall accuracy.

```python
def test_all(data, labels, weights):
	a, b = np.shape(labels)
	predicted = one_all(data, weights)
	correct = predicted == labels[:,0]
	accuracy = np.sum(correct)/float(a)
	return accuracy
```
