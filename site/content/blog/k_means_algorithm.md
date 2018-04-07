---
author: "Zachary S"
date: 2018-02-28
title: Writing the K-Means Algorithm from Scratch
keywords:
  - K-Means
  - Clustering
  - Python
  - Unsupervised Learning
---

*How to write a k-means clustering algorithm in python*

The full code can be found at [github](https://github.com/zswarth/SmallProjects/blob/master/new_k-means.py)


![Iris Data](/img/4_plot.png)


The k-means clustering algorithm is a method for grouping data into clusters, or sections, of similar data.

For the below example, I'm using the famous [Iris Data Set](https://en.wikipedia.org/wiki/Iris_flower_data_set).

```python
from numpy import genfromtxt
import matplotlib.pyplot as plt
from random import randrange
import numpy as np

iris = np.genfromtxt('iris.csv', dtype=str, delimiter = ',')
iris_data = np.asarray(iris[1:,0:4], dtype=float)
iris_labels = iris[1:,4]
```
The iris data set is a series of 3 iris flowers (Setosa, Versicolor, and Virginica), with data given for Sepal Length, Sepal Width, Petal Length, and Petal Width.

Clustering is notoriously difficult with the 3rd flower, so lets only look at Setosa and Versicolor in this project.


![Iris Data](/img/Iris.png)

This is only the first two dimensions, but we can see that there are two groups of data.  We have labels, but since k-means is an unsupervised algorithm, we're going to ignore those labels until the end to see how well clustering works.


Firstly, we need to write a function to find the Euclidian distance between any two points.

```python
def distance(point1, point2):
	dis = point1-point2
	return (np.dot(dis, dis))**.5
```

Next, create an arbitrary first set of centroids.

More efficient k-means clustering use more complex methods to initiate the centroid, but we're going to just use a random number generator.


```python
def create_centroids(data, num_clusters = 2):
	centroids = []
	a,b = np.shape(data)
	for j in range(0, num_clusters):
		temp = []
		for i in range(0, b):
			maxx = int(max(data[:,i]))
			minx = int(min(data[:,i]))
			temp.append(randrange(minx, maxx))
		centroids.append(temp)
	return np.asarray(centroids)

centroids = create_centroids(iris_data[0:100], num_clusters=2)
```

Each line of our data is going to need to be assigned a label representing the nearest centroid.

```python

def label(data, centroids):
	m, n = np.shape(data)
	lab = []
	for i in range(0, m):
		dis = []
		x, y = np.shape(centroids)
		for j in range(0,x):
			dis.append(distance(centroids[j,:], data[i, :]))
		lab.append(np.argmin(dis))
	return np.asarray(lab)


labels = label(iris_data[0:100], centroids)
```

![Iris Cluster](/img/Iris_cluster.png)

We can see above that our random initialization of our points put our centroids pretty close to the center of our data.

Now that we have labeled  our data, we need to update the coordinates of our centroids.  We do this by averaging all the points with label 1, and setting our first centroid to that average, and then doing the same with each subsequent label.


```python
def update_centroids(data, labels, centroids):
	m,n = np.shape(centroids)
	temp_centroid = np.zeros((m,n))
	count = np.zeros((m,n))
	for j in range(0, len(labels)):
	 	for k in range(0, n):
			temp_centroid[labels[j],k]+= data[j,k]
			count[labels[j],k]+= 1
	for i in range(0,m):
		for j in range(0,n):
			if count[i,j] == 0:
				count[i,j] += 1
	return temp_centroid/count


def final_algorithm(data, n_clusters = 2, iter = 3):
	centroids = create_centroids(data, num_clusters = n_clusters)	
	labels = label(data, centroids)
	f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, sharex='col', sharey='row')
	graph_axis(data, labels, centroids, ax1)
	plots = [ax2, ax3, ax4]
	for i in plots:
		centroids = update_centroids(data, labels, centroids)
		labels = label(data, centroids)
		graph_axis(data, labels, centroids, i)
	plt.show()
	for i in range(0,iter-3):
		centroids = update_centroids(data, labels, centroids)
		labels = label(data, centroids)
	return centroids, labels
```	


Our centroids have moved closer to the centers of our data.

![Final Result](/img/perfect.png)

This is exactly what the actual labeled data should look like.  So our clustering yielded  pretty good results.


Here is an example of the process in full:



![Iris Data](/img/4_plot.png)


