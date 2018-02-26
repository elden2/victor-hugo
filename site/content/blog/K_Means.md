---
author: "Zachary Swarth"
date: 2018-02-25
title: Reducing Number of Colors in Photo using K-Means
keywords:
  - K-Means
  - Coloring Photo
  - Sckit learn
---

*There are quite a few million color possiblities in a normal RGB PNG file.  Use K-Mears to pear it down to as many as you'd like for some intersting results.*


This was just a test to see what would come out; my attempt was to make something that looked like the [Obama Hope](https://en.wikipedia.org/wiki/Barack_Obama_%22Hope%22_poster) poster by Shepard Fairey.

The K-Means algorithm is a fun little tool which seperates data into a number of piles based on the Euclidean distance of the data to a point


```python

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import sklearn.cluster as sk

img=mpimg.imread('untitled.png')
```

We'll be using Scikit Learn's library instead of my own for this as it's quite a bit faster.

```python
clf = sk.KMeans(n_clusters = 6)
a,b,c = np.shape(img)
img2 = img.reshape(a*b, 3)

clf.fit(img2)
```

Flatten out the data so each row is a data point and fit the data.

We now have a model which will be able to take our photo and make it into just 6 colors.

Lets create a new array for our color reduced photo.

```python
img3 = np.zeros((a,b, 3))
center = clf.cluster_centers_


for i in range(0,a):
    for j in range(0,b):
        img3[i,j] = center[clf.predict([img[i,j]])]
```
The above loop is painfully slow - not the best implementation.

And lets see what we got:

```python
imgplot = plt.imshow(img3)
```

Our old and new photo:

![K-Means Image](/img/KMeans.png)

In 4 colors instead of 6:


![K-Means Image](/img/KM2.png)

