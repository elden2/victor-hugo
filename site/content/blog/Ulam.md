---
author: "Zachary S"
date: 2018-02-21
title: Making a Ulam Spiral
keywords:
-Ulam Spiral
-Python
-Prime Spiral
description: Write a python code to plot Ulam Spirals for any dimension.
---

*Ulam Spirals*
Creating a program to plot Ulam Spirals for any dimension.

![Spiral](/img/Spiral1.jpg)

I remember coming across [Ulam Spirals](https://en.wikipedia.org/wiki/Ulam_spiral) (also known as Prime Spirals) a few years back, and though they were fascinating.  

Stanislaw Ulam (of the Teller-Ulam Hydrogen Bomb) discovered these while doodling in a conference (and I find myself doodling the same spiral when I'm board in meetings as well.)

It's quite simple to understand the process, almost impossible to understand the why, and a fun little project to implement in python; all in all a perfect afternoon project.


Take a series of integers and arrange them in an square lattice spiraling outward.  Color in all the prime (or not prime, whichever you would like) numbers, and you'll find that the primes lie on long diagonals.


![Integrer Spiral from Wiki](/img/Ulam.jpg)


Doing this up to 100 is easy enough by hand, but a little python code, and we can see this pattern holds true for quite a while.


```python
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
```

Few libraries.  We'll use numpy arrays to create a spiral, and pyplot to plot it.

```python
class Ulam(object):
	def __init__(self, size = 99):
		self.spiral = np.zeros((0,0))
		self.size = 2*size-1
```

Notice above the size is doubled from what you enter.  The way I wrote the spiral, it was easier to do as an odd number - this ensures that you don't have even number sides.

Now we're going to need a function to find all the primes.  We could test each one number, but since we're going to be doing thousands of tests, it is a lot faster to just write a sieve to find all primes up to our limit.


```python

	def prime(self, x):
		primes = [True]*x
		primes[0] = False
		primes[1] = False
		for i in range(2,int(x**.5+int(x**.5)+2)):
			if primes[i]:
				for j in range(i, x//2+1):
					if i*j<x:
						primes[i*j] = False
		return primes

```

Now it's time to actually make our curve

```python

def make_curve(self):

		x  = self.size/2
		y  = self.size/2
		num = 1
		pass_num = 1
		spiral = np.zeros((self.size, self.size))

		spiral[x, y] = num
		num += 1

		finished = False

		while not finished:

			#down
			for i in range(2*pass_num - 1):
				y+= 1
				spiral[x,y] = num
				num+= 1
				if num>self.size*self.size-5:
					finished = True
					break
			
			#right
			for i in range(2*pass_num - 1):
				x+= 1
				spiral[x,y] = num
				num+= 1
				if num>self.size*self.size-5:
					finished = True
					break

			#up
			for i in range(2*pass_num):
				y -= 1
				spiral[x,y] = num
				num+= 1
				if num>self.size*self.size-5:
					finished = True
					break
			#left
			for i in range(2*pass_num):
				x -= 1
				spiral[x,y] = num
				num+= 1
				if num>self.size*self.size-5:
					finished = True
					break
			pass_num += 1


		return spiral
```

The above isn't exactly the fastest way of doing this (4 loops in a weird succession), but it worked nicely.

Lets replace all our numbers with a binary prime or not prime designation

```python

	def replace_curve(self, array):
		self.size = np.shape(array)
		primes = self.prime(self.size[0]*self.size[1])
		for i in range(0,self.size[0]):
			for j in range(0, self.size[1]):
				if primes[int(array[i,j])]:
					array[i,j] = 1
				else:
					array[i,j] = 0
		return array
```


And lets see if it works

```python

	def show(self):
		width = self.size
		height = self.size
		self.spiral = self.make_curve()
		self.spiral = self.replace_curve(self.spiral)	
		plt.imshow(self.spiral, cmap = cm.Greys_r)
		plt.show()

a = Ulam(size=200)
a.show()
```

![Spiral](/img/spiral2.jpg)
![Spiral](/img/Spiral5.jpg)
