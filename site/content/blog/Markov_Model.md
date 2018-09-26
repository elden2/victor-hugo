---
author: "Zachary S"
date: 2018-02-20
title: Marking A Markov Chain Speech Predictor
---


*A program to write in the style of Moby Dick*

 

The idea of a Markov Chain is pretty simple.  Imagine you were to hear the beginning of the sentence "The United States of....." and you needed to fill in the rest.  Chances are, at least if you are from San Francisco in 2018, you would assume "America" is the next word (as opposed to "The United States of Brazil," (which was Brazil's official name half a century ago).  You guessed "America" simply because statistically, this is by far the most common next word spoken after "United States of".

 

My little speech predictor does exactly this.  Based on a collection of texts (I started with Moby Dick), I analyzed every 2 and 3 word phrase, and attempted to determine the most common next word you would find.  Once you have this data, you can start any sentence, and then just have the program fill in words that would make statistical sense until you have a paragraph. 

 

You can find the full code at https://github.com/elden2/TextPredict

 

 

Lets start with a few libraries 



 
```python

import random
import sys
import os
from itertools import groupby

```


My predictor object as a few attributes.

 

depth: how many words back we are going to use to predict.  The code currently doesn't support this being changed, but is a placeholder for updating it later.

 

data: text that our model will learn from

 

three dictionaries: a collection of three word phrases, two word phrases, and single words.

 

Random: it's easy to get stuck in repeating loops.  This tells us how often we might want to throw some noise into our model.

 

```python


class Predictor(object):
	def __init__(self, data = 'as long string', depth = 3, random = 15):
		self.depth = depth
		self.data = data.split()

		self.dictionary_tripple= {}
		self.dictionary_doubles = {}
		self.dictionary_single = {}

		self.tripple = []
		self.target = []
		self.get_trip()
		self.pair_dictionary_tripples()


		self.doubles = []
		self.doubles_target = []
		self.get_doubles()
		self.pair_dictionary_doubles()



		self.single_pair()
		self.random = random
		self.error_count = [0,0,0,0] ##Number of tripples, doubles, singles, none used

```



Grabbing all our keys:

 
```python

	def get_trip(self):
		a = len(self.data)
		for i in range(0, a-self.depth):
			self.tripple.append(self.data[i]+' '+self.data[i+1]+' ' + self.data[i+2])
			self.target.append(self.data[i+3])

	def get_doubles(self):
		a = len(self.data)
		for i in range(0,a-2):
			self.doubles.append(self.data[i]+' '+self.data[i+1])
			self.doubles_target.append(self.data[i+2])

	def pair_dictionary_tripples(self):
		a = len(self.target)
		for i in range(0,a):
			if self.tripple[i] in self.dictionary_tripple:
				self.dictionary_tripple[self.tripple[i]].append(self.target[i])
			else:
				self.dictionary_tripple[self.tripple[i]] = [self.target[i]]

	def pair_dictionary_doubles(self):
		a = len(self.doubles_target)
		for i in range(0,a):
			if self.doubles[i] in self.dictionary_doubles:
				self.dictionary_doubles[self.doubles[i]].append(self.doubles_target[i])
			else:
				self.dictionary_doubles[self.doubles[i]] = [self.doubles_target[i]]


	def single_pair(self):
		a = len(self.data)
		for i in range(0, a-1):
			if self.data[i] in self.dictionary_single:
				self.dictionary_single[self.data[i]].append(self.data[i+1])
			else:
				self.dictionary_single[self.data[i]] = [self.data[i+1]]
```



 

The above functions take our text and chops it up into appropriate length keys.  If a key is not in the dictionary, it adds a new entry with the word that follows as a single element list as the value.  If the key already exits, the next word is added in the value's list.

 

For Moby dick, we get a dictionary of triples that is slightly under 200,000 entries long.  An example of a dictionary entry:

MobyDick["the Sperm Whale"] results in:

 

['fishermen.', 'fishery', 'fishery', 'not', 'fishery,', 'was', 'when', 'Fishery,', 'HAS', 'when,', 'drawings', 'designated', 'is', 'presents,', 'in', 'is', 'and', 'and', 'are', 'only', 'were', 'stove', 'embraces', 'is', 'ever', 'has', 'be', 'has', 'is', 'is', 'will', 'only', 'has', 'is', 'into', 'presents', 'to', 'thus']

 

 Since 'is' comes up 5 times, it is the most common word.  We now want our program to predict 'is' if we were to enter 'the Sperm Whale' into our program,

 

Input: "the Sperm Whale"

output "is"

New Input "Sperm Whale is"

 

Yielding the start of a paragraph 'the Sperm Whale is...."

 

 ```python

 	def common_word(self, L, noise = False):
		if noise:
			return random.choice(L)
		if len(L) == 1 or len(L) == 2:
			return L[0]
		else:
			return max(groupby(sorted(L)), key=lambda(x, v):(len(list(v)),-L.index(x)))[0]



	def predict_word(self, key = 'three word form', noise = False):
		a = key.split()
		temp3 = a[0]+ ' ' + a[1] + ' ' + a[2] 
		temp2 = a[1]+ ' ' + a[2]
		temp1 = a[2]
		if temp3 in self.dictionary_tripple.keys():
			self.error_count[0] += 1
			return self.common_word(self.dictionary_tripple[temp3], noise = noise)
		if temp2 in self.dictionary_doubles.keys():
			self.error_count[1] += 1
			return self.common_word(self.dictionary_doubles[temp2], noise = noise)
		if temp1 in self.dictionary_single.keys():
			self.error_count[2] += 1
			return self.common_word(self.dictionary_single[a[2]], noise = noise)
		self.error_count[3] += 1
		return self.dictionary_tripple[random.choice(self.dictionary_tripple.keys())][0]
		
			
```

 

The noise = False section will be used at the end.  It allows us to bypass the most common word, and instead throw in a less common word that exists in the list.  Without a bit of noise thrown in, the program can get caught in infinite loops.
 

Putting it all together:

```python
	def get_promt(self, sentence, length = 3):
		a = sentence.split()
		prompt = ''
		for i in a[-(length):]:
			prompt += i + ' '
		return prompt[:-1]


	def predict_sentence(self, initial = 'who knows what', chain_length = 3, parg_length = 1000):
		sentence = initial
		for i in range(0, parg_length):
			if i%self.random == 0 and i!= 0:
				prompt = self.get_promt(sentence, length= chain_length)
				sentence += ' ' + self.predict_word(prompt, noise = True)
			else:
				prompt = self.get_promt(sentence, length= chain_length)
				sentence += ' ' + self.predict_word(prompt, noise = False)
		return sentence
```

```python


with open('./Texts/moby.txt', 'r') as temp:
	moby = temp.read().replace('\n', '')


Moby_dick = Predictor(moby, random = 10)

print Moby_dick.predict_sentence('a Sperm Whale',parg_length=100)

```



 

 

Lets see if it works:

 



 

"a Sperm Whale is not a little behind the whale, the body of a whale in the sea, and they are so shut up, belted about, every way defaced, that in the sea, poor Pip came all foaming up to the deck, and pretty soon, going to the deck, and in the sea, even as the great Sperm Whale is not a little plan that had been a great fish to swallow up Jonah." --JONAH. "There go flukes!" was now all alive. He seemed to be the first place, you will be all this as it were, to the deck, and in the"
Finished in 2.5s:
 


Pretty close to gibberish, but still fun.  You can find a lot of books online in txt formats for free to play around with.

 

Unfortunately, this is pretty close to just picking random sections of Moby Dick, and splicing it together.  However, if you play around with how much noise, or just use 2 word chains, or put in several books at once, you can get some interesting results.

 

Here is a passage from Moby Dick only using 2 word chains:

 

"a Sperm Whale is not a little behind the whale, the body of the Sperm Whale is not a little behind the whale, which some were forty-eight, some fifty yards long. He said that the whale is a thing not to be the matter. I went up in the sea, and they are afraid to mention even their names, and carry dung, lime-stone, juniper-wood, and some other thing besides me is this the end. Desecrated as the great Sperm Whale is not a rush for what is called a Commonwealth or State--(in Latin, Civitas) which is the most part, that sort of a
[Finished in 1.8s]"
