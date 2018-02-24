---
author: "Zachary Swarth"
date: 2018-02-21
title: Classifying 311 Noise Complaint Data - Cleaning the data.
---

*Page 1 of 2 - Cleaning the Data*

*Create a model that can classify a noise complaint by type (i.e., commercial, vehicle, park, house of worship, etc.)  The idea came from a friend, who I believe encountered it at a job interview at a now defunct company.*

You can find the entire code and the smaller data set here: [311 Data](https://github.com/zswarth/311Data)

After playing around with Kaggle Competitions (if you've not see these, [Predicting Survival on the Titanic](https://www.kaggle.com/c/titanic-survival) is a great place to start, with tons of tutorials), I decided to look for data somewhere less structured.

NYC Open Data gives access to huge amounts of information from complaints made to 311.  You can find the full data set here:
[311 Data](https://nycopendata.socrata.com/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9).

I'm going to be looking just a the noise complain data, which we can find here:
[311 Noise Complains](https://data.cityofnewyork.us/Social-Services/2012-NYC-Noise-Complaints/w58m-6tbm)

Let's start looking at the data:

The original file is quite large; I can come back to it later, but for now I just want to play around with a few thousand lines.  I've created a csv of the file with only the first 100,000 lines.  This is plenty for the time being.

​

```python
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
```


```python
data = pd.read_csv('small_data.csv')
```


```python
list(data.columns.values)
```




    ['Unnamed: 0',
     'Unnamed: 0.1',
     'Unique Key',
     'Created Date',
     'Closed Date',
     'Agency',
     'Agency Name',
     'Complaint Type',
     'Descriptor',
     'Location Type',
     'Incident Zip',
     'Incident Address',
     'Street Name',
     'Cross Street 1',
     'Cross Street 2',
     'Intersection Street 1',
     'Intersection Street 2',
     'Address Type',
     'City',
     'Landmark',
     'Facility Type',
     'Status',
     'Due Date',
     'Resolution Action Updated Date',
     'Community Board',
     'Borough',
     'X Coordinate (State Plane)',
     'Y Coordinate (State Plane)',
     'Park Facility Name',
     'Park Borough',
     'School Name',
     'School Number',
     'School Region',
     'School Code',
     'School Phone Number',
     'School Address',
     'School City',
     'School State',
     'School Zip',
     'School Not Found',
     'School or Citywide Complaint',
     'Vehicle Type',
     'Taxi Company Borough',
     'Taxi Pick Up Location',
     'Bridge Highway Name',
     'Bridge Highway Direction',
     'Road Ramp',
     'Bridge Highway Segment',
     'Garage Lot Name',
     'Ferry Direction',
     'Ferry Terminal Name',
     'Latitude',
     'Longitude',
     'Location']



 

We can clearly see we have a ton of variables here, but most of them are pretty useless.   I deleted any column that either had no values, or hugely overlapping values.  For example, 'Agency Name' and 'Agency' contain the same information.  "School Name" is an empty column.   Unique Key is unique for each one, so I assume will not give a huge amount of useful information (perhaps I'm wrong there though).

​​
```python
dropped_fields = ['Unique Key','Agency Name','Park Facility Name','School Name','School Number','School Code','School Phone Number','School Address','School City','School State','School Zip','School Region','School Not Found','School or Citywide Complaint','Vehicle Type','Taxi Company Borough','Taxi Pick Up Location','Bridge Highway Name','Bridge Highway Direction','Road Ramp','Bridge Highway Segment','Garage Lot Name','Ferry Direction','Ferry Terminal Name','Latitude','Longitude','Location']
data.drop(dropped_fields, axis=1, inplace = True)
```

​
The rest of the data I wanted to clean up a bit before I ran anything.

For the date, I creased a column for Day of the Week, Year, and Month.  I'm guessing the day of the week and month may have some predictive properties, but the date it's current form would be hard to work with. 

​

```python
import datetime
def date(d):
	month, day, year = (int(x) for x in d.split('/'))
	return datetime.date(year,month, day).weekday()

def month(d):
	month, day, year = (int(x) for x in d.split('/'))
	return month

def year(d):
	month, day, year = (int(x) for x in d.split('/'))
	return year



data['Weekday'] = data['Created Date'].apply(date)
data['Year'] = data['Created Date'].apply(year)
data['Month'] = data['Created Date'].apply(month)

```


 

Depending on what model I choose to run on these, I also need to find a way to deal with the categorical data as well as the non ordered data.
For the data, perhaps I could have dealt with the months as one numerical column (as the order does matter) as opposed to 12 categories (same with days of the week), but i decided to split them up anyway.

I created dummy variables for Day, Month, Year, and Agency.

```python
​dummies = pd.get_dummies(data['Month'])
for m_number in dummies.columns:
	data['month_%s' % m_number] = dummies[m_number]

dummies = pd.get_dummies(data['Year'])
for m_number in dummies.columns:
	data['Year_%s' % m_number] = dummies[m_number]

dummies = pd.get_dummies(data['Weekday'])
for m_number in dummies.columns:
	data['Weekday_%s' % m_number] = dummies[m_number]

data = data.drop('Created Date', axis = 1)
data = data.drop('Weekday', axis = 1)
data = data.drop('Year', axis = 1)
data = data.drop('Month', axis = 1)

```
 

There were a ton of zip codes.  I grouped them together in groups of 10, and labeled all the unlabeled ones with their own zip code.

For the Street name, i divided each section into "Other", "Avenue", "Street", "Broadway", "Road".  If you know NYC, different street names give some indication into whether we're looking at a residential or commercial region, or what part of the city we're in (i.e., Road isn't very common in Manhattan.)


```python

dummies = pd.get_dummies(data['Agency'])
for agent in dummies.columns:
	data['agency_%s' % agent] = dummies[agent]


# Group Zip Codes in nearest 10
def zip(x):
	if np.isnan(x):
		return 10000
	else:
		return x//10
	return x//10

data['Incident Zip'] = data['Incident Zip'].apply(zip)

dummies = pd.get_dummies(data['Incident Zip'])
for a in dummies.columns:
	data['zip_%s' % a] = dummies[a]

data = data.drop('Incident Zip', axis = 1)

```
​
 

If we look at the city data, it's interesting.  4 of the 5 boroughs were generally labeled.  however, Queens was chopped up very finely amongst different neighborhoods.  I'm sure there a decent about of predictive values here, but I din't want to make my model too specific, so I generalized all non New York, Brooklyn, Bronx, and Staten Island lables as "Rare'.

​
```python

#Make City into a few key bigger sections
def C_name(name):
	cities = ['NEW YORK', 'BROOKLYN','BRONX', 'STATEN ISLAND']
	if name not in cities:
		return 'RARE'
	else:
		return name

data['City'] = data['City'].apply(C_name)


dummies = pd.get_dummies(data['City'])
for a in dummies.columns:
	data['City_%s' % a] = dummies[a]


data = data.drop('City', axis = 1)

```
 
I got rid of a few last data points I didn't use (there are things here I may come back to later, but for now I just want something simple I can play with and improve upon).  For example, community board is probably important, and I'd like to go add this in later.  I also just changed the landmark data into a binary value.



```python

def land(x):
	if type(x) == str:
		return 1
	else:
		return 0


data['Landmark'] = data['Landmark'].apply(land)


#Facility Types

dummies = pd.get_dummies(data['Facility Type'])
for a in dummies.columns:
	data['Facility_%s' % a] = dummies[a]


data = data.drop('Facility Type', axis = 1)
```


```python
#Status
dummies = pd.get_dummies(data['Status'])
for a in dummies.columns:
	data['Status_%s' % a] = dummies[a]

data = data.drop('Status', axis = 1)

data = data.drop('Community Board', axis=1)

##address type
dummies = pd.get_dummies(data['Address Type'])
for a in dummies.columns:
	data['CommunityBoard_%s' % a] = dummies[a]

data = data.drop('Address Type', axis = 1)
```

```python

#Location Type
dummies = pd.get_dummies(data['Location Type'])
for a in dummies.columns:
	data['AddressType_%s' % a] = dummies[a]

data = data.drop('Location Type', axis = 1)


drop = ['Incident Address', 'Descriptor', 'Closed Date', 'Cross Street 1', 'Cross Street 2','Intersection Street 1', 'Intersection Street 2', 'Due Date', 'Resolution Action Updated Date', 'X Coordinate (State Plane)', 'Y Coordinate (State Plane)', 'Park Borough'] 
data.drop(drop, axis=1, inplace = True)
```

```python
#Just O.H.A. Borough
dummies = pd.get_dummies(data['Borough'])
for a in dummies.columns:
	data['Borough_%s' % a] = dummies[a]

data = data.drop('Borough', axis = 1)
```
  

We now have a dataset we can begin to look at.


