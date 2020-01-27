#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 09:20:21 2018

@author: emma
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor


df=pd.read_table("pictures-train.tsv")

# Clean the data 
df[df["viewed"]<0]=np.nan
df[df["n_comments"]<0]=np.nan
df.dropna(inplace=True)
df=df[-df["takenon"].str.contains('-00')]
df=df[-df["votedon"].str.contains('-00')]
# Convert datetime to float
df["takenon"]=pd.to_datetime(df["takenon"])
df["votedon"]=pd.to_datetime(df["votedon"])

# Line graphs of average number of pictures, upvotes, views, and comments by year
mean_df=df.groupby(df["votedon"].dt.year).mean()
plt.figure(6, figsize=(15,15))
plt.yscale('log')
plt.plot(mean_df.index.values,mean_df["votes"],label="votes")
plt.plot(mean_df.index.values,mean_df["viewed"],label="views")
plt.plot(mean_df.index.values,mean_df["n_comments"],label="comments")
plt.plot(mean_df.index.values,df.groupby(df["votedon"].dt.year).size().values,label="pictures")
plt.legend()
plt.savefig("line_graphs.png")

epoch=pd.DataFrame(pd.to_datetime({'year':[1970], 'month':[1], 'day':[1]}))
edf=epoch.append([epoch]*len(df), ignore_index=True)
df = df.join(edf)
df.dropna(inplace=True)
df["takenon"]=df["takenon"]-df[0]
df["votedon"]=df["votedon"]-df[0]
df["takenon"]=df["takenon"].astype('timedelta64[s]')
df["votedon"]=df["votedon"].astype('timedelta64[s]')
df.drop([0], axis=1, inplace=True)
df=df[df.votes != 0]
# enlarge the data with other informations gaining from the data
df["log"]=np.log(df["votes"])
lifespan=df.groupby("author_id")["takenon"].agg(["max","min"])
lifespan["tenure"]=lifespan["max"]-lifespan["min"]
df = df.join(lifespan,on='author_id',how='left')
df.drop(['min','max'], axis=1, inplace=True)

# Find suitable features and the predictive model

rfr = RandomForestRegressor()

features=["takenon","votedon","n_comments","etitle","viewed","region","tenure"]
X = pd.get_dummies(df[features])
X_train, X_test, y_train, y_test = tts(X, df["log"], test_size=0.3, random_state=42)

print("RandomForestRegressor")
rfr.fit(X_train, y_train)
print(rfr.score(X_train, y_train), rfr.score(X_test, y_test))

with open("model.p", "wb") as ofile: 
    pickle.dump(rfr, ofile)

# Pie chart of distributions of pictures by region
region_dict=df["region"].value_counts().to_dict()
rlabels=region_dict.keys()
rvalues=region_dict.values()
plt.figure(1,figsize=(15,15))
plt.pie(x=rvalues, labels=rlabels,autopct='%1.1f%%')
plt.savefig("region.png")
# Pie chart of distributions of pictures by category and region
etitle_dict=df["etitle"].value_counts().to_dict()
elabels=etitle_dict.keys()
evalues=etitle_dict.values()
plt.figure(2,figsize=(15,15))
plt.pie(x=evalues, labels=elabels,autopct='%1.1f%%')
plt.savefig("category.png")
# Histograms of upvotes, views, and comments
plt.figure(3,figsize=(15,15))
plt.hist(df.votes)
plt.savefig("upvotes")
plt.figure(4,figsize=(15,15))
plt.hist(df.viewed)
plt.savefig("views")
plt.figure(5,figsize=(15,15))
plt.hist(df.n_comments)
plt.savefig("comments")
# Histogram of the difference between the true and predicted number of upvotes
plt.figure(7, figsize=(15,15))
plt.hist(y_test-rfr.predict(X_test))
plt.savefig("difference.png")