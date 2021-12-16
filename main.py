## problem statement:- finding the species of the penguins based on their sex and island.
## pickle file:-
## Pickle can be used to serialize Python object structures,
## which refers to the process of converting an object in the memory to a byte stream
## that can be stored as a binary file on disk. ...
## When we load it back to a Python program, this binary file can be de-serialized back to a Python object.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Read the dataset
pen = pd.read_csv("penguins_cleaned.csv")
df = pen.copy()
target='species'
encode=['sex','island']
for col in encode:
    dummy=pd.get_dummies(df[col],prefix=col)
    df=pd.concat([df,dummy],axis=1)
    del df[col]

target_mapper={'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]
df['species']=df['species'].apply(target_encode)
## Separting x and y
x=df.drop('species',axis=1)
y=df['species']
## Build random forest model
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x,y)
## saving the model
import pickle
pickle.dump(model,open('penguins_clf.pkl','wb'))



