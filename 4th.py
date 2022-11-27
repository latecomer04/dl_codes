import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import StandardScaler






data=pd.read_csv('creditcard.csv')
data.shape

print("Label values:",data.Class.unique())
pd.value_counts(data['Class'],sort=True)




count_classes=pd.value_counts(data['Class'],sort=True);
print(count_classes)
count_classes.plot(kind='bar')





normal_dataset=data[data.Class==0]
fraud_dataset=data[data.Class==1]



bins=np.linspace(200,2500,100)
plt.hist(normal_dataset.Amount,bins=bins,alpha=1,label='Normal',density=True)
plt.hist(fraud_dataset.Amount,bins=bins,alpha=.5,label='Fraud',density=True)



sc=StandardScaler()
data['Time']=sc.fit_transform(data['Time'].values.reshape(-1,1))
data['Amount']=sc.fit_transform(data['Amount'].values.reshape(-1,1))

