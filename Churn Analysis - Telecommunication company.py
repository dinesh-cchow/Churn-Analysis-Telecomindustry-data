#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd


# In[52]:


churn_dt=pd.read_csv(r'D:\data science and deepl learning 20 case studies\datascienceforbusiness-master\WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[53]:


# let's see the first 10 rows of data in our churn_dt dataframe
churn_dt.head(10)


# In[54]:


#Let's get the summary stats on the numeric column
churn_dt.describe()


# In[55]:


#Let's check the number of columns and rows in our data
churn_dt.shape


# In[56]:


#Let's check the features in our data
churn_dt.columns.tolist()


# In[57]:


#Let's check for any missing values in the data
churn_dt.isnull().sum()


# In[58]:


#Let's check the unique values for the features in the data
churn_dt.nunique()


# In[59]:


#let's provide the info of our churn_dt data
churn_dt.info()


# Exploratory Data Analysis

# In[60]:


#Always keep a copy of original data for the future purpose before doing EDA
churn_dt_copy=churn_dt.copy()


# In[61]:


churn_dt_copy.drop(['customerID','MonthlyCharges','TotalCharges','tenure'],axis=1,inplace=True)
churn_dt_copy.head()


# In[62]:


#create a new dataset called summary so that we can summarize our churn data
#crosstab - compute a simple cross tabulation of two or more factors. By default computes a frequency table of the factors unless an array of val

summary=pd.concat([pd.crosstab(churn_dt_copy[x],churn_dt_copy.Churn) for x in churn_dt_copy.columns[:-1]],
                 keys=churn_dt_copy.columns[:-1])
summary


# In[63]:


# Let's make a percentage column
summary['churn_percentage']=summary['Yes']/(summary['No']+ summary['Yes'])
summary


# # Visualizations and EDA

# In[64]:


import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns  # used for plot interactive graph
from pylab import rcParams # customize Matplotlib plots using rcparams


# Data to plot
labels=churn_dt['Churn'].value_counts(sort=True).index
sizes=churn_dt['Churn'].value_counts(sort=True)

colors=['lightgreen','red']
explode=(0.05,0) # explode first slice

rcParams['figure.figsize']=7,7
#plot
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=90,)

plt.title('Customer churn Breakdown')
plt.show()


# In[65]:


# create a violin plot showing how monthly charges relate to churn
# we can see that churned customers tend to be higher paying customers

vp=sns.factorplot(x='Churn',y='MonthlyCharges',data=churn_dt,kind='violin',palette='Pastel1')
vpp=sns.factorplot(x='Churn',y='tenure',data=churn_dt,kind='violin',palette='Pastel1')


# In[66]:


def plot_corr(df,size=10):
    corr=df.corr()
    fig, ax=plt.subplots(figsize=(size,size))
    ax.legend()
    cax=ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)),corr.columns,rotation='vertical')
    plt.yticks(range(len(corr.columns)),corr.columns)
    
plot_corr(churn_dt)


# # Preparing Data For Machine Learning Classifier

# In[67]:


# check for empty fields, note, " " is not Null but a spaced character
len(churn_dt[churn_dt['TotalCharges']==' '])


# In[68]:


## drop the missing data 
churn_dt=churn_dt[churn_dt['TotalCharges']!=' ']


# In[69]:


## check if still there are any missing 
len(churn_dt[churn_dt['TotalCharges']==' '])


# In[70]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

## customer id col
id_col=['customerID']

#Target columns

target_cols=['Churn']

#categorical column
cat_cols=churn_dt.nunique()[churn_dt.nunique()<6].keys().tolist()
cat_cols=[x for x in cat_cols if x not in target_cols]

#numerical columns
num_cols=[x for x in churn_dt.columns if x not in cat_cols+target_cols+id_col]

#Binary columns with 2 values
bin_cols=churn_dt.nunique()[churn_dt.nunique()==2].keys().tolist()

#columns more than 2 values
multi_cols=[i for i in cat_cols if i not in bin_cols]

#label encoding Binary columns
le=LabelEncoder()
for i in bin_cols:
    churn_dt[i]=le.fit_transform(churn_dt[i])

# Duplicating columns for multi value columns
churn_dt=pd.get_dummies(data=churn_dt,columns=multi_cols)
churn_dt.head()
#churn_dt['TotalCharges']=pd.to_numeric(churn_dt['TotalCharges'])


# In[71]:


# As still the columns tenure,monthlycharges,totalcharges are not in same scale lets do some scaling
#scaling numerical columns
std=StandardScaler()

#scale data
scaled=std.fit_transform(churn_dt[num_cols])
scaled=pd.DataFrame(scaled,columns=num_cols)

## dropping original values merging scaled vales for numerical columns

df_telecom_org=churn_dt.copy()
churn_dt=churn_dt.drop(columns=num_cols,axis=1)
churn_dt=churn_dt.merge(scaled,left_index=True,right_index=True,how='left')

churn_dt.head()


# In[73]:


#let's drop customer ID Column
churn_dt.drop(['customerID'],axis=1,inplace=True)
churn_dt.head()


# In[74]:


## lets check for the null values
churn_dt[churn_dt.isnull().any(axis=1)]


# In[75]:


#lets drop the 11 ROWS Hence it will not effect our data
churn_dt=churn_dt.dropna()


# In[76]:


#lets double check for any Nulls
churn_dt[churn_dt.isnull().any(axis=1)]


# # Modeling

# In[77]:


from sklearn.model_selection import train_test_split

#we remove the target variable from the training dat
x=churn_dt.drop(['Churn'],axis=1).values

#we assigned the target variable in to the Y variable

y=churn_dt['Churn'].values


# In[78]:


#split it to a 70:30 ratio train:test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[79]:


df_train=pd.DataFrame(x_train)
df_train.head()


# In[82]:


churn_dt.head()


# # Fit a Logistic regression 

# In[94]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

model= LogisticRegression()
model.fit(x_train,y_train)

predictions=model.predict(x_test)
score=model.score(x_test,y_test)

print('Accurarcy= '+str(score))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# # Try Random Forest

# In[93]:


# Let's try Random forests now to see if our results get better
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

modelrf= LogisticRegression()
modelrf.fit(x_train,y_train)

predictions1=modelrf.predict(x_test)

print(" Accuracy {0:.2f}%".format(100*accuracy_score(predictions1,y_test)))
print(confusion_matrix(y_test,predictions1))
print(classification_report(y_test,predictions1))


# # Let's try Deep Learning to see if our results get better than Logistic and Random forest

# In[97]:


# create a simple model
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model1=Sequential()

model1.add(Dense(20,kernel_initializer='uniform',activation='relu',input_dim=40))
model1.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[99]:


# Display model summary and show parameters
model1.summary()


# In[100]:


# Start training our classifier
batch_size=64
epochs= 25

history= model1.fit(x_train,y_train,
                   batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))
score=model1.evaluate(x_test,y_test,verbose=0)
print('test loss:',score[0])
print('test Accuracy:',score[1])


# In[101]:


predictions2=model1.predict(x_test)
predictions2=(predictions2>0.5)

print(confusion_matrix(y_test,predictions2))
print(classification_report(y_test,predictions2))


# # Let's try a Deeper Model and Learn to use checkpoints and Early stopping

# In[108]:


from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

model3=Sequential()

#Hidden layer 1
model3.add(Dense(2000,activation='relu',input_dim=40,kernel_regularizer=l2(0.01)))
model3.add(Dropout(0.3, noise_shape=None,seed=None))

#Hidden Layer 2
model3.add(Dense(1000,activation='relu',input_dim=18,kernel_regularizer=l2(0.01)))
model3.add(Dropout(0.3, noise_shape=None,seed=None))

#Hidden Layer 3
model3.add(Dense(500,activation='relu',kernel_regularizer=l2(0.01)))
model3.add(Dropout(0.3, noise_shape=None,seed=None))

model3.add(Dense(1,activation='sigmoid'))

model3.summary()

#create our model checkpoint so that we save model after each epoch
checkpoint=ModelCheckpoint('deep_model_checkpoint.h5',monitor='val_loss',
                          mode='min',save_best_only=True,verbose=1)


# In[109]:


model3.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[110]:


# Define our Earling stopping criteria
from tensorflow.keras.callbacks import EarlyStopping

earlystop=EarlyStopping(monitor='val_loss', # value being monitored for improvement
                       min_delta=0, #abs value and is the min change required before we stop
                       patience=2, #Number of epochs we wait before stopping
                       verbose=1,
                       restore_best_weights=True)#keeps the best weigths once stopped

# we put our call back into a callback list
callbacks=[earlystop,checkpoint]


# In[112]:


# Start training our classifier
batch_size=32
epochs= 10

history= model3.fit(x_train,y_train,
                   batch_size=batch_size,epochs=epochs,verbose=1,callbacks=callbacks,validation_data=(x_test,y_test))
score=model3.evaluate(x_test,y_test,verbose=0)
print('test loss:',score[0])
print('test Accuracy:',score[1])


# In[ ]:




