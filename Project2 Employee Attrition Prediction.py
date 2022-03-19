#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("C:/Users/Amrita/OneDrive/Desktop/Capstone_project/HR_data1.csv")
df.head()


# In[3]:


df.shape


# In[4]:


#Get the column data types
df.dtypes


# In[5]:


df.isna().sum()


# In[6]:


#View some basic statistical details like percentile, mean, standard deviation etc.
df.describe()


# In[7]:


#Get a count of the number of employee attrition, the number of employees that stayed (no) and the number that left (yes)
df['Attrition'].value_counts()


# In[8]:


#Visualize this count 
sns.countplot(df['Attrition'])


# In[9]:


#Show the number of employees that left and stayed by age
import matplotlib.pyplot as plt
fig_dims = (12, 4)
fig, ax = plt.subplots(figsize=fig_dims)

#ax = axis
sns.countplot(x='Age', hue='Attrition', data = df, palette="colorblind", ax = ax,  edgecolor=sns.color_palette("dark", n_colors = 1));


# In[10]:


#What’s interesting here is that you can see the age with the highest count of employee attrition is age 35 & 38.
#The age with the highest retention is age 33 & 42.


# In[11]:


#Print all of the object data types and their unique values
for column in df.columns:
    if df[column].dtype == object:
        print(str(column) + ' : ' + str(df[column].unique()))
        print(df[column].value_counts())
        print("_________________________________________________________________")


# In[12]:


#Remove unneeded columns

df = df.drop('EmpID', axis = 1) #Employee Name

df = df.drop('Employee Name', axis = 1) #Employee Name

df = df.drop('DOB', axis = 1) #DOB 

df = df.drop('Date of Hire', axis = 1) #Date of Hire

df = df.drop('Employment Status', axis = 1) #Employment Status

df = df.drop('Date of Termination', axis = 1) #Date of Hire

df = df.drop('ManagerID', axis = 1) #ManagerID

df = df.drop('Term Reason', axis = 1) #Term Reason


# In[13]:


df.isna().sum()


# In[14]:


#Get the correlation of the columns
df.corr()


# In[15]:


#Visualize the correlation
plt.figure(figsize=(14,14))  #14in by 14in
sns.heatmap(df.corr(), annot=True, fmt='.0%')


# In[16]:


#Transform non-numeric columns into numerical columns
from sklearn.preprocessing import LabelEncoder

for column in df.columns:
        if df[column].dtype == np.number:
            continue
        df[column] = LabelEncoder().fit_transform(df[column])


# In[17]:


#Split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 1:df.shape[1]].values 
Y = df.iloc[:, 0].values


# In[18]:


# Split the dataset into 75% Training set and 25% Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[19]:


#Use Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X_train, Y_train)


# In[20]:


#Get the accuracy on the testing data
forest.score(X_test, Y_test)


# In[26]:


#Show the confusion matrix and accuracy for  the model on the test data
#Classification accuracy is the ratio of correct predictions to total predictions made.
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, forest.predict(X_test))
  
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
  
print(cm)
print('Model Testing Precision = "{}!"'.format(  TP  / (TP + FP )))
print('Model Testing Recall = "{}!"'.format(  TP  / (TP + FN )))
print()# Print a new line


# In[22]:


# Return the feature importances (the higher, the more important the feature).
importances = pd.DataFrame({'feature':df.iloc[:, 1:df.shape[1]].columns,'importance':np.round(forest.feature_importances_,3)}) #Note: The target column is at position 0
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances


# In[23]:


importances.reset_index(inplace = True)


# In[24]:


#Visualize the importance
importances.plot.bar()


# In[ ]:




