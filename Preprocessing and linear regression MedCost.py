#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import missingno
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[38]:


df = pd.read_csv(r"C:\Users\noahw\OneDrive\Desktop\machinelearning\insurance.csv")


# In[39]:


df.head()


# In[40]:


df.sex = df.sex.replace(['male','female'],[1,0])
df.smoker = df.smoker.replace(['yes','no'],[1,0])


# In[41]:


df.head()


# In[42]:


df.info()


# In[43]:


pd.DataFrame(df.groupby("region")["age","charges","bmi","children"].mean().sort_values("age",ascending=False)[:10])


# In[44]:


pd.DataFrame(df.groupby("sex")["age","charges","bmi","children"].mean().sort_values("age",ascending=False)[:10])


# In[45]:


pd.DataFrame(df.groupby("smoker")["age","charges","bmi","children"].mean().sort_values("age",ascending=False)[:10])


# In[46]:


pd.DataFrame(df.groupby("children")["age","charges","bmi"].mean().sort_values("age",ascending=False)[:10])


# In[47]:


X = df.drop(['charges','region'],axis=1)
region = pd.get_dummies(df.region)
X  = pd.concat([X,region], axis=1)
y = df.charges
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=123)


# In[48]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)
print("Score Test:",model.score(X_test,y_test))
pred_br = model.predict(X_test)
print("MAE:",mean_absolute_error(y_test,pred_br))
print("MSE:",mean_squared_error(y_test,pred_br))


# In[ ]:




