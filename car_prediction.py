#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sns.set_palette("GnBu_d")
sns.set_style('whitegrid')q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]


# In[31]:


car_data = pd.read_csv("dataset.csv")


# In[32]:


car_data.head()


# In[33]:


car_data.describe(include='all')


# In[34]:


car_data.info()


# In[37]:


df = car_data.drop(labels='Model', axis=1)


# In[38]:


df.isna().sum()


# In[39]:


df_no_mv = df.dropna()


# In[40]:


plt.figure(figsize=[11,5])
sns.distplot(df_no_mv['Price'])
plt.title('Car Price Distribution Plot')


# In[41]:


plt.figure(figsize=[17,5])
plt.subplot(1,3,1)
sns.distplot(df_no_mv['Year'])
plt.title('Car Year Distribution Plot')


# In[42]:


q = df_no_mv['Price'].quantile(0.99)
data_1 = df_no_mv[df_no_mv['Price']<q]


# In[43]:


plt.figure(figsize=[11,5])
sns.distplot(data_1['Price'])
plt.title('Car Price Distribution Plot')


# In[44]:


sns.distplot(df_no_mv['Mileage'])
plt.title('Car Mileage Distribution Plot')


# In[45]:


q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]


# In[46]:


# Hurray this plot looks kind of normal
sns.distplot(data_2['Mileage'])


# In[47]:


sns.distplot(df_no_mv['Year'])


# In[48]:


q = data_2['Year'].quantile(0.01)
data_3 = data_2[data_2['Year']>q]


# In[49]:


sns.distplot(data_3['Year'])


# In[50]:


sns.distplot(df_no_mv['EngineV'])
plt.title('EngineV Distribution Plot')


# In[51]:


data_4 = data_3[data_3['EngineV']<6.5]
sns.distplot(data_4['EngineV'])


# In[52]:


cleaned_data = data_4.reset_index(drop=True)
cleaned_data.describe()


# In[53]:


plt.figure(figsize=[20,7])
plt.subplot(1,3,1)
plt.title("Price and Year")
sns.scatterplot(x='Year',y='Price',data=cleaned_data)

plt.subplot(1,3,2)
plt.title("Price and Mileage")
sns.scatterplot(x='Price',y='Mileage',data=cleaned_data)

plt.subplot(1,3,3)
sns.scatterplot(y='Price',x='EngineV',data=cleaned_data)
plt.title("Price and EngineV")


# In[56]:


sns.distplot(cleaned_data['Price'])


# In[57]:


log_price = np.log(cleaned_data['Price'])
cleaned_data['log_price'] = log_price
cleaned_data.head()


# In[58]:


plt.figure(figsize=[20,7])
plt.subplot(1,3,1)
plt.title("Log price and Year")
sns.scatterplot(x='Year',y='log_price',data=cleaned_data)

plt.subplot(1,3,2)
plt.title("Log price and Mileage")
sns.scatterplot(y='log_price',x='Mileage',data=cleaned_data)

plt.subplot(1,3,3)
sns.scatterplot(y='log_price',x='EngineV',data=cleaned_data)
plt.title("Log price and EngineV")


# In[59]:


cleaned_data = cleaned_data.drop(['Price'],axis=1)


# In[60]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = cleaned_data[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns
vif


# In[61]:


data_no_multicolinearity = cleaned_data.drop('Year',axis=1)


# In[62]:


data_no_multicolinearity.head()


# In[63]:


from sklearn.preprocessing import LabelEncoder
temp_data = data_no_multicolinearity.copy()
for col in temp_data.columns:
    if temp_data[col].dtypes == 'object':
        encoder = LabelEncoder()
        temp_data[col] = encoder.fit_transform(temp_data[col])
        
# Correated Features with target variable
print('\n--Correated Features with target variable--\n')
print(abs(temp_data.corrwith(temp_data['log_price'])).sort_values(ascending=False)[1:])


# In[64]:


plt.figure(figsize=[15,7])
sns.heatmap(data_no_multicolinearity.corr(), annot=True)


# In[65]:


from sklearn.ensemble import ExtraTreesRegressor
X = temp_data.drop('log_price',axis=1)
y = temp_data['log_price']
model = ExtraTreesRegressor()
model.fit(X,y)


# In[66]:


plt.figure(figsize=[12,6])
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(6).plot(kind='barh')
plt.show()


# In[67]:


print(feat_importances.sort_values(ascending=False))


# In[68]:


data_with_dummies = pd.get_dummies(data_no_multicolinearity,drop_first=True)


# In[69]:


data_with_dummies.head()


# In[70]:


x = data_with_dummies.drop('log_price',axis=1)
y = data_with_dummies['log_price']


# In[71]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x[['Mileage','EngineV']])


# In[72]:


inputs_scaled = scaler.transform(x[['Mileage','EngineV']])
scaled_data = pd.DataFrame(inputs_scaled,columns=['Mileage','EngineV'])


# In[73]:


input_scaled2 =scaled_data.join(x.drop(['Mileage','EngineV'],axis=1))


# In[74]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input_scaled2,y,test_size=0.2, random_state=365)


# In[75]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# Training Model
lr.fit(x_train,y_train)

# Model Summary
y_pred_lr = lr.predict(x_test)

r_squared = r2_score(y_test,y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_test,y_pred_lr))
print("R_squared :",r_squared)
print("RMSE :",rmse)


# In[76]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

# Training Model
rf.fit(x_train,y_train)

# Model Summary
y_pred_rf = rf.predict(x_test)

r_squared = r2_score(y_test,y_pred_rf)
rmse = np.sqrt(mean_squared_error(y_test,y_pred_rf))
print("R_squared :",r_squared)
print("RMSE :",rmse)


# In[77]:


from sklearn.ensemble import GradientBoostingRegressor
gbt = GradientBoostingRegressor()

# Training Model
gbt.fit(x_train,y_train)

# Model Summary
y_pred_gbt = gbt.predict(x_test)

r_squared = r2_score(y_test,y_pred_gbt)
rmse = np.sqrt(mean_squared_error(y_test,y_pred_gbt))
print("R_squared :",r_squared)
print("RMSE :",rmse)


# In[78]:


df_ev = pd.DataFrame(np.exp(y_pred_gbt), columns=['Predicted Price'])

# We can also include the Actual price column in that data frame (so we can manually compare them)
y_test = y_test.reset_index(drop=True)
df_ev['Actual Price'] = np.exp(y_test)

# we can calculate the difference between the targets and the predictions
df_ev['Residual'] = df_ev['Actual Price'] - df_ev['Predicted Price']
df_ev['Difference%'] = np.absolute(df_ev['Residual']/df_ev['Actual Price']*100)

pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_ev.sort_values(by=['Difference%'])

df_ev.tail(5)


# In[ ]:




