#!/usr/bin/env python
# coding: utf-8

# # SALE PRICE PREDICTION

# ## Section 1 : Dataset description
# 
# The dataset has 82 columns that include 23 nominal, 23 ordinal, 14 discrete, and 20 continuous variables (and 2 additional case identifiers).
# 
# **[Data Soucre](https://github.com/Murdocc007/Statistical-Modelling/blob/master/Ames%20Housing/AmesHousing.txt)**  

# In[161]:


### important Librairies 
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


# In[162]:


#Import dataset
df_description_clos = pd.read_csv("df_description_clos.csv")

df_description_clos.head()


# In[163]:


#Transform the column name into an index 
df_description_clos = df_description_clos.set_index("Columns name")
df_description_clos.head()


# [Reference indexing and selecting data](https://pandas.pydata.org/docs/user_guide/indexing.html#indexing-choice:~:text=Cookbook-,Indexing%20and%20selecting%20data,-%23)

# In[164]:


#Set a function to display the feature description
def info_func(col_name):
    return df_description_clos.loc[col_name]["Columns description"]


# It is very important to have a function that displays the feature description, because when processing the data, we will need it to see the feature descriptions as we go.

# In[165]:


# Display the description of the some features 
info_func(["Order", "Lot Area", "Street", "Alley", "Pool Area", "Pool QC", "Fence"])


# In[166]:


#Import dataset
df =pd.read_csv("AmesHousing.txt", delimiter="\t")


# In[167]:


#Show the shape and types of the dataset
print(df.shape)
print(len(str(df.shape))*'-')
print(df.dtypes.value_counts())


# In[168]:


#Display the first 5 rows of the dataset
df.head(3)


# We have a dataset with 2930 observations and 82 characteristics including 43 objects, 28 int64 and 11 float64

# In[169]:


#The last 5 rows of the datset
df.tail(3)


# ## Section 2 : Exploratory Data Analysis (EDA)
# 
# 
# Exploratory data analysis (EDA) is a very important step in data analysis. It allows us with visualizations and statistical analysis (univariate, bivariate and multivariate) to understand the data with which we work and to better understand their relationships. So let's start exploring our target variable and how other features influence it.

# ### ***Duplicated***
# 
# In machine learning models, duplicates can lead to biases and inaccuracies. This is why it is very important to manage duplicates in the dataset.
# 
# [Reference duplicated](http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.duplicated.html)
# 
# [Reference duplicated](https://www.tutorialspoint.com/handling-duplicate-values-from-datasets-in-python)

# In[170]:


df[df.duplicated(keep = False)]


# There are no duplicate values

# ### ***Data info***
# 
# [Reference data info](https://pandas.pydata.org/docs/getting_started/intro_tutorials/02_read_write.html#min-tut-02-read-write:~:text=La%20m%C3%A9thode%20info()fournit%20des%20informations%20techniques%20sur%20un%20DataFrame%2C%20expliquons%20donc%20la%20sortie%20plus%20en%20d%C3%A9tail%C2%A0%3A)

# In[171]:


#Show dataset information
df.info()


# We can see the type (int64, float64 and object), the shape (2930 entries and 82 columns) of the dataset and the missing values for each feature. 

# ### ***Let’s delete columns that are not useful for Maching learning***
# 
# These are not useful columns for machine learning. Indeed, these two variables will not have much effect on price prediction, because they are only property identifiers.  

# In[172]:


#Let's look at the columns to delete
info_func(["Order", "PID"])


# In[173]:


def fun_sup(df):
    df =df.drop(["Order", "PID"], axis=1)
    return df


# In[174]:


# Delete columns that are not useful for Machine Learning
data =fun_sup(df)


# In[175]:


# Display the shape of the dataset
data.shape


# ### ***Descriptive statistics***
# 
# [Reference descriptive statistic](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)

# In[176]:


#Descriptive statistics
data.describe().T


# These descriptive statistics show a large variation between our data (for example by looking at the average, the min and the max of the variable of X, we notice this variation very quickly). This data will therefore have to be scaled before being used in a machine learning model. It should be noted that many models are subject to variations in the data. For example, linear regression and k-nearest neighbors algorithms are sensitive to variations in the data.

# ### ***Correlation***
# 
# Correlation explains how one or more variables relate to each other. These variables can be features of input data that were used to predict our target variable.
# 
# [Reference correlation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html)

# In[177]:


corr =round(data.select_dtypes(include =np.number).corr().abs(), 2)
plt.figure(figsize=(26, 20))

heatmap = sns.heatmap(corr, fmt='.1f', cmap='GnBu', vmin=0.25, vmax=0.5, annot = True, annot_kws={"size": 15})

heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':28}, pad=14);


# We see some high and low correlation between characteristics. We will later use a feature selection technique to select the most relevant features.

# In[429]:


# Create correlation matrix from train data excluding `SalePrice`
corr = data.select_dtypes(include =np.number).corr().abs()

# Select correlations greater than 0.5
high_corr = corr[abs(corr_mat) >= 0.5]

# Plot correlation heatmap
plt.figure(figsize=(18, 16))
heatmap = sns.heatmap(high_corr_mat, annot=True, fmt='.1f', cmap='GnBu', vmin=0.25, vmax=1, annot_kws={"size": 14})
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':22}, pad=14)


# There is multicollinearity in our data. The features below are highly correlated :
# 
# - Garage Cars and Garage Area
# - Garage Yr Blt and Year Built
# - 1st Flr SF and Total Bsmt SF
# - Gr Liv Area and TotRms AbvGrd
# 
# Multicolliniarity negatively impacts prediction models because it duplicates the same information and thus increases the standard errors of estimators. Therefore, it is useful to keep only one feature from each pair of highly correlated features. So, in each pair, we remove the feature that is weakly correlated with the sale price.

# ***Correlation of all features with the sale price***

# In[179]:


#Correlation level with features
plt.figure(figsize=(3, 12))
corr_SalePrice =data.select_dtypes(include =np.number).corr()[["SalePrice"]].abs().sort_values(by = "SalePrice", ascending = False)

heatmap =sns.heatmap(corr_SalePrice, cmap='GnBu', vmin=0.25, vmax=1, annot = True)

heatmap.set_title('Features Correlating with Sales Price', fontdict={'fontsize':12}, pad=10);


# After examining the correlation of the features with the sale price, we remove the characteristics weakly correlated with the selling price such as: Garage Cars, Garage Yr Built, 1st Flr SF and TotRms AbvGrd.

# In[180]:


info_func(["Garage Cars", "Garage Yr Blt", "1st Flr SF", "TotRms AbvGrd"])


# In[181]:


data[["Garage Cars", "Garage Yr Blt", "1st Flr SF", "TotRms AbvGrd"]]


# In[182]:


for col in ["Garage Cars", "Garage Yr Blt", "1st Flr SF", "TotRms AbvGrd"]:
    data = data.drop(col, axis = 1)


# In[183]:


data.shape


# https://ecampusontario.pressbooks.pub/introstats/chapter/13-3-standard-error-of-the-estimate/
# 
# https://chriskhanhtran.github.io/minimal-portfolio/projects/ames-house-price.html

# ### ***Data visualization***
# 
# Let's draw the graphs of some features of the dataset
# 

# ***Sale Price***

# In[184]:


info_func("SalePrice")


# In[185]:


# Price distribution
fig, ax = plt.subplots(figsize=(16,6), ncols=2)
sns.histplot(x = "SalePrice", bins=50, kde=True, ax=ax[0], data =data)
ax[0].set_title('Histogram of Sales Price', fontsize=20)
ax[0].set_xlabel('Sale Price')
ax[0].set_ylabel('Number of Sales')

#Price box plot
sns.boxplot(y = "SalePrice", ax=ax[1], data =data)
ax[1].set_title("Box plot Sale Price distribution", fontsize=20)
ax[1].set_ylabel('Sale Price')


# Note that most homes cost between 100,000 dollars and 300,000 dollars. On the other hand, the most expensive houses cost up to more than 700,000 dollars, few houses are sold as soon as the price exceeds a value of 400,000 dollars.
# This variable has many outliers as seen in the histogram and boxplot above. 

# There is a high correlation between price and the following variables : "Overall Qual", "Gr Liv Area", "Garage Cars", "Garage Area"... Let's plot a scatter plot of these features with the sale price.

# ***Feature scatterplot with high correlation to price***

# In[186]:


info_func("Overall Qual")


# In[187]:


#Scatterplot of sale price and overall quality
plt.figure(figsize=(8, 6))
sns.regplot(x ="Overall Qual", y="SalePrice", scatter_kws={'alpha': 0.4}, 
                line_kws={'color': 'red','linewidth':0.8} ,data = data)
plt.title("Scatterplot of sale price and overall quality")


# In[188]:


info_func("Gr Liv Area")


# In[189]:


plt.figure(figsize=(8, 6))
sns.regplot(x ="Gr Liv Area", y="SalePrice", scatter_kws={'alpha': 0.4}, 
                line_kws={'color': 'red','linewidth':0.8}, data =data)
plt.title("Scatterplot of sale price and Gr Liv Area")


# In[190]:


info_func("Garage Cars")


# In[430]:


plt.figure(figsize=(8, 6))
sns.regplot(x ="Total Bsmt SF", y="SalePrice", scatter_kws={'alpha': 0.4}, 
                line_kws={'color': 'red','linewidth':0.8} ,data = data)
plt.title("Scatterplot of sale price and Total Bsmt SF")


# In[192]:


info_func("Garage Area")


# In[193]:


plt.figure(figsize=(8, 6))
sns.regplot(x ="Garage Area", y="SalePrice", scatter_kws={'alpha': 0.4}, 
                line_kws={'color': 'red','linewidth':0.8} ,data = data)
plt.title("Scatterplot of sale price and Garage Area")


# All scatter plots show many outliers. Let's adjust the data and use the logarithm to transform the selling price to reduce the discrepancy between the data.
# We visualize the distribution of certain characteristics with histograms, bar charts and box plots

# ***Zoning type***

# In[194]:


print(info_func("MS Zoning"))

print( "\nUnique values :", data['MS Zoning'].unique())

print( "\nCount values :\n", data['MS Zoning'].value_counts(dropna = False))


# In[195]:


fig, ax = plt.subplots(figsize=(16,6), ncols=2)
data['MS Zoning'].value_counts().plot(kind='bar', ax=ax[0])
ax[0].set_title("Bar Chart of MS Zoning", fontsize=20)
ax[0].set_xlabel('Zoning type')
ax[0].set_ylabel('Number of Sales')

sns.boxplot(x = "MS Zoning", y = "SalePrice", ax=ax[1], data = data)
ax[1].set_title("Distribution MS Zoning and Sale Price", fontsize=20)
ax[1].set_xlabel('Zoning type')
ax[1].set_ylabel('Sale Price')


# MS Zoning (Nominal) : Identifies the general zoning classification of the sale.
# 		
#        A	Agriculture
#        C	Commercial
#        FV	Floating Village Residential
#        I	Industrial
#        RH	Residential High Density
#        RL	Residential Low Density
#        RP	Residential Low Density Park 
#        RM	Residential Medium Density
#        
# Note that the low-density residential area includes the best-selling and most expensive homes with extreme prices of up to over $700,000. However, there are many outliers in this category.

# ***Year Sold***

# In[196]:


print(info_func("Yr Sold"))

print( "\nUnique values :", data['Yr Sold'].unique())

print( "\nCount values :\n", data['Yr Sold'].value_counts(dropna = False))


# In[197]:


fig, ax = plt.subplots(figsize=(16,6), ncols=2)
data['Yr Sold'].value_counts().plot(kind='bar', ax=ax[0])
ax[0].set_title("Bar Chart of Year of Home Sales", fontsize=20)
ax[0].set_xlabel('Sale Year')
ax[0].set_ylabel('Number of Sales')

sns.boxplot(x = 'Yr Sold', y = "SalePrice", data = data)
ax[1].set_title("Distribution Sale price and year sold" , fontsize=20)
ax[1].set_xlabel('Year sold')
ax[1].set_ylabel('Sale Price')


# We see that the years from 2006 to 2009 have almost the same number of sales, this corresponds to the real estate boom before the financial crisis of 2008. In 2010, sales fell by almost half.

# ***Lot Area***

# In[198]:


info_func("Lot Area")


# In[199]:


plt.figure(figsize=(9,6))
data['Lot Area'].hist(bins=50)
plt.title("Histogram of Lot Area")
plt.xlabel('Lot size in square feet')
plt.ylabel('Number of Sales')


# Most of the homes sold are about less than 20,000 square feet. We also note that the values are very asymmetrical on the right.

# ***Histograms of all numerical features***

# In[200]:


data.hist(figsize=(16, 20), color="purple", bins=50, xlabelsize=8, ylabelsize=8);
plt.suptitle("Distribution of numerical features", x = 0.5, y = 0.91, fontsize = 20)


# Just looking at the table of descriptive statistics and graphs of some variables above, we find that most of our numerical variables are asymmetries and that numerical and categorical variables have many outliers. In this case, as we said, let's need to scale data before implementing some machine learning models, especially linear models that require data normality.

# ## Section 3 : ***Preprocessing Data***

# ### ***Missing values***
# 
# In data science, whenever we have to deal with missing values in a data set, we have to ask ourselves, "Do we know what the absence of these values means?" If the answer is no, we should ask our source. It is very important to know where the missing values come from, because after processing them will be faster and much better.
# 
# Here we deal with missing data taking into account the documentation you will find [here](https://github.com/Murdocc007/Statistical-Modelling/blob/master/Ames%20Housing/AmesHousingDataDocumentation.txt)
# 
# For more information, see the references
# 
# 

# In[201]:


#Missing values
def missing_values(df):
    missing_values = df.isnull().sum().sort_values()
    missing_values = missing_values[missing_values!=0]
    pourcentages_missing_values = round(missing_values[(missing_values>0)]/len(df)*100, 3)
    data_missing_values = pd.DataFrame({"number of missing values" : missing_values,"pourcentage of missing values":pourcentages_missing_values})
    return data_missing_values
    


# In[202]:


# Number and percentage of missing values
data_missing_values = missing_values(data)
data_missing_values 


# ### ***Imputation of missing values***
# 
# Imputation is the process of replacing a missing value with a substituted value, the “best estimate”. Imputation should be one of the first feature engineering steps we undertake as it will affect any downstream pre-processing.
# 
# Let's first convert the MS SubClass feature to object type feature and create two new features.
# 
# 

# In[203]:


info_func('MS SubClass')


# In[204]:


# According to the documentation, the X function is nominal, so we convert it into an object type
new_data = data.copy()

print(new_data['MS SubClass'].dtypes, "\n")
print(new_data['MS SubClass'].value_counts())


# In[205]:


# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html
new_data['MS SubClass'] = new_data['MS SubClass'].astype("str")

for col in ['MS SubClass'] :
    print(new_data[col].dtypes)


# ***Create two new features***
# 
# We create two new features, a binary feature that takes the value 1 if the property is renewed and 0 if not, and the second feature is the age of the property, from the year of construction to the year of sale.

# In[206]:


new_data["YearRemodDemy"]= new_data.apply(lambda x: 1 if x["Year Built"]== x["Year Remod/Add"] else 0, axis=1)

new_data["AgeBuilt"]= new_data.apply(lambda x: x["Yr Sold"] - x["Year Built"], axis =1) 


# In[207]:


new_data[new_data["AgeBuilt"]<0]


# ***Remove negative value in AgeBuilt feature and unnecessary features***

# In[208]:


new_data = new_data.drop(2180, axis = 0)
new_data = new_data.drop(["Year Remod/Add", 'Mo Sold',"Year Built", "Yr Sold"] , axis =1)


# In[209]:


new_data.shape


# ***Handling missing values***
# 
# [reference](https://pandas.pydata.org/docs/user_guide/missing_data.html)
# 
# [reference](https://towardsdatascience.com/imputing-missing-data-with-simple-and-advanced-techniques-f5c7b157fb87)
# 

# In[210]:


# Describe features with missing values
info_func(data_missing_values.index)


# Fill missing values in categorical features considering [documentation](https://github.com/Murdocc007/Statistical-Modelling/blob/master/Ames%20Housing/AmesHousingDataDocumentation.txt)

# In[211]:


for col in ['Mas Vnr Type', 'Misc Feature']:
    new_data[col]=new_data['Misc Feature'].fillna("None")
    
for col in ['BsmtFin Type 1', 'Bsmt Cond', 'Bsmt Qual', 'BsmtFin Type 2', 'Bsmt Exposure']:
    new_data[col]=new_data[col].fillna('NoBasement')

for col in ['Garage Type', 'Garage Finish', 'Garage Qual','Garage Cond']:
    new_data[col]=new_data[col].fillna("NoGarage")

new_data['Fireplace Qu'] = new_data['Fireplace Qu'].fillna("NoFireplace")
new_data['Electrical'] = new_data['Electrical'].fillna("SBrkr")
new_data['Fence'] = new_data['Fence'].fillna("NoFence")
new_data['Alley'] = new_data['Alley'].fillna("Noalleyaccess")
new_data['Pool QC'] = new_data['Pool QC'].fillna("NoPool")


# Check if there are still categorical features with NaNs
nan_object =new_data.select_dtypes(include =["object"])
nan_object = nan_object.isnull().sum()
nan_object= nan_object[nan_object !=0]
nan_object


# Checking the numerical features with missing values

# In[212]:


#Checking the numerical features with missing values
data_na = new_data.select_dtypes(include =np.number)
data_na = data_na.isnull().sum()
data_na = data_na[data_na !=0]
data_na


# In[213]:


# Let's calculate the most frequent value for each column.
replacement_values_dict = new_data[data_na.index].mode().to_dict(
    orient='records')[0]
replacement_values_dict


# In[214]:


# Replace missing values with the most frequent value of the corresponding column.
new_data= new_data.fillna(replacement_values_dict)

## Check that all columns have 0 missing values
new_data.isnull().sum().value_counts()


# ### ***Convert categorical features type to "category" type***
# 
# Let's check the categorical features before conversion
# 
# [Reference](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/)

# In[215]:


new_data.select_dtypes(include = "object").head()


# In[216]:


new_data.select_dtypes(include = "object").columns


# In[217]:


print("Shape data with the type object :\n", new_data.select_dtypes(include = "object").shape)


# In[218]:


#Description of categorical features 
info_func(new_data.select_dtypes(include = "object").columns)


# In[219]:


# How many unique values in each categorical column?
unique_values = new_data.select_dtypes(include = "object").apply(lambda cols: len(cols.value_counts())).sort_values()
unique_values


# The X function contains up to 28 different categories, that is, we will have 28 functions once this function is converted to a numeric variable. For this reason, we limit the different categories to 10. Of course, we could have kept all the categories, but we are just experimenting with the possibilities. 

# In[220]:


# Arbitrary limit of 10 unique values
drop_cate_cols = unique_values[unique_values > 10].index
print(drop_cate_cols, "\n")


# In[221]:


#Delete this features : 'Exterior 1st', 'Exterior 2nd', 'Neighborhood'
data_transform = new_data.drop(drop_cate_cols, axis=1)
print(data_transform.shape)
data_transform.head()


# In[222]:


#shape of categorical columns 
data_transform.select_dtypes(include=['object']).shape


# In[223]:


# Select only the remaining text columns and convert them to categories
objetc_cols = data_transform.select_dtypes(include=['object'])
for col in objetc_cols:
    data_transform[col] = data_transform[col].astype('category')


# In[224]:


data_transform.info()


# In[225]:


# Create dummy columns and add them to the DataFrame
data_transform = pd.concat([
    data_transform,
    pd.get_dummies(data_transform.select_dtypes(include=['category']), drop_first=True)
], axis=1)


# In[226]:


data_transform.shape


# In[227]:


data_transform.head()


# In[228]:


#Delete categorical columns
categorical_columns = data_transform.select_dtypes(include=['category']).columns

data_transform = data_transform.drop(categorical_columns, axis = 1)


# In[229]:


data_transform.head()


# In[230]:


data_transform.shape


# ### ***Training / Test split***

# In[231]:


from sklearn.model_selection import train_test_split


# ***Let's set the x and y variables to the feature and label values***

# In[232]:


X =data_transform.drop("SalePrice", axis = 1)
y =data_transform["SalePrice"]


# In[233]:


X.shape


# In[234]:


y.shape


# ***Train / test split with test_size=0.3 and a random_state of 42***

# In[235]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.30, random_state=42)


# In[236]:


X_train.shape


# In[237]:


X_test.shape


# ### ***Normalization***
# 
# Normalization reduces outliers by compressing values between an accurate scale.
# 
# [Reference](https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029)
# 
# [Reference](https://developers.google.com/machine-learning/data-prep/transform/normalization)

# In[238]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[239]:


X_train.shape


# In[240]:


X_test.shape


# ## Section 4 :  Modelization
# 

# In[241]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# ### ***Lineaire Regression***
# 
# [Reference](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

# In[246]:


# Entrainement
np.random.seed(0)
model = LinearRegression()
model.fit(X_train, y_train.values)
    
# Prédiction
predictions = model.predict(X_test)
mse = mean_squared_error(y_test.values, predictions)
rmse = np.sqrt(mse)
print(rmse)


# We have a very very high RMSE, which proves the inefficiency of our model which is in a situation of under-learning. We will apply a feature selection technique to see if there will be an improvement.

# ### ***Features selection***
# 
# The feature selection allows according to certain techniques to remove the least information features and which aims to improve the performance of machine learning models. 
# 
# 
# 
# [Reference](https://scikit-learn.org/stable/modules/feature_selection.html)
# 
# [Reference](https://medium.com/nerd-for-tech/removing-constant-variables-feature-selection-463e2d6a30d9)

# In[247]:


#We use a threshold of 0.01 to remove all features with zero and near-zero variance. 

from sklearn.feature_selection import VarianceThreshold
var_thr = VarianceThreshold(threshold = 0.01) 
var_thr.fit(X)

var_thr.get_support()


# - True : High Variance
# 
# - False : Low Variance

# In[248]:


# The low Variance features 
concol = [column for column in X.columns 
          if column not in X.columns[var_thr.get_support()]]

for features in concol:
    print(features)


# In[249]:


# Delete the low Variance features :
data_transform_select = data_transform.drop(concol,axis=1)
data_transform_select.shape


# In[250]:


X =data_transform_select.drop("SalePrice", axis = 1)
y =data_transform_select["SalePrice"]


# In[252]:


y.shape


# ### ***Lineaire Regression with the high Variance features***

# In[254]:


X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.30, random_state=42)


# In[255]:


X_train.shape


# In[257]:


X_test.shape


# In[258]:


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[261]:


# Entrainement
np.random.seed(0)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train.values)
    
# Prédiction
lr_predictions = lr_model.predict(X_test)
mse = mean_squared_error(y_test.values, lr_predictions)
lr_rmse = np.sqrt(mse)
print(lr_rmse)


# In[262]:


data_pred =pd.DataFrame({"y_test" : y_test, "lr_predictions" : lr_predictions})


# In[263]:


plt.figure(figsize=(8, 6))
sns.regplot(x = "lr_predictions", y ="y_test",  scatter_kws={'alpha': 0.4}, 
                line_kws={'color': 'red','linewidth':0.8} ,data = data_pred)
plt.title("Linear Regression")


# We have significantly reduced the rmse, but we are left with another problem: outliers related to the sale price. We will use the logarithmic technique to scale the selling price.

# In[264]:


# SalePrice after transformation
fig, ax = plt.subplots(figsize=(16,6), ncols=2)
sns.histplot(x = "SalePrice", bins=50, kde=True, ax=ax[0], data =data)
ax[0].set_title('Sale Price', fontsize=20)
ax[0].set_xlabel('Sale Price')
ax[0].set_ylabel('Number of Sales')

sns.histplot(x="lr_predictions", kde=True, ax=ax[1], data=data_pred)
ax[1].set_title("Histogram of Predicted Sale Price", fontsize=20)
plt.xlabel('Sale Price')
plt.ylabel('Number of Sales');
plt.savefig('pred_hist.png', bbox_inches="tight")


# We notice that the price prediction histogram is better adjusted (the values are well distributed).

# ### ***Random Forest Regressor***
# 
# [Reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

# In[265]:


rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(X_train, y_train.values)

rf_predictions = rf_model.predict(X_test)

mse = mean_squared_error(y_test.values, rf_predictions)

rf_rmse = np.sqrt(mse)

print(rf_rmse)


# In[266]:


data_pred =pd.DataFrame({"y_test" : y_test, "rf_predictions" : rf_predictions})


# In[267]:


plt.figure(figsize=(8, 6))
sns.regplot(x = "rf_predictions", y ="y_test",  scatter_kws={'alpha': 0.4}, 
                line_kws={'color': 'red','linewidth':0.8} ,data = data_pred)
plt.title("Random Forest Regression")


# ### ***Gradient Boosting Regressor***
# 
# [Reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

# In[268]:


gbr_model = GradientBoostingRegressor(random_state=0)
gbr_model.fit(X_train, y_train.values)

gbr_predictions = gbr_model.predict(X_test)

mse = mean_squared_error(y_test.values, gbr_predictions)
gbr_rmse = np.sqrt(mse)
print(gbr_rmse)


# In[269]:


data_pred =pd.DataFrame({"y_test" : y_test, "gbr_predictions" : gbr_predictions})


# In[270]:


plt.figure(figsize=(8, 6))
sns.regplot(x = "gbr_predictions", y ="y_test",  scatter_kws={'alpha': 0.4}, 
                line_kws={'color': 'red','linewidth':0.8} ,data = data_pred)
plt.title("Gradient Boosting Regression")


# ### ***Delete outliers values of sale price***
# 
# [Reference](https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html)

# In[271]:


y_log = y.copy() 
target_log = np.log1p(y)

data_target_log =pd.Series({"target_log":target_log})
data_target_log


# In[272]:


# SalePrice after transformation
fig, ax = plt.subplots(figsize=(16,6), ncols=2)
sns.histplot(x = "SalePrice", bins=50, kde=True, ax=ax[0], data =data)
ax[0].set_title('Sale Price before transformation', fontsize=20)
ax[0].set_xlabel('Sale Price')
ax[0].set_ylabel('Number of Sales')

sns.histplot(x = "target_log", bins=50, kde=True, ax=ax[1], data =data_target_log)
ax[1].set_title('Sale Price after transformation', fontsize=20)
ax[1].set_xlabel('Sale Price')
ax[1].set_ylabel('Number of Sales')


# ### ***Linear Regression after logarithmic transformation of the sale price***

# In[273]:


X_train, X_test, y_train, y_test_log = train_test_split(
                        X, target_log, test_size=0.30, random_state=42)


# In[274]:


# Entrainement
lr_model_log = LinearRegression()
lr_model_log.fit(X_train, y_train.values)
    
# Prédiction
lr_predictions_log = lr_model_log.predict(X_test)
mse = mean_squared_error(y_test_log.values, lr_predictions_log )
lr_rmse_log = np.sqrt(mse)
print(round(lr_rmse_log, 3)*100)


# In[275]:


data_log =pd.DataFrame({"y_test_log" : y_test_log, "lr_predictions_log" : lr_predictions_log})


# In[276]:


plt.figure(figsize=(8, 6))
sns.regplot(x = "lr_predictions_log", y ="y_test_log",  scatter_kws={'alpha': 0.4}, 
                line_kws={'color': 'red','linewidth':0.8} ,data = data_log)
plt.title("Linear Regression after log transformation of sale price")


# In[277]:


# SalePrice after transformation
fig, ax = plt.subplots(figsize=(16,6), ncols=2)
sns.histplot(x = "SalePrice", bins=50, kde=True, ax=ax[0], data =data)
ax[0].set_title('Sale Price', fontsize=20)
ax[0].set_xlabel('Sale Price')
ax[0].set_ylabel('Number of Sales')

sns.histplot(x="lr_predictions_log", kde=True, ax=ax[1], data=data_log)
ax[1].set_title("Histogram of Predicted Sale Price", fontsize=20)
plt.xlabel('Sale Price')
plt.ylabel('Number of Sales');
plt.savefig('pred_hist.png', bbox_inches="tight")


# We notice that the price prediction histogram is better and better adjusted (the values are well distributed) after the logarithmic transformation.

# ### ***Random Forest Regressor after logarithmic transformation of the sale price***

# In[278]:


rf_model_log= RandomForestRegressor(random_state=0)

rf_model_log.fit(X_train, y_train.values)

rf_predictions_log = rf_model_log.predict(X_test)

mse = mean_squared_error(y_test_log.values, rf_predictions_log)
rf_rmse_log = np.sqrt(mse)
print(round(rf_rmse_log, 3)*100)


# In[279]:


data_log =pd.DataFrame({"y_test_log" : y_test_log, "rf_predictions_log" : rf_predictions_log})


# In[280]:


plt.figure(figsize=(8, 6))

sns.regplot(x = "rf_predictions_log", y ="y_test_log",  scatter_kws={'alpha': 0.4}, 

                line_kws={'color': 'red','linewidth':0.8} ,data = data_log)
plt.title("Random Forest Regression after log transformation of sale price")


# ### ***Gradient Boosting Regressor after logarithmic transformation of the sale price***

# In[281]:


gbr_model_log= GradientBoostingRegressor(random_state=0)

gbr_model_log.fit(X_train, y_train.values)

gbr_predictions_log = gbr_model_log.predict(X_test)

mse = mean_squared_error(y_test_log.values, gbr_predictions_log)

gbr_rmse_log  = np.sqrt(mse)

print(round(gbr_rmse_log , 3)*100)


# In[282]:


data_pred_log =pd.DataFrame({"y_test_log" : y_test_log, "gbr_predictions_log" : gbr_predictions_log})


# In[283]:


plt.figure(figsize=(8, 6))

sns.regplot(x = "gbr_predictions_log", y ="y_test_log",  scatter_kws={'alpha': 0.4}, 

                line_kws={'color': 'red','linewidth':0.8} ,data = data_pred_log)
plt.title("Gradient Boosting Regression after log transformation of sale price")


# As you can see, scaling the sale price reduced the rate significantly to 12.5%.

# In[284]:


data_models = pd.DataFrame({"Linear Regression" : [lr_rmse, lr_rmse_log], "Random Forest Regressor": [rf_rmse, rf_rmse_log], 
                            "Gradient Boosting Regressor" : [gbr_rmse, gbr_rmse_log]}, index = ["rmse", "rmse_log"])
data_models


# Looking at the RMSEs of the different models, we notice that the ensemble models are less affected by outliers related to the selling price, but as soon as we use the natural logarithmic transformation to transform the selling price, the linear regression becomes better than all models d'ensemble.

# In[285]:


data_preds = pd.DataFrame({"y_test" : y_test, "lr_predictions": lr_predictions , "rf_predictions": rf_predictions, 
                           "gbr_predictions " : gbr_predictions , "y_test_log" : y_test_log, "lr_predictions_log" : lr_predictions_log, 
                           "rf_predictions_log" : rf_predictions_log, "gbr_predictions_log" : gbr_predictions_log})
data_preds.reset_index(drop = True).head()


# The predictions after the logarithmic transformation of the selling price approximate true selling price values (y_test_log).

# ### ***Predict the price of a house taken randomly from the database data_transform_select***
# 
# Let's take a random house from the data_transform_select database and predict its price and compare it to its current selling price.

# In[402]:


import random
random.seed(101)
random_ind = random.randint(0,len(data_transform_select))

single_house = data_transform_select.drop('SalePrice',axis=1).iloc[random_ind]
single_house


# In[403]:


single_house = scaler.transform(single_house.values.reshape(-1, 139))


# In[411]:


gbr_model.predict(single_house)


# In[405]:


data_transform_select.iloc[random_ind]["SalePrice"]


# We were predicting a selling price of 253913 dollars for this house while its actual selling price is 252000 dollars. We are not far from the actual price. If we continue to choose another house at random, we will necessarily find a price almost equal to the actual sale price.

# ## Conclusion

# To push this work, we can adjust the hyperparameters of different models, increase the variance threshold for entity selection, adjust other regression models or artificial neural networks to see if we are further improving the performance of our models.

