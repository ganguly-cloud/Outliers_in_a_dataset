import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('file.csv')
print df.head()

print df.isnull().sum()
'''
age        0
income     1
expence    0
dtype: int64'''

# null value is replacing  with median

df['income'].fillna((df['income'].median()),inplace=True)

#print df
print df.isnull().sum()
'''
age        0
income     0
expence    0
dtype: int64'''

print df.describe()
'''
              age        income       expence
count   14.000000     14.000000     14.000000
mean    48.500000  65357.142857  25285.714286
std     25.928155  17032.128657  11795.883059
min     25.000000  35000.000000  11000.000000
25%     30.250000  56000.000000  18500.000000
50%     48.000000  70000.000000  24500.000000
75%     55.750000  75750.000000  29500.000000
max    125.000000  90000.000000  60000.000000'''

print pd.DataFrame(df['age']).describe(percentiles=(1,0.99,0.9,0.75,0.5,0.1,0.01))

'''
              age
count   14.000000
mean    48.500000
std     25.928155
min     25.000000
1%      25.130000
10%     26.300000
50%     48.000000
75%     55.750000
90%     61.400000
99%    116.810000
100%   125.000000
max    125.000000 '''

# boxplot to visualize the data

plt.boxplot(df['age'])
#plt.savefig('boxplot_before_age_data_cleaning')
plt.show()

''' Here starts how to check outliers and solution of it
What is an outlier?

--> An outlier is a data point in a data set that is which is far away from all other observations.
   A data point that lies outside the overall distribution of the dataset.

What are the criteria to identify an outlier?

--> Data point that falls outside of 1.5 times of an interquartile range above the 3rd quartile and below the 1st quartile
--> Data point that falls outside of 3 standard deviations. we can use a z score and if the z score falls outside of 2 standard deviation

What is the reason for an outlier to exists in a dataset?

--> Variability in the data
--> An experimental measurement error

What are the impacts of having outliers in a dataset?
--> It causes various problems during our statistical analysis
--> It may cause a significant impact on the mean and the standard deviation

Various ways of finding the outlier.

--> Here am using box plot to check an outlier

'''
#checking outliers by definition and treating outliers

# getting median age

age_col_df = pd.DataFrame(df['age'])
age_median = age_col_df.median()
print age_col_df   # will print age column
print age_median
'''
age    48.0
dtype: float64'''

# Getting IQR[Inter quartile range] of age column i.e.Q3-Q1
Q3 = age_col_df.quantile(q=.75)  # to get the 75th percentile value
print 'Quartile 3 :',q3
'''
Quartile 3 : age    55.75
Name: 0.75, dtype: float64'''

Q1 = age_col_df.quantile(q=.25) # to get the 25th percentile value
print 'Quartile 1 : ',q1
'''
Quartile 1 : age    30.25
Name: 0.25, dtype: float64'''
IQR = Q3-Q1
print IQR
'''
age    25.5
dtype: float64'''
# Deriving Boundries of outliers

IQR_LL = int(Q1 - 1.5*IQR)
IQR_UL = int(Q3 + 1.5*IQR)

print IQR_LL  # -8
print IQR_UL  # 94
print

# finding and treating outliers - both lower and upper end
   # we can take 95% also 96,97 or 98 based on problem
df.loc[df['age'] > IQR_UL, 'age'] = int(age_col_df.quantile(q=.99))# loc:locate
df.loc[df['age'] < IQR_LL, 'age'] = int(age_col_df.quantile(q=.01))
print int(age_col_df.quantile(q=.99)) # 116
print df.loc[df['age'] > IQR_UL, 'age']
'''
7    116
Name: age, dtype: int64'''

print df.loc[df['age'] < IQR_LL, 'age']  # Series([], Name: age, dtype: int64)
# now check the max value in age column
print df['age']
'''
0      25
1      26
2      27
3      32
4      31
5      30
6      47
7     116
8      49
9      55
10     54
11     56
12     60
13     62
Name: age, dtype: int64'''
print max(df['age'])# 116 that is 99th percentile value is from previous column
## Box plot after setting outliers
plt.boxplot(df['age'])
plt.show()

x = df['income']
y = df['expence']

plt.scatter(x,y,label ='income_expence')
#plt.savefig('income_vs_expence')
plt.show()

# plot the corelation matrix
   # it is to check the strength of variation b/w two variables 

correlation_matrix = df.corr().round(2)
f,ax = plt.subplots(figsize =(8,4))
sns.heatmap(data = correlation_matrix,annot =True)
plt.savefig('correlation_representation_heatmap')
plt.show()

''' Feature Engineering '''


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
#print scaled_data
'''
[[0.         0.09090909 0.18367347]
 [0.01098901 0.         0.14285714]
 [0.02197802 1.         1.        ]
 [0.07692308 0.63636364 0.34693878]
 [0.06593407 0.72727273 0.42857143]
 [0.05494505 0.65454545 0.3877551 ]
 [0.24175824 0.63636364 0.28571429]
 [1.         0.74545455 0.3877551 ]
 [0.26373626 0.38181818 0.10204082]
 [0.32967033 0.58181818 0.18367347]
 [0.31868132 0.81818182 0.26530612]
 [0.34065934 0.92727273 0.28571429]
 [0.38461538 0.38181818 0.08163265]
 [0.40659341 0.14545455 0.        ]]'''

df_scaled = pd.DataFrame(scaled_data)
print df_scaled

'''
           0         1         2
0   0.000000  0.090909  0.183673
1   0.010989  0.000000  0.142857
2   0.021978  1.000000  1.000000
3   0.076923  0.636364  0.346939
4   0.065934  0.727273  0.428571
5   0.054945  0.654545  0.387755
6   0.241758  0.636364  0.285714
7   1.000000  0.745455  0.387755
8   0.263736  0.381818  0.102041
9   0.329670  0.581818  0.183673
10  0.318681  0.818182  0.265306
11  0.340659  0.927273  0.285714
12  0.384615  0.381818  0.081633
13  0.406593  0.145455  0.000000'''

df_scaled.columns = ['age','income','expence']
print df_scaled.columns# Index([u'age', u'income', u'expence'], dtype='object')

features = ['age','income']
response = ['expence']  # because we are trying to predict expences

x1 = df_scaled[features]
y1 = df_scaled[response]

print x
'''
      income       age
0   0.090909  0.000000
1   0.000000  0.010989
2   1.000000  0.021978'''
print y
'''
     expence
0   0.183673
1   0.142857
2   1.000000
3   0.346939'''


x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size =.2,random_state=0)

print x_train
'''
      income       age
11  0.927273  0.340659
2   1.000000  0.021978'''
print y_train
'''
     expence
11  0.285714
2   1.000000
13  0.000000'''

print x_test
'''
     income       age
8  0.381818  0.263736
6  0.636364  0.241758
4  0.727273  0.065934'''
print y_test
'''
    expence
8  0.102041
6  0.285714
4  0.428571'''



model = LinearRegression()
model.fit(x_train,y_train)

# predicting the values on test data

pred = model.predict(x_test)
print pred
'''y= mx+c
our case : y =m1*x1 + m2*x2 +c

[[0.20244557]
 [0.35979061]
 [0.4694941 ]]'''
# checking accuracy

accu = model.score(x_test,y_test)

print accu  # 0.6782333705931674


print model.intercept_   # [0.06094365]  ; c-value
print model.coef_   #   [[ 0.59063665 -0.31855265]]  ; m1 & m2- values

'''
# creating equation manually

expence = (0.59063665*income - 0.31855265*age)+ 0.06094365
sub one value and matching result manually

expence = (0.59063665*0.381818 - 0.31855265*0.263736)+ 0.06094365 =0.2024
      =15000 approx.  '''




