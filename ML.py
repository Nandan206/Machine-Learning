import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import scipy.stats as stats

import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, clport, f1_score, precision_score, recall_score, roc_auc_score, roc_curveassification_re

import warnings
warnings.simplefilter('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

pd.set_option('display.float_format', lambda x: '%.2f' % x)

sns.set_style('whitegrid')

%matplotlib inline

df = pd.read_csv(r'/kaggle/input/road-accident-united-kingdom-uk-dataset/UK_Accident.csv', parse_dates=['Date', 'Time'])

df.columns

df.drop(columns=['Unnamed: 0', 'Location_Easting_OSGR', 'Location_Northing_OSGR', 
                 'Local_Authority_(Highway)', 'LSOA_of_Accident_Location'], inplace=True)

df.sample(5)

print("No. of rows: {}".format(df.shape[0]))
print("No. of cols: {}".format(df.shape[1]))

df.info()
df.isna().any()
df.isnull().sum() / len(df) * 100
df.dropna(subset=['Longitude', 'Time', 'Pedestrian_Crossing-Human_Control', 
                  'Pedestrian_Crossing-Physical_Facilities'], inplace=True)
                  dup_rows = df[df.duplicated()]
print("No. of duplicate rows: ", dup_rows.shape[0])

df.drop_duplicates(inplace=True)
print("No. of rows remaining: ", df.shape[0])
df.describe(include=np.number)
df.describe(include=np.object)
numerical_data = df.select_dtypes(include='number')
num_cols = numerical_data.columns
len(num_cols)

categorical_data = df.select_dtypes(include='object')
cat_cols = categorical_data.columns
len(cat_cols)
sns.set(style="whitegrid")
fig = plt.figure(figsize=(20, 50))
fig.subplots_adjust(right=1.5)

for plot in range(1, len(num_cols)+1):
    plt.subplot(6, 4, plot)
    sns.boxplot(y=df[num_cols[plot-1]])

plt.show()
%time

import scipy.stats as stats
def diagnostic_plot(data, col):
    fig = plt.figure(figsize=(20, 5))
    fig.subplots_adjust(right=1.5)
    
    plt.subplot(1, 3, 1)
    sns.distplot(data[col], kde=True, color='teal')
    plt.title('Histogram')
    
    plt.subplot(1, 3, 2)
    stats.probplot(data[col], dist='norm', fit=True, plot=plt)
    plt.title('Q-Q Plot')
    
    plt.subplot(1, 3, 3)
    sns.boxplot(data[col],color='teal')
    plt.title('Box Plot')
    
    plt.show()
    dist_lst = ['Police_Force', 'Accident_Severity',
            'Number_of_Vehicles', 'Number_of_Casualties', 
            'Local_Authority_(District)', '1st_Road_Class', '1st_Road_Number',
            'Speed_limit', '2nd_Road_Class', '2nd_Road_Number',
            'Urban_or_Rural_Area']

for col in dist_lst:
    diagnostic_plot(df, col)
plt.figure(figsize = (15,10))
corr = df.corr(method='spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))
cormat = sns.heatmap(corr, mask=mask, annot=True, cmap='YlGnBu', linewidths=1, fmt=".2f")
cormat.set_title('Correlation Matrix')
plt.show()

def get_corr(data, threshold):
    corr_col = set()
    cormat = data.corr()
    for i in range(len(cormat.columns)):
        for j in range(i):
            if abs(cormat.iloc[i, j])>threshold:
                col_name = cormat.columns[i]
                corr_col.add(col_name)
    return corr_col

corr_features = get_corr(df, 0.80)
print(corr_features)

df.drop(columns=['Local_Authority_(District)'], 
        axis=1, inplace=True)

        def pie_chart(data, col):

  x = data[col].value_counts().values
  plt.figure(figsize=(7, 6))
  plt.pie(x, center=(0, 0), radius=1.5, labels=data[col].unique(), 
          autopct='%1.1f%%', pctdistance=0.5)
  plt.axis('equal')
  plt.show()

pie_lst = ['Did_Police_Officer_Attend_Scene_of_Accident']
for col in pie_lst:
  pie_chart(df, col)

  def cnt_plot(data, col):

  plt.figure(figsize=(15, 7))
  ax1 = sns.countplot(x=col, data=data,palette='rainbow')

  for p in ax1.patches:
    ax1.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1), ha='center')

  plt.show()

  print('\n')

cnt_lst1 = ['Road_Type', 'Junction_Control',
           'Pedestrian_Crossing-Human_Control',
           'Road_Surface_Conditions']

for col in cnt_lst1:
  cnt_plot(df, col)

  def cnt_plot(data, col):

  plt.figure(figsize=(10, 7))
  sns.countplot(y=col, data=data,palette='rainbow')
  plt.show()

  print('\n')
  
cnt_lst2 = ['Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions',
            'Weather_Conditions',
            'Special_Conditions_at_Site', 'Carriageway_Hazards']

for col in cnt_lst2:
  cnt_plot(df, col)
  df.sample(5)

df['Urban_or_Rural_Area'].value_counts()
df['Urban_or_Rural_Area'].replace(3, 1, inplace=True)

df['Accident_Severity'].value_counts()

df['Number_of_Vehicles'].value_counts()[:10]

df['Number_of_Casualties'].value_counts()[:10]

dt1 = df.groupby('Date')['Accident_Index'].count()\
.reset_index()\
.rename(columns={'Accident_Index':'No. of Accidents'})

fig = px.line(dt1, x='Date', y='No. of Accidents',
              labels={'index': 'Date', 'value': 'No. of Accidents'})
fig.show()

dt2 = df.groupby('Year')['Accident_Index'].count()\
.reset_index()\
.rename(columns={'Accident_Index':'No. of Accidents'})

fig = px.line(dt2, x='Year', y='No. of Accidents',
              labels={'index': 'Year', 'value': 'No. of Accidents'})
fig.show()dt1 = df.groupby('Date')['Accident_Index'].count()\
.reset_index()\
.rename(columns={'Accident_Index':'No. of Accidents'})

fig = px.line(dt1, x='Date', y='No. of Accidents',
              labels={'index': 'Date', 'value': 'No. of Accidents'})
fig.show()

dt3 = df.groupby('Day_of_Week')['Accident_Index'].count()\
.reset_index()\
.rename(columns={'Accident_Index':'No. of Accidents'})

fig = px.line(dt3, x='Day_of_Week', y='No. of Accidents',
              labels={'index': 'Day_of_Week', 'value': 'No. of Accidents'})
fig.show()

cat_cols
len(df['Accident_Index'].unique())
df.drop('Accident_Index',axis=1,inplace=True)

df.head()
plt.figure(figsize=(18, 5))
sns.heatmap(df.isna(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

X = df.drop(columns=['Accident_Severity'], axis=1)

plt.figure(figsize=(8, 10))
X.corrwith(df['Accident_Severity']).plot(kind='barh', 
                               title="Correlation with 'Convert' column -")
plt.show()

cat_cols=[feature for feature in df.columns if df[feature].dtype=='O']
print(cat_cols)
for feature in cat_cols:
    print(f'The {feature} has following number of {len(df[feature].unique())}')

    from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()

for feature in cat_cols:
    df[feature]=labelencoder.fit_transform(df[feature])
df.head()
df.drop('Year',axis=1,inplace=True)
df["day"] = df['Date'].map(lambda x: x.day)
df["month"] = df['Date'].map(lambda x: x.month)
df["year"] = df['Date'].map(lambda x: x.year)
df.head()
df.drop("Date",axis=1,inplace=True)
df.drop("Time",axis=1,inplace=True)
df['Accident_Severity']=df['Accident_Severity'].map({1:0,2:1,3:2})
dfnew=df[['Latitude','Longitude','day','month','1st_Road_Number','year','Day_of_Week','Accident_Severity']]
df.columns
dfnew.head()

from sklearn.preprocessing import StandardScaler
features = [feature for feature in dfnew.columns if feature!='Accident_Severity']
x = dfnew.iloc[0:50000, :-1]
y = dfnew.iloc[0:50000,[-1]]
x = StandardScaler().fit_transform(x)

%%time
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split


from imblearn.over_sampling import SMOTE
oversample = RandomOverSampler()
x, y = oversample.fit_resample(x, y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier()
clf.fit(x_train, y_train)

preds=clf.predict(x_test)

from sklearn.metrics import accuracy_score,classification_report
score=accuracy_score(y_test,preds)
print(score)

fig = plt.figure(figsize=(15,5))
conmat = confusion_matrix(y_test, preds)
sns.heatmap(conmat, annot=True, cbar=False)
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_test,preds))
