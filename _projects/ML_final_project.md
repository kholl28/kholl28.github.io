---
layout: page
title: Menstrual Cycle Prediction Tool
description: with ML model
img: assets/img/12.jpg
importance: 1
category: school
related_publications: false
---

**Helpful Links**

[Research Article](https://epublications.marquette.edu/cgi/viewcontent.cgi?article=1002&context=data_nfp)

[Dataset](https://epublications.marquette.edu/data_nfp/7/)

[Dataset (Kaggle)](https://www.kaggle.com/datasets/nikitabisht/menstrual-cycle-data/data)


The present analysis' primary goal is to create a menstrual cycle tracker that can be used to predict the day a user's next period will start based on their demographic information and the lengths of their last two menstrual cycles.

The user's self-reported age and BMI will be used as secondary predictor variables that will adjust the final prediction outputted by the model.

Previous research has indicated that age and BMI has a statistically significant relationship with menstrual cycle length, in that women in different age groups and BMI categories experience significant differences in cycle length. Thus, the model will adjust its final prediction based on this User-inputted information.

This will be achieved through the use of a machine learning model that will be trained using data from a study which collected the menstrual cycle and demographic information of a large nationally-representative sample of women.

User-inputted information will be piped into the final machine learning model in order to provide the user with a predicted start date of their next period.




## Motivation

The motivation for this project is to create an accurate regression model that can be used to predict the start date of a person's next menstrual period (the target variable) based on their last three period dates, age, and BMI (the predictor variables).

There is no straightforward math that can be used to calculate *exactly* when someone's next period will start. For instance, the average menstrual cycle length is around 28 days, but this varies from person to person. It's very common to have cycles that vary by up to *five days* from one month to the next.

There is therefore often a lot of uncertainty and anxiety that can come from not knowing when one's next period will begin. Without using predictive models, a person's only way to keep track of their cycles is by writing the dates that each period occured for each month in a calendar. They could then predict that their next period would start in 28 days from the start date of their most recent period, but as mentiond previously, cycle length can naturally vary by about 5 days month to month.

Although no model can predict with 100% accuracy the exact start date of someone's next period, identifying patterns among individual people in the lengths of each of their cycles would provide a person with more accurate, informed information, reducing uncertainty and allowing people to be more prepared.


# Importing required libraries


```python
import warnings
warnings.filterwarnings('ignore')

# import kagglehub
import math
import numpy as np
import os
import pandas as pd
import pip
import random
import scipy.stats as st
import statistics
from scipy.stats import mstats
import scipy
import shutil
import torch
import json

# dates
from datetime import datetime, date, timedelta

# visualizations
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# pre-processing
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer # Needed for IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats.mstats import winsorize # winsorization of outliers
from sklearn.model_selection import cross_val_score # cross-validation performance
import statsmodels.api as sm # Cook's distance

# models
from sklearn.linear_model import LinearRegression # for control sample model
from sklearn.linear_model import ElasticNet # elasticnet regression model
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor # random forest
import xgboost as xgb # XGB
from sklearn.neighbors import KNeighborsRegressor # KNN

# model selection
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import learning_curve # testing for over/under fitting

from google.colab import files
```


```python
# seed everything
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

```

# Data Collection

The data used for this analysis comes from a 2013 study by [Fehring et al.](https://epublications.marquette.edu/cgi/viewcontent.cgi?article=1002&context=data_nfp) titled "Randomized comparison of two internet-supported natural family planning methods".

This study collected data of the menstrual cycles of 159 anonymous American women over 12 months, with each woman having approximately 10 cycles logged. For each cycle, information such as the length of the cycle, the overall mean cycle length, estimated day of ovulation, length of menstruation, etc. is logged.

The dataset used for this study is freely-available to the public via Marquette University's e-Publications site (which can be accessed [here](https://epublications.marquette.edu/data_nfp/7/)) and is in .csv format.

I will now upload the dataset from GitHub as a pandas DataFrame and take a glimpse at it using the ```head()``` function.




```python
# import dataset
df = pd.read_csv("https://github.com/kholl28/data/raw/refs/heads/main/FedCycleData071012.csv")

df.head()
```





  <div id="df-01c8ab57-9aa6-474a-94a0-c6faf3c61e90" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ClientID</th>
      <th>CycleNumber</th>
      <th>Group</th>
      <th>CycleWithPeakorNot</th>
      <th>ReproductiveCategory</th>
      <th>LengthofCycle</th>
      <th>MeanCycleLength</th>
      <th>EstimatedDayofOvulation</th>
      <th>LengthofLutealPhase</th>
      <th>FirstDayofHigh</th>
      <th>...</th>
      <th>Method</th>
      <th>Prevmethod</th>
      <th>Methoddate</th>
      <th>Whychart</th>
      <th>Nextpreg</th>
      <th>NextpregM</th>
      <th>Spousesame</th>
      <th>SpousesameM</th>
      <th>Timeattemptpreg</th>
      <th>BMI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nfp8122</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>29</td>
      <td>27.33</td>
      <td>17</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td>9</td>
      <td></td>
      <td></td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>21.254724111867</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nfp8122</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>27</td>
      <td></td>
      <td>15</td>
      <td>12</td>
      <td>13</td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>nfp8122</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>29</td>
      <td></td>
      <td>15</td>
      <td>14</td>
      <td></td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>nfp8122</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>27</td>
      <td></td>
      <td>15</td>
      <td>12</td>
      <td>13</td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>nfp8122</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>28</td>
      <td></td>
      <td>16</td>
      <td>12</td>
      <td>12</td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-01c8ab57-9aa6-474a-94a0-c6faf3c61e90')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-01c8ab57-9aa6-474a-94a0-c6faf3c61e90 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-01c8ab57-9aa6-474a-94a0-c6faf3c61e90');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>

    </div>
  </div>





```python
df.shape
```




    (1665, 80)



The current uncleaned dataframe has 1665 rows and 80 columns, with each row being a cycle for a single Client (study participant) and each column containing information about the cycle and the Client it belongs to.


```python
df.columns
```




    Index(['ClientID', 'CycleNumber', 'Group', 'CycleWithPeakorNot',
           'ReproductiveCategory', 'LengthofCycle', 'MeanCycleLength',
           'EstimatedDayofOvulation', 'LengthofLutealPhase', 'FirstDayofHigh',
           'TotalNumberofHighDays', 'TotalHighPostPeak', 'TotalNumberofPeakDays',
           'TotalDaysofFertility', 'TotalFertilityFormula', 'LengthofMenses',
           'MeanMensesLength', 'MensesScoreDayOne', 'MensesScoreDayTwo',
           'MensesScoreDayThree', 'MensesScoreDayFour', 'MensesScoreDayFive',
           'MensesScoreDaySix', 'MensesScoreDaySeven', 'MensesScoreDayEight',
           'MensesScoreDayNine', 'MensesScoreDayTen', 'MensesScoreDay11',
           'MensesScoreDay12', 'MensesScoreDay13', 'MensesScoreDay14',
           'MensesScoreDay15', 'TotalMensesScore', 'MeanBleedingIntensity',
           'NumberofDaysofIntercourse', 'IntercourseInFertileWindow',
           'UnusualBleeding', 'PhasesBleeding', 'IntercourseDuringUnusBleed',
           'Age', 'AgeM', 'Maristatus', 'MaristatusM', 'Yearsmarried', 'Wedding',
           'Religion', 'ReligionM', 'Ethnicity', 'EthnicityM', 'Schoolyears',
           'SchoolyearsM', 'OccupationM', 'IncomeM', 'Height', 'Weight',
           'Reprocate', 'Numberpreg', 'Livingkids', 'Miscarriages', 'Abortions',
           'Medvits', 'Medvitexplain', 'Gynosurgeries', 'LivingkidsM', 'Boys',
           'Girls', 'MedvitsM', 'MedvitexplainM', 'Urosurgeries', 'Breastfeeding',
           'Method', 'Prevmethod', 'Methoddate', 'Whychart', 'Nextpreg',
           'NextpregM', 'Spousesame', 'SpousesameM', 'Timeattemptpreg', 'BMI'],
          dtype='object')



# Cleaning the dataset

## Subsetting dataset columns

The current uncleaned DataFrame, ```df```, has some columns that contain information not needed for the present analysis, such as the Client's ethnicity, marital status, or intercouse information.

So, let's start by cleaning this dataset to make it fit for analysis.

More information on the columns used for the present analysis are listed below.


```python
# specifying which columns of df we want to keep
clean_df = df[["ClientID", "CycleNumber", "LengthofCycle", "Age", "Height", "Weight", "BMI"]]
```

### Column information

**The following information on column information comes directly from Fehring et al.'s 2013 study.**

**ClientID:** Randomized ID for client anonymity

- The inclusion criteria for female participants were that they needed to be between the age of 18 and 42 years, have a stated menstrual cycle length ranging between 21-42 days long, have no history of hormonal contraceptives for the past 3 months and, if post-breastfeeding, have experienced at least three cycles past weaning.

**CycleNumber:** study lasted 12-months for a potential total of 13 cycles tracked for each participant (one cycle approx. every month)

**LengthofCycle:** A menstrual cycle is defined as the time from the first day of a woman's period to the day before her next period.

- The length of the menstrual cycle varies from woman to woman, but the average is to have periods around every 28 days.

    ([American College of Obstetrics and Gynecologists. The Menstrual Cycle: Menstruation, Ovulation, and How Pregnancy Occurs.](https://www.acog.org/womens-health/infographics/the-menstrual-cycle))

**Demographic Variables:**

- Age
- Height
- Weight
- BMI (calculated from Height & Weight)


```python
clean_df["ClientID"].nunique()
```




    159



There are 159 unique Client IDs.

Thus, the sample size for this DataFrame is 159 women between the ages of 18 and 42 years old.


```python
clean_df.shape
```




    (1665, 7)



There are now 1,665 rows and 7 columns in our newly cleaned dataset.

## Changing class of variables

Let's change the variable classes for all of the columns other than "ClientID" to numeric so we can conduct descriptive statistics of them.

Let's first see the current variable types of all the columns within ```clean_df```.


```python
clean_df.dtypes
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ClientID</th>
      <td>object</td>
    </tr>
    <tr>
      <th>CycleNumber</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>LengthofCycle</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>object</td>
    </tr>
    <tr>
      <th>Height</th>
      <td>object</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>object</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>object</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> object</label>



Now, let's go ahead and convert all the columns besides ClientId to numeric.


```python
cols = ["LengthofCycle", "Age", "Height", "Weight", "BMI"]
clean_df[cols] = clean_df[cols].apply(pd.to_numeric, errors='coerce') # coerce cols to numeric

clean_df.dtypes # checking work
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ClientID</th>
      <td>object</td>
    </tr>
    <tr>
      <th>CycleNumber</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>LengthofCycle</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>Height</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>float64</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> object</label>



However, we don't want the "Age" column to be coded as float class. As you can see in the below input, the values of "Age" have decimal places at the end even though this is not necessary, since "Age" is almost always considered an integer variable.




```python
clean_df["Age"].unique()
```




    array([36., nan, 39., 29., 26., 25., 23., 33., 30., 31., 24., 27., 35.,
           37., 32., 38., 21., 34., 22., 28., 43., 41., 40., 42.])



Let's use the ```.astype()``` function to convert "Age" to an "Int64" type.


```python
clean_df["Age"] = clean_df["Age"].astype('Int64')
```


```python
clean_df["Age"].unique() # checking work
```




    <IntegerArray>
    [  36, <NA>,   39,   29,   26,   25,   23,   33,   30,   31,   24,   27,   35,
       37,   32,   38,   21,   34,   22,   28,   43,   41,   40,   42]
    Length: 24, dtype: Int64



Now all of the variables in our dataset are the right variable types!

## Changing empty values to nan

Now, let's see what unique values exist for all of the variables within our clean_df DataFrame.


```python
for cols in clean_df.columns:
    print(f"{cols} : \n {clean_df[cols].unique()} \n \n")
```

    ClientID : 
     ['nfp8122' 'nfp8114' 'nfp8109' 'nfp8107' 'nfp8106' 'nfp8024' 'nfp8020'
     'nfp8026' 'nfp8030' 'nfp8031' 'nfp8032' 'nfp8034' 'nfp8036' 'nfp8040'
     'nfp8041' 'nfp8042' 'nfp8043' 'nfp8045' 'nfp8046' 'nfp8047' 'nfp8049'
     'nfp8050' 'nfp8051' 'nfp8057' 'nfp8058' 'nfp8060' 'nfp8062' 'nfp8063'
     'nfp8064' 'nfp8066' 'nfp8068' 'nfp8069' 'nfp8072' 'nfp8073' 'nfp8074'
     'nfp8076' 'nfp8079' 'nfp8080' 'nfp8083' 'nfp8085' 'nfp8087' 'nfp8091'
     'nfp8094' 'nfp8099' 'nfp8100' 'nfp8101' 'nfp8102' 'nfp8110' 'nfp8113'
     'nfp8116' 'nfp8123' 'nfp8124' 'nfp8129' 'nfp8131' 'nfp8133' 'nfp8137'
     'nfp8140' 'nfp8143' 'nfp8144' 'nfp8149' 'nfp8150' 'nfp8152' 'nfp8154'
     'nfp8155' 'nfp8159' 'nfp8161' 'nfp8164' 'nfp8165' 'nfp8168' 'nfp8172'
     'nfp8173' 'nfp8174' 'nfp8176' 'nfp8177' 'nfp8178' 'nfp8179' 'nfp8184'
     'nfp8186' 'nfp8187' 'nfp8188' 'nfp8189' 'nfp8190' 'nfp8192' 'nfp8193'
     'nfp8195' 'nfp8196' 'nfp8197' 'nfp8200' 'nfp8206' 'nfp8207' 'nfp8209'
     'nfp8210' 'nfp8211' 'nfp8212' 'nfp8218' 'nfp8221' 'nfp8223' 'nfp8226'
     'nfp8228' 'nfp8229' 'nfp8230' 'nfp8233' 'nfp8234' 'nfp8235' 'nfp8236'
     'nfp8237' 'nfp8238' 'nfp8240' 'nfp8242' 'nfp8244' 'nfp8246' 'nfp8247'
     'nfp8248' 'nfp8249' 'nfp8252' 'nfp8253' 'nfp8254' 'nfp8257' 'nfp8260'
     'nfp8263' 'nfp8264' 'nfp8266' 'nfp8268' 'nfp8269' 'nfp8270' 'nfp8271'
     'nfp8272' 'nfp8276' 'nfp8278' 'nfp8279' 'nfp8281' 'nfp8282' 'nfp8284'
     'nfp8286' 'nfp8288' 'nfp8289' 'nfp8290' 'nfp8292' 'nfp8293' 'nfp8294'
     'nfp8296' 'nfp8298' 'nfp8299' 'nfp8302' 'nfp8303' 'nfp8305' 'nfp8306'
     'nfp8308' 'nfp8309' 'nfp8310' 'nfp8311' 'nfp8312' 'nfp8313' 'nfp8317'
     'nfp8322' 'nfp8323' 'nfp8324' 'nfp8328' 'nfp8334'] 
     
    
    CycleNumber : 
     [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45] 
     
    
    LengthofCycle : 
     [29 27 28 26 24 30 25 32 31 34 23 18 33 35 41 38 36 39 37 40 21 48 22 43
     45 54 42 20 44 49 19 51] 
     
    
    Age : 
     <IntegerArray>
    [  36, <NA>,   39,   29,   26,   25,   23,   33,   30,   31,   24,   27,   35,
       37,   32,   38,   21,   34,   22,   28,   43,   41,   40,   42]
    Length: 24, dtype: Int64 
     
    
    Height : 
     [63. nan 68. 66. 71. 65. 67. 72. 69. 70. 64. 62. 61. 60. 59.] 
     
    
    Weight : 
     [120.  nan 185. 180. 200. 150. 155. 137. 135. 136. 178. 106. 130. 187.
     168. 170. 175. 220.  95. 209. 110. 115. 141. 201. 160. 190. 145. 300.
     260. 194. 125. 179. 165. 189. 140. 122. 124. 157.   0. 268. 128. 214.
     161. 127. 138. 116. 132. 104. 195. 123. 117. 240. 152. 112. 146. 114.
     121.] 
     
    
    BMI : 
     [21.25472411         nan 28.12608131 29.04958678 27.89129141 24.95857988
     25.79053254 21.45488973 22.46272189 21.94857668 22.62911243 24.13850309
     18.7750063  19.19554715 29.95029586 29.28514146 24.10285714 30.11085916
     27.40588104 37.75878906 16.82665659 38.96699421 19.96686391 34.7756213
     29.17724609 20.11706556 20.59570312 18.00957897 25.84558824 25.7864204
     31.47761194 24.32525952 34.74765869 26.51795005 49.91715976 21.14167966
     22.31201172 38.39109431 25.82185491 20.02633701 23.17016602 35.47918835
     25.8244898  22.86030177 28.88820018 26.62878788 37.10277778 28.33963215
     34.56477627 24.79717813 21.60896951 26.62248521 18.85207612 26.25395001
     18.79260414 22.52469388 43.25160698 19.46020761 29.85651974 36.72900391
     24.36357908 26.78887574 20.49609734 35.42454019 20.52443772 20.37681159
     21.45385742 19.36639118 21.9458897  24.12662722 22.14870825 20.54623331
     18.55945822 20.67186456 19.64848159 27.97653061 25.74462891 22.88699853
     20.4660355  20.80306122 27.45401864 18.8822314  22.59412305 21.78719008
     33.46954969 25.8398307  20.17332415 21.28222656 19.73754883 19.93383743
     21.03147763 25.29136095 20.93896484 23.49075518 30.66345271 23.56509516
     20.48283039 20.94653061 23.02595112 24.68912591 25.10216227 30.26805556
     26.62285587 22.10996327 25.10186936 22.19679931 21.92470483 22.26166667
     25.68279164 34.32617188 21.63076923 24.43636886] 
     
    


It looks like we have quite a few missing values in the variables of our ``clean_df`` DataFrame.

It also looks like our ``Weight`` variable has some 0 values, which should be recoded as missing. Let's do that first.


```python
clean_df['Weight'] = clean_df['Weight'].replace(0.0, np.nan)
```

Now, let's find the **number** of missing values in all the columns.


```python
clean_df.isnull().sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ClientID</th>
      <td>0</td>
    </tr>
    <tr>
      <th>CycleNumber</th>
      <td>0</td>
    </tr>
    <tr>
      <th>LengthofCycle</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>1523</td>
    </tr>
    <tr>
      <th>Height</th>
      <td>1532</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>1532</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>1534</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



So, that's 4 columns with missing values and 3 columns without missing values.

However, it makes sense that the demographic columns ("Age", "Height", "Weight", "BMI) have missing values, since there is only one of these values for each Client in the dataset. That is to say, when the first row (ie., cycle) of a Client appears, this information for said Client is logged in that row, but for all subsequent cycles (rows) of that same Client, this information is left blank.  

Because of this, we only really need to worry about how many missing values are in these columns in our final dataframe which will be used to train the predictive model. We will return to this concern once the final dataframe has been created.

# Grouping CycleNumber columns by threes

As mentioned in the beginning of this analysis, the goal is to create a menstrual cycle tracker that can be used to predict the day a user's next period will start based on information from their last two menstrual cycles.

Therefore, in order to train the model that will make these predictions, we need to find a way to extract 3 cycles from the same Client and convert these into columns within a new DataFrame.

Since some Clients had a larger number of reported cycle lengths than others, the number of times each Client will be repeated as a row in the new DataFrame will vary. However, each of these "triplets" will only come from a *single* Client to ensure that the model is accurately predicting menstrual cycle length using a single person as each sample.

-------------

These "triplets" will have overlap to create a larger training and testing set to use with our model. To illustrate, here is a visual example:

Cycle 1, Cycle 2, Cycle 3 = one triplet

Cycle 2, Cyle 3, Cycle 4 = one triplet

Cycle 3, Cycle 4, Cycle 5 = one triplet

Cycle 4, Cycle 5, Cycle 6 = one triplet

-----------------

In order to pivot our ``clean_df`` DataFrame into a new DataFrame wherein every row is a single client (``ClientID``) and there are columns for every ``CycleLength`` that make up a "triplet", we need to first create a new column within ``clean_df``, ``ClientCycleID``, which will contain a unique identifier for each row. This will allow us to use ``ClientID`` as the index in our ```clean_df.pivot_table()``` function.

Each value within the newly created column ``ClientCycleID`` is a combination of that row's ``ClientID`` and ``CycleNumber`` values, written as a string value.


```python
# create a new column "ClientCycleID" by combining "ClientID" and "CycleNumber"
clean_df["ClientCycleID"] = clean_df["ClientID"].astype(str) + "_" + clean_df["CycleNumber"].astype(str)

# shift newly created "ClientCycleID" column to front of clean_df DataFrame
# -----------------------------------------
first_column = clean_df.pop("ClientCycleID")
# insert column using insert(position,column_name, first_column) function
clean_df.insert(0, "ClientCycleID", first_column)

clean_df
```





  <div id="df-d38a71be-6dc5-4ccf-a460-6c234f49abf2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ClientCycleID</th>
      <th>ClientID</th>
      <th>CycleNumber</th>
      <th>LengthofCycle</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>BMI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nfp8122_1</td>
      <td>nfp8122</td>
      <td>1</td>
      <td>29</td>
      <td>36</td>
      <td>63.0</td>
      <td>120.0</td>
      <td>21.254724</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nfp8122_2</td>
      <td>nfp8122</td>
      <td>2</td>
      <td>27</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nfp8122_3</td>
      <td>nfp8122</td>
      <td>3</td>
      <td>29</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nfp8122_4</td>
      <td>nfp8122</td>
      <td>4</td>
      <td>27</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nfp8122_5</td>
      <td>nfp8122</td>
      <td>5</td>
      <td>28</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1660</th>
      <td>nfp8334_7</td>
      <td>nfp8334</td>
      <td>7</td>
      <td>29</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1661</th>
      <td>nfp8334_8</td>
      <td>nfp8334</td>
      <td>8</td>
      <td>28</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1662</th>
      <td>nfp8334_9</td>
      <td>nfp8334</td>
      <td>9</td>
      <td>28</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1663</th>
      <td>nfp8334_10</td>
      <td>nfp8334</td>
      <td>10</td>
      <td>40</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1664</th>
      <td>nfp8334_11</td>
      <td>nfp8334</td>
      <td>11</td>
      <td>24</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1665 rows × 8 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d38a71be-6dc5-4ccf-a460-6c234f49abf2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d38a71be-6dc5-4ccf-a460-6c234f49abf2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d38a71be-6dc5-4ccf-a460-6c234f49abf2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>

    </div>
  </div>




Now we can create the new DataFrame mentioned previously, where every row is a single ClientID and there is a column for every triplet.


```python
# get all the unique client ids
unique_client_ids = set(clean_df['ClientID'])

# create a new dataframe where we will store everything
new_df = pd.DataFrame({'ClientID' : [], 'Cycle_1' : [], 'Cycle_2' : [], 'Cycle_3' : []})

# loop through client ids
for client_id in unique_client_ids:
    # only take the portion of clean_df that belong to this client
    t1 = clean_df[clean_df['ClientID'] == client_id]
    # only take these two columns
    t1 = t1[['ClientID', 'LengthofCycle']]

    # loop through all the rows except last two
    for i in range(t1.shape[0]-2):
        # for every row, get that row and the two rows after that
        t2 = t1.iloc[i:(i+3)].copy()
        # create a new column
        t2['CycleID'] = ['Cycle_1', 'Cycle_2', 'Cycle_3']
        # reshape
        t3 = t2.pivot(index = 'ClientID', columns = 'CycleID', values = 'LengthofCycle')
        # add to the dataframe
        new_df = pd.concat([new_df, t3.reset_index()], axis = 0)

# check out
new_df
```





  <div id="df-4187519c-1a82-4451-a05c-e46939079da4" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ClientID</th>
      <th>Cycle_1</th>
      <th>Cycle_2</th>
      <th>Cycle_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nfp8294</td>
      <td>37.0</td>
      <td>40.0</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>nfp8294</td>
      <td>40.0</td>
      <td>39.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>nfp8294</td>
      <td>39.0</td>
      <td>30.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>nfp8294</td>
      <td>30.0</td>
      <td>29.0</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>nfp8294</td>
      <td>29.0</td>
      <td>35.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>nfp8246</td>
      <td>28.0</td>
      <td>28.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>nfp8159</td>
      <td>38.0</td>
      <td>38.0</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>nfp8159</td>
      <td>38.0</td>
      <td>42.0</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>nfp8159</td>
      <td>42.0</td>
      <td>37.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>nfp8159</td>
      <td>37.0</td>
      <td>30.0</td>
      <td>38.0</td>
    </tr>
  </tbody>
</table>
<p>1358 rows × 4 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-4187519c-1a82-4451-a05c-e46939079da4')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-4187519c-1a82-4451-a05c-e46939079da4 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-4187519c-1a82-4451-a05c-e46939079da4');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>

    </div>
  </div>




Success! Let's look at the shape of our ``new_df`` dataframe.


```python
new_df.shape
```




    (1358, 4)



Our new DataFrame, ``new_df``, which contains the triplet CycleLengths, has 1,358 rows and 4 columns.

## Adding secondary predictor variables

Since we want to use the demographic variables ``Age``, ``Height``, ``Weight``, and ``BMI`` as secondary variables for menstrual cycle prediction, let's create a new DataFrame, ``dems``, which contains only the ``ClientID`` and demographic columns of ``clean_df``.

We will need to preserve the ClientID column to use as a key during merging.





```python
dems = clean_df[["ClientID", "Age", "Height", "Weight", "BMI"]]

# convert to pandas DataFrame
dems = dems.reset_index()

dems
```





  <div id="df-7a3dd8f0-3ecf-4b6e-a1f4-ae8336567051" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>ClientID</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>BMI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>nfp8122</td>
      <td>36</td>
      <td>63.0</td>
      <td>120.0</td>
      <td>21.254724</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>nfp8122</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>nfp8122</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>nfp8122</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>nfp8122</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1660</th>
      <td>1660</td>
      <td>nfp8334</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1661</th>
      <td>1661</td>
      <td>nfp8334</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1662</th>
      <td>1662</td>
      <td>nfp8334</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1663</th>
      <td>1663</td>
      <td>nfp8334</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1664</th>
      <td>1664</td>
      <td>nfp8334</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1665 rows × 6 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7a3dd8f0-3ecf-4b6e-a1f4-ae8336567051')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7a3dd8f0-3ecf-4b6e-a1f4-ae8336567051 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7a3dd8f0-3ecf-4b6e-a1f4-ae8336567051');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>

    </div>
  </div>





```python
clean_df['Age'].count()
```




    np.int64(142)




```python
clean_df['Age'].isna().sum()
```




    np.int64(1523)




```python
dems['Age'].count()
```




    np.int64(142)




```python
dems['Age'].isna().sum()
```




    np.int64(1523)



During the creation of this dems DataFrame, we will drop rows where both ``Age``, ``Height``, ``Weight``, *and* ``BMI`` have missing values, since, as mentioned previously, demographic information is only listed in a single row for each Client.

This will ensure that only one row exists for each Client in our final dataframe.


```python
# # Drop rows where both 'Age', 'Height', and 'Weight' are missing
dems = dems.dropna(subset=['Age', 'Height', 'Weight', 'BMI'], how='all')

# checking work
# --------------
# dems.head()
# dems.shape # 1665, 4

dems
```





  <div id="df-81a1f976-e476-41df-a6b8-b1736995831c" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>ClientID</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>BMI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>nfp8122</td>
      <td>36</td>
      <td>63.0</td>
      <td>120.0</td>
      <td>21.254724</td>
    </tr>
    <tr>
      <th>45</th>
      <td>45</td>
      <td>nfp8114</td>
      <td>39</td>
      <td>68.0</td>
      <td>185.0</td>
      <td>28.126081</td>
    </tr>
    <tr>
      <th>47</th>
      <td>47</td>
      <td>nfp8109</td>
      <td>29</td>
      <td>66.0</td>
      <td>180.0</td>
      <td>29.049587</td>
    </tr>
    <tr>
      <th>50</th>
      <td>50</td>
      <td>nfp8107</td>
      <td>26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>58</th>
      <td>58</td>
      <td>nfp8106</td>
      <td>25</td>
      <td>71.0</td>
      <td>200.0</td>
      <td>27.891291</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>1345</td>
      <td>nfp8288</td>
      <td>30</td>
      <td>70.0</td>
      <td>195.0</td>
      <td>27.976531</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>1348</td>
      <td>nfp8289</td>
      <td>40</td>
      <td>65.0</td>
      <td>135.0</td>
      <td>22.462722</td>
    </tr>
    <tr>
      <th>1380</th>
      <td>1380</td>
      <td>nfp8290</td>
      <td>33</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1393</th>
      <td>1393</td>
      <td>nfp8292</td>
      <td>34</td>
      <td>66.0</td>
      <td>150.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1406</th>
      <td>1406</td>
      <td>nfp8293</td>
      <td>23</td>
      <td>63.0</td>
      <td>110.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>142 rows × 6 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-81a1f976-e476-41df-a6b8-b1736995831c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-81a1f976-e476-41df-a6b8-b1736995831c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-81a1f976-e476-41df-a6b8-b1736995831c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>

    </div>
  </div>





```python
dems['ClientID'].nunique()
```




    138



It looks like our sample of women was reduced from 159 to 138 when we subsetted ``dems`` to not include rows where all of the demographic columns had missing values.

From this, we can deduce that 23 of the women in the original sample did not have any demographic information recorded when Fehring et al. conducted the study.

Since it would be difficult to attempt to use an imputer to fill in missing information for four columns for each of these 23 women, it's best to just remove them fully from the study and use the 138 women we do have more complete data on.

Now, let's merge dems and new_df together into a single DataFrame.

We will be using a left merge in order to keep every row of new_df and discard any rows in dems that do not have a matching ClientID in new_df.

This new dataframe will contain the menstrual cycle triplets and secondary predictor variables of each client.


```python
final_df = new_df.merge(dems, on = ["ClientID"], how = "left")

final_df
```





  <div id="df-5aa7457d-f3b3-403c-a697-e3d483a9c82f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ClientID</th>
      <th>Cycle_1</th>
      <th>Cycle_2</th>
      <th>Cycle_3</th>
      <th>index</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>BMI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nfp8294</td>
      <td>37.0</td>
      <td>40.0</td>
      <td>39.0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nfp8294</td>
      <td>40.0</td>
      <td>39.0</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nfp8294</td>
      <td>39.0</td>
      <td>30.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nfp8294</td>
      <td>30.0</td>
      <td>29.0</td>
      <td>35.0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nfp8294</td>
      <td>29.0</td>
      <td>35.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1377</th>
      <td>nfp8246</td>
      <td>28.0</td>
      <td>28.0</td>
      <td>31.0</td>
      <td>1145.0</td>
      <td>22</td>
      <td>62.0</td>
      <td>112.0</td>
      <td>20.482830</td>
    </tr>
    <tr>
      <th>1378</th>
      <td>nfp8159</td>
      <td>38.0</td>
      <td>38.0</td>
      <td>42.0</td>
      <td>711.0</td>
      <td>33</td>
      <td>66.0</td>
      <td>268.0</td>
      <td>43.251607</td>
    </tr>
    <tr>
      <th>1379</th>
      <td>nfp8159</td>
      <td>38.0</td>
      <td>42.0</td>
      <td>37.0</td>
      <td>711.0</td>
      <td>33</td>
      <td>66.0</td>
      <td>268.0</td>
      <td>43.251607</td>
    </tr>
    <tr>
      <th>1380</th>
      <td>nfp8159</td>
      <td>42.0</td>
      <td>37.0</td>
      <td>30.0</td>
      <td>711.0</td>
      <td>33</td>
      <td>66.0</td>
      <td>268.0</td>
      <td>43.251607</td>
    </tr>
    <tr>
      <th>1381</th>
      <td>nfp8159</td>
      <td>37.0</td>
      <td>30.0</td>
      <td>38.0</td>
      <td>711.0</td>
      <td>33</td>
      <td>66.0</td>
      <td>268.0</td>
      <td>43.251607</td>
    </tr>
  </tbody>
</table>
<p>1382 rows × 9 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5aa7457d-f3b3-403c-a697-e3d483a9c82f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5aa7457d-f3b3-403c-a697-e3d483a9c82f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5aa7457d-f3b3-403c-a697-e3d483a9c82f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>

    </div>
  </div>




I will also go ahead and drop the "index" column that was created from the merge, as it is not necessary for analysis.



```python
final_df = final_df.drop(["index"], axis = 1)
```

# train test split

We will use ```train_test_split()``` to split our data into a training set and test set.

Since our dataset is relatively small (1,382 samples), we will want to use a slightly larger test size to ensure that our test set has enough data for reliable evaluation.

Thus, we will set the train_size parameter to 0.7 in order to extract 30% of the dataset for the testing set.


```python
seed_everything(42)

X = final_df[['Cycle_1', 'Cycle_2', 'Age', 'Height', 'Weight', 'BMI']] # denote all predictos with uppercase X
y = final_df['Cycle_3'] # denote target (outcome variable) with lowercase Y

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 42)
# test_size = 30% should go to the test set
```


```python
X_train
```





  <div id="df-46c5bdba-e8b8-4eeb-9275-23c42284b071" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cycle_1</th>
      <th>Cycle_2</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>BMI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>482</th>
      <td>27.0</td>
      <td>29.0</td>
      <td>32</td>
      <td>68.0</td>
      <td>135.0</td>
      <td>20.524438</td>
    </tr>
    <tr>
      <th>59</th>
      <td>23.0</td>
      <td>25.0</td>
      <td>37</td>
      <td>64.0</td>
      <td>130.0</td>
      <td>22.312012</td>
    </tr>
    <tr>
      <th>405</th>
      <td>27.0</td>
      <td>34.0</td>
      <td>32</td>
      <td>65.0</td>
      <td>152.0</td>
      <td>25.291361</td>
    </tr>
    <tr>
      <th>464</th>
      <td>40.0</td>
      <td>33.0</td>
      <td>22</td>
      <td>59.0</td>
      <td>130.0</td>
      <td>26.253950</td>
    </tr>
    <tr>
      <th>1303</th>
      <td>33.0</td>
      <td>27.0</td>
      <td>25</td>
      <td>68.0</td>
      <td>135.0</td>
      <td>20.524438</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>24.0</td>
      <td>24.0</td>
      <td>36</td>
      <td>63.0</td>
      <td>140.0</td>
      <td>24.797178</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>27.0</td>
      <td>28.0</td>
      <td>31</td>
      <td>68.0</td>
      <td>146.0</td>
      <td>22.196799</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>42.0</td>
      <td>43.0</td>
      <td>32</td>
      <td>65.0</td>
      <td>161.0</td>
      <td>26.788876</td>
    </tr>
    <tr>
      <th>860</th>
      <td>25.0</td>
      <td>27.0</td>
      <td>35</td>
      <td>66.0</td>
      <td>120.0</td>
      <td>19.366391</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>35.0</td>
      <td>28.0</td>
      <td>31</td>
      <td>68.0</td>
      <td>146.0</td>
      <td>22.196799</td>
    </tr>
  </tbody>
</table>
<p>967 rows × 6 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-46c5bdba-e8b8-4eeb-9275-23c42284b071')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-46c5bdba-e8b8-4eeb-9275-23c42284b071 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-46c5bdba-e8b8-4eeb-9275-23c42284b071');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>

    </div>
  </div>




``X_train`` has 967 rows and 6 columns.


```python
X_test
```





  <div id="df-f30c099f-64a4-457e-9f84-7e7a4524b77b" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cycle_1</th>
      <th>Cycle_2</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>BMI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>309</th>
      <td>25.0</td>
      <td>26.0</td>
      <td>29</td>
      <td>69.0</td>
      <td>138.0</td>
      <td>20.376812</td>
    </tr>
    <tr>
      <th>741</th>
      <td>36.0</td>
      <td>28.0</td>
      <td>31</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>265</th>
      <td>26.0</td>
      <td>28.0</td>
      <td>36</td>
      <td>63.0</td>
      <td>120.0</td>
      <td>21.254724</td>
    </tr>
    <tr>
      <th>823</th>
      <td>28.0</td>
      <td>29.0</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>778</th>
      <td>27.0</td>
      <td>25.0</td>
      <td>33</td>
      <td>65.0</td>
      <td>155.0</td>
      <td>25.790533</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>506</th>
      <td>29.0</td>
      <td>33.0</td>
      <td>25</td>
      <td>67.0</td>
      <td>170.0</td>
      <td>26.622856</td>
    </tr>
    <tr>
      <th>985</th>
      <td>25.0</td>
      <td>28.0</td>
      <td>34</td>
      <td>66.0</td>
      <td>150.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>266</th>
      <td>28.0</td>
      <td>28.0</td>
      <td>36</td>
      <td>63.0</td>
      <td>120.0</td>
      <td>21.254724</td>
    </tr>
    <tr>
      <th>327</th>
      <td>34.0</td>
      <td>30.0</td>
      <td>28</td>
      <td>60.0</td>
      <td>190.0</td>
      <td>37.102778</td>
    </tr>
    <tr>
      <th>348</th>
      <td>29.0</td>
      <td>27.0</td>
      <td>31</td>
      <td>64.0</td>
      <td>170.0</td>
      <td>29.177246</td>
    </tr>
  </tbody>
</table>
<p>415 rows × 6 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f30c099f-64a4-457e-9f84-7e7a4524b77b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f30c099f-64a4-457e-9f84-7e7a4524b77b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f30c099f-64a4-457e-9f84-7e7a4524b77b');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>

    </div>
  </div>




``X_test`` has 416 rows and 6 columns.


```python
y_train.isnull().sum()
```




    np.int64(0)



Since ``Cycle_3`` had no missing values in our ``final_df`` dataframe, our y_train dataset has no missing values either!


```python
y_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cycle_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>482</th>
      <td>28.0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>25.0</td>
    </tr>
    <tr>
      <th>405</th>
      <td>32.0</td>
    </tr>
    <tr>
      <th>464</th>
      <td>54.0</td>
    </tr>
    <tr>
      <th>1303</th>
      <td>29.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>23.0</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>30.0</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>31.0</td>
    </tr>
    <tr>
      <th>860</th>
      <td>26.0</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>30.0</td>
    </tr>
  </tbody>
</table>
<p>967 rows × 1 columns</p>
</div><br><label><b>dtype:</b> float64</label>



``y_train`` has 967 rows and 1 column, ``Cycle_3``, our target variable.


```python
y_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cycle_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>309</th>
      <td>26.0</td>
    </tr>
    <tr>
      <th>741</th>
      <td>32.0</td>
    </tr>
    <tr>
      <th>265</th>
      <td>28.0</td>
    </tr>
    <tr>
      <th>823</th>
      <td>27.0</td>
    </tr>
    <tr>
      <th>778</th>
      <td>24.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>506</th>
      <td>31.0</td>
    </tr>
    <tr>
      <th>985</th>
      <td>26.0</td>
    </tr>
    <tr>
      <th>266</th>
      <td>24.0</td>
    </tr>
    <tr>
      <th>327</th>
      <td>29.0</td>
    </tr>
    <tr>
      <th>348</th>
      <td>29.0</td>
    </tr>
  </tbody>
</table>
<p>415 rows × 1 columns</p>
</div><br><label><b>dtype:</b> float64</label>



``y_test`` has 415 rows and 1 column, ``Cycle_3``, our target variable.

# Exploratory Data Analysis

## Data structure

Let's see the structure of our final dataset, final_df.


```python
final_df
```





  <div id="df-f500abf6-bfa6-42b3-a084-a26b11d94b11" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ClientID</th>
      <th>Cycle_1</th>
      <th>Cycle_2</th>
      <th>Cycle_3</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>BMI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nfp8294</td>
      <td>37.0</td>
      <td>40.0</td>
      <td>39.0</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nfp8294</td>
      <td>40.0</td>
      <td>39.0</td>
      <td>30.0</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nfp8294</td>
      <td>39.0</td>
      <td>30.0</td>
      <td>29.0</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nfp8294</td>
      <td>30.0</td>
      <td>29.0</td>
      <td>35.0</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nfp8294</td>
      <td>29.0</td>
      <td>35.0</td>
      <td>29.0</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1377</th>
      <td>nfp8246</td>
      <td>28.0</td>
      <td>28.0</td>
      <td>31.0</td>
      <td>22</td>
      <td>62.0</td>
      <td>112.0</td>
      <td>20.482830</td>
    </tr>
    <tr>
      <th>1378</th>
      <td>nfp8159</td>
      <td>38.0</td>
      <td>38.0</td>
      <td>42.0</td>
      <td>33</td>
      <td>66.0</td>
      <td>268.0</td>
      <td>43.251607</td>
    </tr>
    <tr>
      <th>1379</th>
      <td>nfp8159</td>
      <td>38.0</td>
      <td>42.0</td>
      <td>37.0</td>
      <td>33</td>
      <td>66.0</td>
      <td>268.0</td>
      <td>43.251607</td>
    </tr>
    <tr>
      <th>1380</th>
      <td>nfp8159</td>
      <td>42.0</td>
      <td>37.0</td>
      <td>30.0</td>
      <td>33</td>
      <td>66.0</td>
      <td>268.0</td>
      <td>43.251607</td>
    </tr>
    <tr>
      <th>1381</th>
      <td>nfp8159</td>
      <td>37.0</td>
      <td>30.0</td>
      <td>38.0</td>
      <td>33</td>
      <td>66.0</td>
      <td>268.0</td>
      <td>43.251607</td>
    </tr>
  </tbody>
</table>
<p>1382 rows × 8 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f500abf6-bfa6-42b3-a084-a26b11d94b11')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f500abf6-bfa6-42b3-a084-a26b11d94b11 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f500abf6-bfa6-42b3-a084-a26b11d94b11');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>

    </div>
  </div>





```python
final_df.shape
```




    (1382, 8)



Our final dataset that will be used for analysis, final_df, has 1,382 rows and 8 columns.

Each row represents a grouping of three cycle lengths for a single participant, as well as the age, height, weight, and BMI of that participant.


```python
final_df.dtypes
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ClientID</th>
      <td>object</td>
    </tr>
    <tr>
      <th>Cycle_1</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>Cycle_2</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>Cycle_3</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>Int64</td>
    </tr>
    <tr>
      <th>Height</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>float64</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> object</label>



Of the 8 columns, all are numeric except for ``ClientID``, which is categorical. This column contains unique identification numbers that are used to identify each client in the study.

``ClientID`` will not be used as a predictor variable in the final model, but was useful in helping to group the cycle columns by threes for our final_df.


## Distribution of cycle_3

Our target variable is ``Cycle_3``, which is the length of the third menstrual cycle (for the client, this is the length of their current cycle).

We should visualise the distribution of ``Cycle_3`` to see if it looks like any common probability distributions. This will help us understand the range and frequency of potential outcomes from the final model and help us any potential biases in our data.



```python
sns.histplot(final_df["Cycle_3"], bins = 37, kde = True) # reduced number of bins until there were no empty spaces in plot
plt.xlabel("Cycle Length (days)") # change x-axis label
plt.ylabel("Frequency") # change y-axis label
plt.title("Menstrual Cycle 3 Length Distribution") # change title
plt.show()
```


    
![png](ML_final_project_files/ML_final_project_85_0.png)
    


From the above plot, it appears as though ``Cycle_3`` has a **right-skewed** distribution, meaning that the majority of the data is located on the left side of the graph, and the mean is greater than the median.

Most values are concentrated between 25 and 32 days, although the tail stretches to the right, indicating that some women in the sample have cycle lengths up to 50 or more days.

In other words, on average, most of the women in the sample have menstrual cycles that are shorter than longer. A small number of individuals had significantly longer cycles, but are outliers in the data.

Let's look at the mean, mode, minimum, and maximum length of ``Cycle_3`` in ``final_df`` in order to understand our target variable more.


```python
print("The average length of Cycle 3 of the entire sample is", round(final_df["Cycle_3"].mean(), 2), "days.")

print("The most common length of Cycle 3 of the entire sample is", round(statistics.mode(final_df["Cycle_3"]), 2), "days.")

print("The shortest length of Cycle 3 in the dataset is", final_df['Cycle_3'].min(), "days.")
print("The longest length of Cycle 3 in the dataset is", final_df['Cycle_3'].max(), "days.")
```

    The average length of Cycle 3 of the entire sample is 29.14 days.
    The most common length of Cycle 3 of the entire sample is 28.0 days.
    The shortest length of Cycle 3 in the dataset is 18.0 days.
    The longest length of Cycle 3 in the dataset is 54.0 days.


Compared to the aforementioned information reported by the American College of Obstetrics and Gynecologists that found the average menstrual cycle to be around 28 days long, the average length of ``Cycle_3`` in the current sample is slightly longer by about 1.15 days.

However, the most common length of ``Cycle_3`` for our dataset was 28 days, which aligns with the previous research.

As the above histogram illustrates, the distribution of ``Cycle_3`` is right-skewed, with some participants having irregular lengths of 50 or more days. The longest length was 54 days, which is significantly longer than average.

Outlier values like this one are pulling the mean to the right so it is greater (longer) than the median length.

However, a mean length of 29.14 days is not significantly longer than the median of 28 days. There are natural deviations in cycle length for each individual month to month. Thus, this difference is likely due to natural deviations from the mean.

(Source: [Harvard T.H. Chan School of Public Health, Menstrual cycles today](https://hsph.harvard.edu/research/apple-womens-health-study/study-updates/menstrual-cycles-today-how-menstrual-cycles-vary-by-age-weight-race-and-ethnicity/))

However, further analysis with be conducted to explore these outliers and whether they are cause for concern when creating our final model.



## Missing values

As we saw earlier, our demographic columns are the columns with missing values.

One thing we can do to reduce the number of missing values in ``BMI`` without using an imputer is by simply calculating the BMI for rows that have values for ``Height`` and ``Weight``.

Let's use this method for both the training and set sets. Since our y_train and y_test only contain ``Cycle_3``, which contains no missing values, we can just focus on using this method for ``X_train`` and ``X_test``.

We can then look at the number of missing values for ``BMI`` for both dataframes before and after using this method.

### training set


```python
# before
missing_bmi_count = X_train['BMI'].isnull().sum()
print(f"Number of missing values for BMI before imputation: {missing_bmi_count}")
```

    Number of missing values for BMI before imputation: 209



```python
X_train["BMI"] = round((X_train["Weight"] / (X_train["Height"] ** 2) * 703), 2)
```


```python
# after
missing_bmi_count = X_train['BMI'].isnull().sum()
print(f"Number of missing values for BMI after imputation: {missing_bmi_count}")
```

    Number of missing values for BMI after imputation: 205


So, it looks like by using this imputation strategy, we filled in 6 missing values in ``X_train`` for ``BMI`` just by using the values of ``Height`` and ``Weight`` that we already had in our dataset!

Now, let's look at the *proportion* of missing values for each column in our training set, as well as the actual *count* of missing values.


```python
# training set

# concatenate X_train and y_train into a single DataFrame
train = pd.concat([X_train, y_train], axis=1)

null = train.isnull().sum().sort_values(ascending=False)
null_per = ((train.isnull().sum()) / (train.shape[0])).sort_values(ascending=False)*100.
null_values = pd.DataFrame({
    "Column Name": null.index,
    "Total Number of Missing Values": null.values,
    "Percentage NA": null_per.values
})

null_values
```





  <div id="df-79d22303-c26b-4b47-8472-84be6de24f98" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column Name</th>
      <th>Total Number of Missing Values</th>
      <th>Percentage NA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BMI</td>
      <td>205</td>
      <td>21.199586</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Weight</td>
      <td>205</td>
      <td>21.199586</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Height</td>
      <td>205</td>
      <td>21.199586</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age</td>
      <td>157</td>
      <td>16.235781</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cycle_1</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cycle_2</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cycle_3</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-79d22303-c26b-4b47-8472-84be6de24f98')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-79d22303-c26b-4b47-8472-84be6de24f98 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-79d22303-c26b-4b47-8472-84be6de24f98');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>

    </div>
  </div>




Our demographic variables ``BMI``, ``Height``, ``Weight``, and ``Age`` are the columns with missing values in our training set.

``BMI``, ``Height``, and ``Weight`` all have approximately 20% of their values set to missing. ``Age`` has about 16% of its values missing.

### test set

Let's first use the BMI calculation strategy with our test set now too!


```python
# before
missing_bmi_count = X_test['BMI'].isnull().sum()
print(f"Number of missing values for BMI before imputation: {missing_bmi_count}")
```

    Number of missing values for BMI before imputation: 99



```python
X_test["BMI"] = round((X_test["Weight"] / (X_test["Height"] ** 2) * 703), 2)
```


```python
# after
missing_bmi_count = X_test['BMI'].isnull().sum()
print(f"Number of missing values for BMI after imputation: {missing_bmi_count}")
```

    Number of missing values for BMI after imputation: 91


Using the BMI calculation strategy filled in 6 values for ``X_test``!

Now for the proportion and count of missing values.


```python
# test set

# concatenate X_test and y_test into a single DataFrame
test = pd.concat([X_test, y_test], axis=1)

null = test.isnull().sum().sort_values(ascending=False)
null_per = ((test.isnull().sum()) / (test.shape[0])).sort_values(ascending=False)*100.
null_values = pd.DataFrame({
    "Column Name": null.index,
    "Total Number of Missing Values": null.values,
    "Percentage NA": null_per.values
})

null_values
```





  <div id="df-4097d005-8640-4a5e-84b9-ffad21cf2071" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column Name</th>
      <th>Total Number of Missing Values</th>
      <th>Percentage NA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BMI</td>
      <td>91</td>
      <td>21.927711</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Weight</td>
      <td>91</td>
      <td>21.927711</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Height</td>
      <td>91</td>
      <td>21.927711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age</td>
      <td>68</td>
      <td>16.385542</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cycle_1</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cycle_2</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cycle_3</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-4097d005-8640-4a5e-84b9-ffad21cf2071')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-4097d005-8640-4a5e-84b9-ffad21cf2071 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-4097d005-8640-4a5e-84b9-ffad21cf2071');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>

    </div>
  </div>




Our test set has a smaller *count* of missing values than our training set (100 for ``BMI``, ``Weight``, and ``Height`` versus 196, respectively), but because our test set only contains 30% of the data from ``final_df``, the proportion of missing values is much higher in comparison (24% versus 20%).

Once we create the pipelines for our final models, we can include imputers to fill in these remaining missing values.

## Outliers

Outliers, while potentially representing unusual or erroneous values, can also be informative about the real menstrual cycle lengths of American women. Removing them can skew our final model's understanding of the underlying data distribution, leading to incorrect predictions on new data points that may include similar outliers.

Let's start by plotting our variables with boxplots to see whether outliers exist in each of them.



```python
# create figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # one row, three columns

# list of cycles and subplot axes
cycles = ['Cycle_1', 'Cycle_2', 'Cycle_3']
titles = ['Cycle 1', 'Cycle 2', 'Cycle 3']

for i, (cycle, ax, title) in enumerate(zip(cycles, axes, titles)):
    # calculate quartiles and IQR
    Q1 = np.percentile(final_df[cycle], 25)
    Q3 = np.percentile(final_df[cycle], 75)
    IQR = Q3 - Q1

    # calculate median
    median = final_df[cycle].median()

    # define outlier bounds
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR

    # identify outliers
    outliers = final_df[cycle][(final_df[cycle] < lower_bound) | (final_df[cycle] > upper_bound)]

    # create the boxplot
    ax.boxplot(final_df[cycle], showfliers=False)
    ax.set_title(f'{title}: Boxplot with Labeled Median and Outliers')
    ax.set_xticks([1])
    ax.set_xticklabels([title])
    ax.set_ylabel('Value')

    # add median label
    ax.text(1, median + 0.5, f'Median: {median:.2f}', ha='center', va='bottom', fontsize=8, color='blue')

    # plot and label outliers
    ax.scatter(np.ones(len(outliers)), outliers, color='red', label='Outliers')
    for j, outlier in enumerate(outliers):
        ax.text(1.05, outlier, str(round(outlier, 2)), va='center', fontsize=8)

    # add legend only to first plot to avoid clutter
    if i == 0:
        ax.legend()

# adjust final layout
plt.tight_layout()
plt.show()
```


    
![png](ML_final_project_files/ML_final_project_106_0.png)
    


From these cycle boxplots, we can see that all three cycle variables have the same median, 28, which matches previous findings on average cycle lengths among the population.

Additionally, we can see that all three variables have a relatively large number of outliers, with more outliers present in the farther range of values (outliers with longer cycle lengths than typical). Indeed, some women in the sample had cycles that were 54 days long, much longer than the average 28 days.

This contradicts the study inclusion criteria, which required that participants have a stated menstrual cycle length ranging between 21-42 days long. However, it is unlikely that 54 days is an implausible or inaccurate measurement. A menstrual cycle length of 54 days, although concerning and atypical, is not impossible.

The large number of outliers within our cycle variables may be some cause for concern, especially when it comes to building accurate models. However, let's first look at our other predictor variables to see if outliers exist in them too.

Let's focus on just ``Age``, ``Height``, and ``Weight`` when looking for outliers, since ``Height`` and ``Weight`` capture all the information within ``BMI``.

Since these variables contain missing values, we will have to filter them first to remove the NAs before creating the boxplots.

Let's look at each of these variables individually.


```python
# Age

# create filtered dataset that removes NAs
filtered_data = final_df[~final_df['Age'].isnull()]

# calculate quartiles and IQR
Q1 = np.percentile(filtered_data["Age"], 25)
Q3 = np.percentile(filtered_data["Age"], 75)
IQR = Q3 - Q1

# calculate median
median = filtered_data['Age'].median()

# define outlier bounds
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

# identify outliers
outliers = filtered_data["Age"][(filtered_data["Age"] < lower_bound) | (filtered_data["Age"] > upper_bound)]

# create the plot
plt.figure(figsize=(8, 6))
plt.boxplot(filtered_data["Age"], tick_labels=['Age'], showfliers=False)  # ✅ Fixed this line

# add median label slightly above the box
plt.text(1, median + 0.5, f'Median: {median:.0f}', ha='center', va='bottom', fontsize=9, color='blue')

# plot and label outliers
plt.scatter(np.ones(len(outliers)), outliers, color='red', label='Outliers')
for i, outlier in enumerate(outliers):
    plt.text(1.05, outlier, str(round(outlier, 2)), va='center', fontsize=8)

# add labels and title
plt.title('Age: Boxplot with Labeled Median and Outliers')
plt.ylabel('Age')
plt.legend()

plt.show()
```


    
![png](ML_final_project_files/ML_final_project_109_0.png)
    


Our ``Age`` variable has a median age of 32 years old.

From this boxplot, we can see that there are no outliers within ``Age``!

This is likely explained by the original study by Fehring et al. having inclusion criteria for participants that they must be between the ages of 18 and 42 years old. This leaves only 25 possible values that can exist for ``Age``. Thus, there are no outliers present!


```python
# Height

# create filtered dataset that removes NAs
filtered_data = final_df[~final_df['Height'].isnull()]

# calculate quartiles and IQR
Q1 = np.percentile(filtered_data["Height"], 25)
Q3 = np.percentile(filtered_data["Height"], 75)
IQR = Q3 - Q1

# calculate median
median = filtered_data['Height'].median()

# define outlier bounds
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

# identify outliers
outliers = filtered_data["Height"][(filtered_data["Height"] < lower_bound) | (filtered_data["Height"] > upper_bound)]

# create the plot
plt.figure(figsize=(8, 6))
plt.boxplot(filtered_data["Height"], tick_labels=['Height'], showfliers=False)  # ✅ Fixed this line

# add median label slightly above the box
plt.text(1, median + 0.5, f'Median: {median:.0f}', ha='center', va='bottom', fontsize=9, color='blue')

# plot and label outliers
plt.scatter(np.ones(len(outliers)), outliers, color='red', label='Outliers')
for i, outlier in enumerate(outliers):
    plt.text(1.05, outlier, str(round(outlier, 2)), va='center', fontsize=8)

# add labels and title
plt.title('Height: Boxplot with Labeled Median and Outliers')
plt.ylabel('Height')
plt.legend()

plt.show()
```


    
![png](ML_final_project_files/ML_final_project_111_0.png)
    


The median height of participants within our dataset is 65 inches, which is about 5 feet, 4 inches. This matches the population, with the average American women's height being 5 feet, 4 inches.

Similar to ``Age``, ``Height`` has no outliers present. The upper and lower whiskers contain minimum and maximum heights that fall within normal ranges found in the population as well (about 5 feet at the lowest and 6 feet at the highest).

Source: [CDC: Body Measurements](https://www.cdc.gov/nchs/fastats/body-measurements.htm)


```python
# Weight

# create filtered dataset that removes NAs
filtered_data = final_df[~final_df['Weight'].isnull()]

# calculate quartiles and IQR
Q1 = np.percentile(filtered_data["Weight"], 25)
Q3 = np.percentile(filtered_data["Weight"], 75)
IQR = Q3 - Q1

# calculate median
median = filtered_data['Weight'].median()

# define outlier bounds
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

# identify outliers
outliers = filtered_data["Weight"][(filtered_data["Weight"] < lower_bound) | (filtered_data["Weight"] > upper_bound)]

# create the plot
plt.figure(figsize=(8, 6))
plt.boxplot(filtered_data["Weight"], tick_labels=['Weight'], showfliers=False)  # ✅ Fixed this line

# add median label slightly above the box
plt.text(1, median + 0.5, f'Median: {median:.0f}', ha='center', va='bottom', fontsize=9, color='blue')

# plot and label outliers
plt.scatter(np.ones(len(outliers)), outliers, color='red', label='Outliers')
for i, outlier in enumerate(outliers):
    plt.text(1.05, outlier, str(round(outlier, 2)), va='center', fontsize=8)

# add labels and title
plt.title('Weight: Boxplot with Labeled Median and Outliers')
plt.ylabel('Weight')
plt.legend()

plt.show()
```


    
![png](ML_final_project_files/ML_final_project_113_0.png)
    


The median weight of our sample is 145 pounds.

There are three outlier values present in ``Weight``: 260 pounds, 268 pounds, and 300 pounds.

Given that bodies come in a wide range of sizes (and therefore weights), and that there are only three outliers present in ``Weight``, I do not believe these outliers are cause for concern and do not need to be "dealt with" in any way.

Overall, from these boxplots we can see that the only columns with outliers that are truly cause for concern when creating our models are ``Cycle_1``, ``Cycle_2``, and ``Cycle_3``.

However, the final model that we end up selecting for the predictor tool has an impact on whether we need to transform these outliers.

If our final model is a linear regression model, these types of models are very sensitive to outliers, since these models try to minimize the sum of squared errors (SSE). The SSE is a measure of how well the model fits with the data. Outliers, being data poins far from the general trend, can disproportionally impact the SSE calculations, resulting in a skewed regression line. This skewed line can result in a poor fit for the rest of the data, reducing prediction accuracy.

If model fitting and tuning determines that a linear regression model is the best fit model for our predictor tool, we will have to determine a strategy to transform the outliers to reduce this skewed result.

Similarly, if a KNN model is found to have the best predictive power, we will have to edit these outliers. KNN models can be sensitive to outliers, especially when the value of k (number of neighbours) is small. When k is small, the influence of a single outlier is magnified. Outliers can disproportionately influence the prediction because KNN relies on the distances to the nearest neighbours to make a classification.

However, if the final model ends up being a tree-based model such as xgboost, we will not need to worry about removing these outliers. Tree-based models are generally robust to outliers as they focus mainly on dividing data based on splits instead of relying on absolute values or distances. Outliers are thus often "quarantined" within their own nodes, which limits their influence on the entire model.

So, with all this in mind, let's determine which model is optimal and then decide whether we need to remove outliers from our data or not.


# Pipeline

We will need to create a pipeline in order to impute missing values into our dataset.



### BMI

First, however, I want to make sure that the missing values within our ``BMI`` column are not imputed using the imputer. Its values can only be accurately determined from the values within ``Height`` and ``Weight``, not just the mean or median BMI for the entire sample.

So, let's write some code that will calculate the BMI for each Client to ensure that any missing BMI values will be filled.

Since the values of the "Height" and "Weight" columns are in the US customary units (pounds (lbs) and inches), we can calculate BMI for each row using the following formula:

**US customary units: BMI = weight (pounds) / [height (in)]^2 x 703**

Source: [CDC, What Is Body Mass Index (BMI)?](https://www.cdc.gov/growth-chart-training/hcp/using-bmi/body-mass-index.html)

Written in code, that formula looks like this:


```python
class BMICalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # BMI, Weight, and Height are at indices 5, 4, and 3 respectively
        # Calculate BMI for rows where it's missing but Weight and Height are available
        missing_bmi_mask = np.isnan(X[:, 5]) & ~np.isnan(X[:, 4]) & ~np.isnan(X[:, 3])
        X[missing_bmi_mask, 5] = X[missing_bmi_mask, 4] / ((X[missing_bmi_mask, 3] / 100) ** 2)  # Calculate BMI using NumPy array operations
        # Convert kg and cm to m^2 by /100^2

        return X
```

Before building our models, let's first create a K-fold object to set up our cross-validation.

Creating a kf object allows us to:

- specify a random_state so that results are always the same
- shuffle = True shuffles the dataset
    - If there is something non-random about the order of the rows, then not shuffling with affect results


```python
# set up cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

# Model fitting

We will determine which model is the most optimal by calculating the difference between the actual target values in the testing set and the values predicted by the model with Root Mean Square Error (RMSE).

The RMSE determines the absolute fit of the model to the data and indicates how close the actual data points are to the model's predicted values.

Notably, we will not be making any predictions on the length of ``Cycle_3`` that we dont' already have data on. Instead, we will have each of our models predict the lengths of Cycle 3 of each Client in the test set based on the lengths of Cycles 1 and 2 and we will then *compare* those predicted lengths to the *actual* Cycle 3 lengths in our dataset.

RMSE is the most appropriate metric to use to compare these models, as it is considered a standard metric in regression problems and is measured in the same units as the dependent variable, ``Cycle_3`` (days), making it directly interpretable.

A *low* RMSE value indicates a better fitting model and is a good measure for determining the accuracy of the model's predictions.







## Linear regression

We should first create a simple linear regression model that will serve as our *comparison model* for all the other more complex models we will build. This is because a linear regression model is one of the simplest models with which to build a machine learning model from.

The only parameter we will tune is the imputer strategy, just to see which we should potentially use for the rest of our models.

By doing so, we can answer the question: "Does this model perform better than a simple linear regression model?"


```python
# simple linear regression
linear_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'mean')),
    ('bmi_calc', BMICalculator()), # after SimpleImputer and before StandardScaler
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

# Define the parameter grid for GridSearchCV
linear_param_grid = {
    'imputer__strategy': ['mean', 'median'], # explore different imputation strategies
}

linear_cv = GridSearchCV(
    linear_pipeline,
    linear_param_grid,
    cv=kf,
    scoring='neg_root_mean_squared_error',
    return_train_score=True  # also track training scores
)

linear_cv.fit(X_train, y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=KFold(n_splits=5, random_state=42, shuffle=True),
             estimator=Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()),
                                       (&#x27;bmi_calc&#x27;, BMICalculator()),
                                       (&#x27;scaler&#x27;, StandardScaler()),
                                       (&#x27;linear&#x27;, LinearRegression())]),
             param_grid={&#x27;imputer__strategy&#x27;: [&#x27;mean&#x27;, &#x27;median&#x27;]},
             return_train_score=True, scoring=&#x27;neg_root_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=KFold(n_splits=5, random_state=42, shuffle=True),
             estimator=Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()),
                                       (&#x27;bmi_calc&#x27;, BMICalculator()),
                                       (&#x27;scaler&#x27;, StandardScaler()),
                                       (&#x27;linear&#x27;, LinearRegression())]),
             param_grid={&#x27;imputer__strategy&#x27;: [&#x27;mean&#x27;, &#x27;median&#x27;]},
             return_train_score=True, scoring=&#x27;neg_root_mean_squared_error&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: Pipeline</div></div></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()), (&#x27;bmi_calc&#x27;, BMICalculator()),
                (&#x27;scaler&#x27;, StandardScaler()), (&#x27;linear&#x27;, LinearRegression())])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>BMICalculator</div></div></label><div class="sk-toggleable__content fitted"><pre>BMICalculator()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>LinearRegression</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a></div></label><div class="sk-toggleable__content fitted"><pre>LinearRegression()</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
best_linear_score = -linear_cv.best_score_  # convert back to MSE
linear_model = linear_cv.best_estimator_

# print optimization results
print(f"Linear regression: \n Best parameters:\n {linear_cv.best_params_} \n\n RMSE: **{best_linear_score}**")
```

    Linear regression: 
     Best parameters:
     {'imputer__strategy': 'mean'} 
    
     RMSE: **3.2793947453484256**



```python
print(f"From our 5-fold cross-validation, the simple linear regression model's predictions for Cycle_3 were on average {best_linear_score:.2f} days off from the actual lengths.")
```

    From our 5-fold cross-validation, the simple linear regression model's predictions for Cycle_3 were on average 3.28 days off from the actual lengths.


From our hyperparameter tuning of the imputer strategy, it looks like **mean** is the best imputer strategy. Since it was deemed the most optimal strategy for our simple linear regression model, let's go ahead and use a mean imputer for our subsequent models.

Let's move onto a slightly more advanced algorithm: ElasticNet Regression.



## ElasticNet regression

ElasticNet Regression is a powerful machine learning algorithm that combines the features of Lasso (L1) and Ridge (L2) Regression by determining the degree and strength with which to apply **both an L1 and L2 penalty at the same time**.

It is an improvement to a simple linear regression model in that it addresses the issues that simple linear regression has with multicollinearity. When predictor variables are highly correlated with one another (ie., multicollinearity), this can lead to unstable and inaccurate coefficient estimates. Additionally, a simple linear regression model does not perform feature selection and includes all predictors in the model, potentially leading to an overcomplicated model.

ElasticNet, on the other hand, combines the strengths of Lasso and Ridge, allowing it to address multicollinearity more effectively. By instituing an L1 penalty term, it can also force coefficients of less important predictors to zero.

Let's determine which hyperparameters we should tune for our ElasticNet model.



**Hyperparameters to tune**

``alpha`` = strength of the regularization

- must be a positive float, value between 0 and 1
    - 0 = all weight to the L2 penalty
    - 1 = all weight to the L1 penalty
- 1 minus the alpha value = the strength of the L2 penalty
- For example, an alpha of 0.5 would provide a 50 percent contribution of each penalty to the loss function. An alpha value of 0 gives all weight to the L2 penalty and a value of 1 gives all weight to the L1 penalty.

``l1_ratio`` = determines the mix between L1 and L2 regularization
- = 1: entirely lasso penalty (L1)
- = 0: entirely ridge penalty (L2)






```python
# print best params correclty (not as float()
np.set_printoptions(legacy='1.25')

# ElasticNet Regression with GridSearchCV
alphas = np.logspace(-2, 2, 11)
l1_ratios = np.linspace(0, 1, 11)

elasticnet_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'mean')),
    ('bmi_calc', BMICalculator()), # after SimpleImputer and before StandardScaler
    ('scaler', StandardScaler()),
    ('elasticnet', ElasticNet())
])

elasticnet_params = {'elasticnet__alpha': alphas,
                    'elasticnet__l1_ratio': l1_ratios}

elasticnet_cv = GridSearchCV(
    elasticnet_pipeline,
    elasticnet_params,
    cv=kf,
    scoring='neg_root_mean_squared_error',
    return_train_score=True  # also track training scores
)
elasticnet_cv.fit(X_train, y_train)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=KFold(n_splits=5, random_state=42, shuffle=True),
             estimator=Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()),
                                       (&#x27;bmi_calc&#x27;, BMICalculator()),
                                       (&#x27;scaler&#x27;, StandardScaler()),
                                       (&#x27;elasticnet&#x27;, ElasticNet())]),
             param_grid={&#x27;elasticnet__alpha&#x27;: array([1.00000000e-02, 2.51188643e-02, 6.30957344e-02, 1.58489319e-01,
       3.98107171e-01, 1.00000000e+00, 2.51188643e+00, 6.30957344e+00,
       1.58489319e+01, 3.98107171e+01, 1.00000000e+02]),
                         &#x27;elasticnet__l1_ratio&#x27;: array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])},
             return_train_score=True, scoring=&#x27;neg_root_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=KFold(n_splits=5, random_state=42, shuffle=True),
             estimator=Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()),
                                       (&#x27;bmi_calc&#x27;, BMICalculator()),
                                       (&#x27;scaler&#x27;, StandardScaler()),
                                       (&#x27;elasticnet&#x27;, ElasticNet())]),
             param_grid={&#x27;elasticnet__alpha&#x27;: array([1.00000000e-02, 2.51188643e-02, 6.30957344e-02, 1.58489319e-01,
       3.98107171e-01, 1.00000000e+00, 2.51188643e+00, 6.30957344e+00,
       1.58489319e+01, 3.98107171e+01, 1.00000000e+02]),
                         &#x27;elasticnet__l1_ratio&#x27;: array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])},
             return_train_score=True, scoring=&#x27;neg_root_mean_squared_error&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: Pipeline</div></div></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()), (&#x27;bmi_calc&#x27;, BMICalculator()),
                (&#x27;scaler&#x27;, StandardScaler()),
                (&#x27;elasticnet&#x27;,
                 ElasticNet(alpha=0.06309573444801933, l1_ratio=1.0))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>BMICalculator</div></div></label><div class="sk-toggleable__content fitted"><pre>BMICalculator()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>ElasticNet</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.ElasticNet.html">?<span>Documentation for ElasticNet</span></a></div></label><div class="sk-toggleable__content fitted"><pre>ElasticNet(alpha=0.06309573444801933, l1_ratio=1.0)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
best_elasticnet_score = -elasticnet_cv.best_score_  # convert back to MSE
elasticnet_model = elasticnet_cv.best_estimator_

# print optimization results
print(f"ElasticNet: \n Best parameters:\n {elasticnet_cv.best_params_} \n\n RMSE: **{best_elasticnet_score}**")
```

    ElasticNet: 
     Best parameters:
     {'elasticnet__alpha': 0.06309573444801933, 'elasticnet__l1_ratio': 1.0} 
    
     RMSE: **3.2781404657526734**



```python
print(f"From our 5-fold cross-validation, the ElasticNet model's predictions for Cycle_3 were on average {best_elasticnet_score:.2f} days off from the actual lengths.")
```

    From our 5-fold cross-validation, the ElasticNet model's predictions for Cycle_3 were on average 3.28 days off from the actual lengths.


Based on our hyperparameter tuning results, we can see that:

An **alpha value** of 0.01 indicates that the model will primarily be influenced by the L2 penalty, as it is very close to 0 (with 0 meaning all weight is given to the L2 penalty). L2 (Ridge) regularization favours the shrinkage of coefficients without forcing them to zero.

An **l1_ratio** value of 0 indicates that the mix of penalties resulted in an entirely ridge penalty. Ridge regressions create *stable* models by minimizing the impact of less important features while keeping all variables in the model. This prevents the model from overfitting, especially when there is not enough data, which is the case for our data.

Overall, our ElasticNet regression model is behaving more like a Ridge regression.

This suggests that overfitting may be a concern for our data. **Overfitting** occurs when a model learns the training data too well, including noise and irrelevant details, resulting in poor performance on new, unseen data.

When a model yields a small training RMSE but a very large test RMSE, we are said to be overfitting the data. The testing set RMSE will be very large because the supposed patterns that the model found in the training set does not exist in the testing set.

We should keep this in mind when evaluating our models' training and test performance in the future.

## XGB

With the knowledge that our data may be prone to overfitting, implementing an XGBoost model may be more appropriate as they are robust to noisy data, outliers, and missing values.

XGBoost is a type of ensemble learning method that combines multiple weak models to form a stronger model.

It uses decision trees as its "base learners", combining them sequentially to improve the model's performance. Each new tree "learns" (ie., is trained) from the previous tree to correct its errors; this process is called **boosting**.

Like all machine learning algorithms, XGBoost supports customizations to allow users to adjust model hyperparameters to optimize performance based on the specific problem.

So, let's go ahead and review which hyperparameters we will be tuning for our own XGBoost model.




**Hyperparameters to tune**

``n_estimators`` = the number of boosting stages to perform, ie. how many trees will be in the model

- large number usually results in better performance but can lead to overfitting

``learning_rate`` = aka "shrinkage", governs how "greedy" each tree is. Is used to control and adjust the weighting of the internal model estimators.

- should always be a small value to force long-term learning.
- if you have a slow learning rate, it will take more time to converge, but it will resist overfitting more, as each tree will learn and improve only by a little bit each time

``max_depth`` = the maximum depth of each tree

- increasing this = model more complex & like to overfit
- a smaller value results in simpler models, which can lead to underfitting



```python
# create the pipeline
xgb_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'mean')),
    ('bmi_calc', BMICalculator()), # after SimpleImputer and before StandardScaler
    ('scaler', StandardScaler()),
    ('xgb', xgb.XGBRegressor())
])

# Define the parameter grid
xgb_params = {
    'xgb__n_estimators': [20, 25, 50],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__max_depth': [1, 2, 3]
}

# Initialize GridSearchCV
xgb_cv = GridSearchCV(
    estimator = xgb_pipeline,
    param_grid = xgb_params,
    scoring = 'neg_root_mean_squared_error',
    cv = kf,
    n_jobs = -1,
    return_train_score = True,  # store training scores
)

# Fit the grid search to the data
xgb_cv.fit(X_train, y_train)
```




<style>#sk-container-id-3 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-3 {
  color: var(--sklearn-color-text);
}

#sk-container-id-3 pre {
  padding: 0;
}

#sk-container-id-3 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-3 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-3 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-3 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-3 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-3 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-3 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-3 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-3 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-3 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-3 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-3 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-3 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-3 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-3 div.sk-label label.sk-toggleable__label,
#sk-container-id-3 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-3 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-3 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-3 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-3 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-3 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-3 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-3 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-3 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-3 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=KFold(n_splits=5, random_state=42, shuffle=True),
             estimator=Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()),
                                       (&#x27;bmi_calc&#x27;, BMICalculator()),
                                       (&#x27;scaler&#x27;, StandardScaler()),
                                       (&#x27;xgb&#x27;,
                                        XGBRegressor(base_score=None,
                                                     booster=None,
                                                     callbacks=None,
                                                     colsample_bylevel=None,
                                                     colsample_bynode=None,
                                                     colsample_bytree=None,
                                                     device=None,
                                                     early_stopping_rounds=None,
                                                     enable...
                                                     max_leaves=None,
                                                     min_child_weight=None,
                                                     missing=nan,
                                                     monotone_constraints=None,
                                                     multi_strategy=None,
                                                     n_estimators=None,
                                                     n_jobs=None,
                                                     num_parallel_tree=None,
                                                     random_state=None, ...))]),
             n_jobs=-1,
             param_grid={&#x27;xgb__learning_rate&#x27;: [0.01, 0.1, 0.2],
                         &#x27;xgb__max_depth&#x27;: [1, 2, 3],
                         &#x27;xgb__n_estimators&#x27;: [20, 25, 50]},
             return_train_score=True, scoring=&#x27;neg_root_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=KFold(n_splits=5, random_state=42, shuffle=True),
             estimator=Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()),
                                       (&#x27;bmi_calc&#x27;, BMICalculator()),
                                       (&#x27;scaler&#x27;, StandardScaler()),
                                       (&#x27;xgb&#x27;,
                                        XGBRegressor(base_score=None,
                                                     booster=None,
                                                     callbacks=None,
                                                     colsample_bylevel=None,
                                                     colsample_bynode=None,
                                                     colsample_bytree=None,
                                                     device=None,
                                                     early_stopping_rounds=None,
                                                     enable...
                                                     max_leaves=None,
                                                     min_child_weight=None,
                                                     missing=nan,
                                                     monotone_constraints=None,
                                                     multi_strategy=None,
                                                     n_estimators=None,
                                                     n_jobs=None,
                                                     num_parallel_tree=None,
                                                     random_state=None, ...))]),
             n_jobs=-1,
             param_grid={&#x27;xgb__learning_rate&#x27;: [0.01, 0.1, 0.2],
                         &#x27;xgb__max_depth&#x27;: [1, 2, 3],
                         &#x27;xgb__n_estimators&#x27;: [20, 25, 50]},
             return_train_score=True, scoring=&#x27;neg_root_mean_squared_error&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" ><label for="sk-estimator-id-14" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: Pipeline</div></div></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()), (&#x27;bmi_calc&#x27;, BMICalculator()),
                (&#x27;scaler&#x27;, StandardScaler()),
                (&#x27;xgb&#x27;,
                 XGBRegressor(base_score=None, booster=None, callbacks=None,
                              colsample_bylevel=None, colsample_bynode=None,
                              colsample_bytree=None, device=None,
                              early_stopping_rounds=None,
                              enable_categorical=False, eval_metric=None,
                              feature_types=None, gamma=None, grow_policy=None,
                              importance_type=None,
                              interaction_constraints=None, learning_rate=0.2,
                              max_bin=None, max_cat_threshold=None,
                              max_cat_to_onehot=None, max_delta_step=None,
                              max_depth=2, max_leaves=None,
                              min_child_weight=None, missing=nan,
                              monotone_constraints=None, multi_strategy=None,
                              n_estimators=50, n_jobs=None,
                              num_parallel_tree=None, random_state=None, ...))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" ><label for="sk-estimator-id-15" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-16" type="checkbox" ><label for="sk-estimator-id-16" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>BMICalculator</div></div></label><div class="sk-toggleable__content fitted"><pre>BMICalculator()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-17" type="checkbox" ><label for="sk-estimator-id-17" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-18" type="checkbox" ><label for="sk-estimator-id-18" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>XGBRegressor</div></div></label><div class="sk-toggleable__content fitted"><pre>XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.2, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=2, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=50, n_jobs=None,
             num_parallel_tree=None, random_state=None, ...)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
xgb_cv.best_params_
```




    {'xgb__learning_rate': 0.2, 'xgb__max_depth': 2, 'xgb__n_estimators': 50}




```python
best_xgb_score = -xgb_cv.best_score_  # convert back to MSE
xgb_model = xgb_cv.best_estimator_

# print optimization results
print(f"XGB: \n Best parameters:\n{xgb_cv.best_params_} \n\n RMSE: **{best_xgb_score}**")
```

    XGB: 
     Best parameters:
    {'xgb__learning_rate': 0.2, 'xgb__max_depth': 2, 'xgb__n_estimators': 50} 
    
     RMSE: **3.2180051078575618**



```python
print(f"From our 5-fold cross-validation, the XGBOost model's predictions for Cycle_3 were on average {best_xgb_score:.2f} days off from the actual lengths.")
```

    From our 5-fold cross-validation, the XGBOost model's predictions for Cycle_3 were on average 3.22 days off from the actual lengths.


Based on our hyperparameter tuning results, we can see that:

A **learning_rate** value of 0.2 indicates that the model will likely have long-term learning. Additionally, as mentioned previously, smaller (aka slower) learning rates resist overfitting more. From our ElasticNet model, we gleaned that overfitting will likely be an issue with our dataset due to its small size. A small learning rate of 0.2 further supports the theory that the most optimal models are ones that reduce overfitting.

A **max_depth** value of 1 additionally supports this notion, as smaller values result in simpler models that are less prone to overfitting.

Finally, a **n_estimators** value of 25 is lower than typical which helps reduce overfitting. A common value for this hyperparameter is between 100 and 1000, which is much larger than our optimal value of 25.

Overall, our best-performing cross-validated XGBoost model took numerous steps to reduce the issue of overfitting that is likely present in our data.



## KNN

k-Nearest Neighbour models, also known as KNN models, is an algorithm that makes predictions for the target variable by finding the "k" nearest data points to a given input and averaging their target values.

It is the responsibility of the person making the model (that's us!) to choose the number of nearest neighbours "k" that will be used to make predictions.

There are additionally other hyperparameters that we can tune to find the most optimal KNN model.

**Hyperparameters to tune**

``n_neighbours`` = the number of nearest neighbours to make predictions from

- a small k (ie., 1 or 3) may lead to noisy predictions (overfitting)
- a large k may lead to overly smoothed predictions (underfitting)

``metric`` = a distance metric that is used to measure the similarity between data points.

- euclidean = calculates the *straight-line* distance between two data points
- manhattan = calculates the distance by summing the absolute differences between the coordinates of two poins.
    - useful for grid-based data or when the distance is measured along axes
    - less sensitive to outliers than euclidean distance



```python
knn_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'mean')),
    ('bmi_calc', BMICalculator()), # after SimpleImputer and before StandardScaler
    ("scaler", StandardScaler()),
    ('knn', KNeighborsRegressor())
])

# Define the parameter grid for KNN
knn_param_grid = {
    'knn__n_neighbors': [i for i in range(2, 40, 2)], # dictionary key, needs to be double underscored to be understood
    'knn__metric': ['euclidean', 'manhattan'],  # weighting scheme for neighbors
}

# Initialize GridSearchCV for KNN
knn_cv = GridSearchCV(
    knn_pipeline,
    param_grid = knn_param_grid,
    cv=kf,  # using the predefined KFold object
    scoring='neg_root_mean_squared_error',
    return_train_score=True,
    n_jobs=-1
)

# Fit the grid search to the training data
knn_cv.fit(X_train, y_train)
```




<style>#sk-container-id-4 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-4 {
  color: var(--sklearn-color-text);
}

#sk-container-id-4 pre {
  padding: 0;
}

#sk-container-id-4 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-4 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-4 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-4 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-4 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-4 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-4 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-4 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-4 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-4 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-4 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-4 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-4 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-4 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-4 div.sk-label label.sk-toggleable__label,
#sk-container-id-4 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-4 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-4 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-4 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-4 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-4 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-4 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-4 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-4 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-4 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=KFold(n_splits=5, random_state=42, shuffle=True),
             estimator=Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()),
                                       (&#x27;bmi_calc&#x27;, BMICalculator()),
                                       (&#x27;scaler&#x27;, StandardScaler()),
                                       (&#x27;knn&#x27;, KNeighborsRegressor())]),
             n_jobs=-1,
             param_grid={&#x27;knn__metric&#x27;: [&#x27;euclidean&#x27;, &#x27;manhattan&#x27;],
                         &#x27;knn__n_neighbors&#x27;: [2, 4, 6, 8, 10, 12, 14, 16, 18,
                                              20, 22, 24, 26, 28, 30, 32, 34,
                                              36, 38]},
             return_train_score=True, scoring=&#x27;neg_root_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-19" type="checkbox" ><label for="sk-estimator-id-19" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=KFold(n_splits=5, random_state=42, shuffle=True),
             estimator=Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()),
                                       (&#x27;bmi_calc&#x27;, BMICalculator()),
                                       (&#x27;scaler&#x27;, StandardScaler()),
                                       (&#x27;knn&#x27;, KNeighborsRegressor())]),
             n_jobs=-1,
             param_grid={&#x27;knn__metric&#x27;: [&#x27;euclidean&#x27;, &#x27;manhattan&#x27;],
                         &#x27;knn__n_neighbors&#x27;: [2, 4, 6, 8, 10, 12, 14, 16, 18,
                                              20, 22, 24, 26, 28, 30, 32, 34,
                                              36, 38]},
             return_train_score=True, scoring=&#x27;neg_root_mean_squared_error&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-20" type="checkbox" ><label for="sk-estimator-id-20" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: Pipeline</div></div></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()), (&#x27;bmi_calc&#x27;, BMICalculator()),
                (&#x27;scaler&#x27;, StandardScaler()),
                (&#x27;knn&#x27;,
                 KNeighborsRegressor(metric=&#x27;manhattan&#x27;, n_neighbors=18))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-21" type="checkbox" ><label for="sk-estimator-id-21" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-22" type="checkbox" ><label for="sk-estimator-id-22" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>BMICalculator</div></div></label><div class="sk-toggleable__content fitted"><pre>BMICalculator()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-23" type="checkbox" ><label for="sk-estimator-id-23" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-24" type="checkbox" ><label for="sk-estimator-id-24" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>KNeighborsRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">?<span>Documentation for KNeighborsRegressor</span></a></div></label><div class="sk-toggleable__content fitted"><pre>KNeighborsRegressor(metric=&#x27;manhattan&#x27;, n_neighbors=18)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
# get the best score and estimator
best_knn_score = -knn_cv.best_score_
knn_model = knn_cv.best_estimator_

# print optimization results
print(f"KNN: \nBest parameters:\n{knn_cv.best_params_}\n\nRMSE: {best_knn_score}")
```

    KNN: 
    Best parameters:
    {'knn__metric': 'manhattan', 'knn__n_neighbors': 18}
    
    RMSE: 3.238150776096615



```python
print(f"From our 5-fold cross-validation, the KNN model's predictions for Cycle_3 were on average {best_knn_score:.2f} days off from the actual lengths.")
```

    From our 5-fold cross-validation, the KNN model's predictions for Cycle_3 were on average 3.24 days off from the actual lengths.


From our hyperparameter tuning results, we can see that:

A **distance metric** of manhattan indicates that the most optimal model is one that is robust to outliers, since manhattan is less sensitive to outliers than euclidean.

A **n_neighbours** value of 18 is relatively small compared to the maximum number of 40 neighbours that it could have taken. A small k value is more prone to overfitting than a large k value, so it is surpising that the most optimal model has such a relatively low k value. KNN models are pretty sensitive to overiffting as well, so this finding goes against our previous theory of there being a risk of overfitting in our data.

However, a k value of 18 is still much higher than 1 or 2, and is just about the middle value in the range of 2 to 39 that we supplised the cross-validated model. So, 18 neighbours may in actuality be the perfect balance between overfitting and underfitting.



Let's visualise the GridSearchCV results using a heatmap.


```python
# Extract the results
results = pd.DataFrame(knn_cv.cv_results_)

# Convert scores from negative to positive RMSE
results['positive_rmse'] = -results['mean_test_score']

# Create a pivot table for the heatmap
pivot_df = pd.pivot_table(
    data=results,
    values='positive_rmse',
    index='param_knn__n_neighbors',  # k values on y-axis
    columns='param_knn__metric',     # distance metric on x-axis
)

# Create the plot
plt.figure(figsize=(12, 8))
ax = sns.heatmap(
    pivot_df,
    annot=True,
    fmt=',.3f',
    cmap='gray_r',
    linewidths=0.25,
    vmin = 3.163,
    vmax = 3.760,
    cbar_kws={'label': 'RMSE (days)'}
)

# Set title and labels
plt.title('KNN Regression: RMSE by Distance Metric and Number of Neighbors', fontsize=14)
plt.xlabel('Distance Metric', fontsize=12)
plt.ylabel('Number of Neighbors (k)', fontsize=12)

# highlight best cell in red box
# ------------------------
# Get location of min RMSE
min_val = pivot_df.min().min()
min_coords = np.where(pivot_df == min_val)
row_idx = min_coords[0][0]
col_idx = min_coords[1][0]

# Draw rectangle (box) around the cell
ax.add_patch(plt.Rectangle(
    (col_idx, row_idx), 1, 1, fill=False, edgecolor='red', lw=3
))

# Add note about the best parameters
best_params = knn_cv.best_params_
best_rmse = -knn_cv.best_score_
plt.figtext(
    0.5, 0.01,
    f"Best parameters: k={best_params['knn__n_neighbors']}, metric={best_params['knn__metric']} with RMSE: {best_rmse:,.2f}",
    ha='center', fontsize=12
)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

```


    
![png](ML_final_project_files/ML_final_project_149_0.png)
    


From the plot, we can see that there isn't a whole lot of variation in the RMSE cross-validation score for each hyperparameter combination. There is only about a 0.60 difference between the worst performing and best preforming KNN model.

However, overall, k values in the middle of the range of possible values we supplied have better RMSE scores than K values in the lowest and highest ranges.

Additionally, besides a few values at the highest possible number of neighbours, models with the manhattan distance metric performed better overall in cross-validation than models with the euclidean distance metric.

# Model comparison

## test set performance

Now that we've found the optimal parameters for all our models with cross-validation, we will fit each model using all of the training set and use the test set as the final evaluation for how each model performs.




```python
# create a dictionary to store the best models
models = {
    'linear': linear_model,
    'elasticnet': elasticnet_model,
    'xgboost': xgb_model,
    'knn': knn_model,
}
```


```python
# Evaluate the best models and print results
results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = rmse
    print(f"{name}: RMSE = {rmse:.4f}")

# Find the best-performing model based on RMSE
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name} with RMSE = {results[best_model_name]:.4f}")

```

    linear: RMSE = 3.0059
    elasticnet: RMSE = 3.0024
    xgboost: RMSE = 2.9943
    knn: RMSE = 2.9979
    
    Best Model: xgboost with RMSE = 2.9943


As you can see, our XGBoost model performed the best on the test set of all the other models, with an RMSE of about 3.12 days.

Our simple linear regression model, on the other hand, performed the worst among all the models. Thus, we can confirm that our ElasticNet, XGBoost, and KNN models all perform better than a simple linear regression model.

Overall, though, all four of our models have test RMSEs that are fairly simllar to each other, with little variance.  

Close final test scores across different machine learning algorithms can be due to insufficient data or overfititng, both of which are issues we have run into already. Even seemingly different models can converge to similar performance if they are not complex enough to capture the subtle nuances that exist in the data. These underlying problems can make it difficult for any single algorithm to achieve a significant advantage.

We should further evaluate each model based on overfitting versus underfitting, bias versus variance tradeoff, and flexibility and interpretability to see how prominent these issues are.

## Evaluating overfitting

In order to determine which model is the best fit, we can compare the differences between the training RMSE and test RMSE of the two models.

In the case of RMSE, when a model yields a small training RMSE but a very large test RMSE, we are said to be overfitting the data.

The testing set RMSE will be very large because the supposed patterns that the model found in the training set does not exist in the testing set.

The model with the smallest difference will indicate that it has a stronger fit and is also at less risk for overfitting.

Let's go ahead and calculate the differences between the testing and training RMSEs of each of our models.



```python
models = {
    'linear': linear_model,
    'elasticnet': elasticnet_model,
    'xgboost': xgb_model,
    'knn': knn_model,
}

# calculate the y_pred for train and test sets
results = {}

for name, model in models.items():
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    diff = rmse_test - rmse_train
    results[name] = rmse_train, rmse_test, diff
    print(f"{name}: Train RMSE = {rmse_train:.4f}, Test RMSE = {rmse_test:.4f}, Difference = {diff:.4f}")

```

    linear: Train RMSE = 3.2624, Test RMSE = 3.0059, Difference = -0.2565
    elasticnet: Train RMSE = 3.2669, Test RMSE = 3.0024, Difference = -0.2645
    xgboost: Train RMSE = 2.8909, Test RMSE = 2.9943, Difference = 0.1034
    knn: Train RMSE = 3.0599, Test RMSE = 2.9979, Difference = -0.0620


Regarding overfitting concerns, a generally accepted rule of thumb is that if the difference between the training and test set performance is more than 5-10%, it suggests overfitting.

The model that performed better better on the training set than the test set to the largest degree is our KNN model, with a 0.1326 difference. This translates to about a 4.23% difference, which is very close to our threshold of 5-10%. This suggests that our KNN model is overfitting. This is in accordance with our previous statement about outliers and KNN models. KNN models can be sensitive to outliers, as they can lead to amplified and potentially inaccurate decision boundaries. Thus, overall, we can state with pretty high confidence that our final KNN model is overfitting.

The rest of our models: simple linear regression, ElasticNet, and XGBoost, do not have have a better performing training set compared to the test set. If anything, our more simple models (linear and ElasticNet) are more prone to underfitting than overfitting. These models assume a linear relationship and predict the data in a straight line format, which may not always be the case in real-world data. Since these models are too simple, it cannot learn the patterns in the data, leading to underfitting.

A good sign of underfitting is when performance on both the training and set set are poor. Compared to our XGBoost model, these models performed worst on both the training and set sets.

Thus, choosing XGBOost as our final model is still the right move. Not only does it have the best test set performance, XGBoost is impervious to outliers, noisy data, etc., and thus is the best model to use for our data.

## Bias-variance tradeoff

The **bias-variance tradeoff** is the delicate balance between two sources of error in a predictive model: bias and variance.

**Bias** represents the error due to overly simplistic assumptions in the learning algorithm. It is the difference between the prediction of the values by the machine learning model and the actual values.

Conversely, **variance** reflects the model's sensitivity to small fluctuations in the training data.

A model that exhibits small variance and high bias will underfit the target, whereas a model with high variance and low bias will overfit the target.

From the previous section, we determined that our KNN model is very close to the 5-10% threshold for overfitting. Thus, it can be concluded that this model likely has high variance and low bias.

Additionally, our linear regression and ElasticNet models are likely too simple to accurately capture the complex patterns in our dataset. Thus, they likely have low variance and high bias.

Overall, it seems like our XGBoost model achieved the bias-variance tradeoff the most out of all the algorithms. It has low bias and low variance, meaning it is able to capture the underlying patterns in our data and is not too sensitive to changes in the training data. It has the right level of complexity to minimize both vias and variance, achieving accurate generalizations to new unseen data.






## Flexibility vs interpretability

A **flexible** model can adapt to complex patterns in the data, leading to more accurate predictions, but they are often harder to understand and explain. Flexibility defines how *restrictive* the model is.

**Interpretable** models, on the other hand, are easier to understand and explain, but may be less flexible, more rigid, and less accurate.

Our simple linear regression model is known for its interpretability. In fact, linear regression models are one of the first regression models taught in statistics classes due to their simplicity and use case beyond just the world of machine learning. However, by being interpretable, it sacrifices flexibility; as mentioned previously, linear regression models are unable to capture complex, non-linear relationships.

ElasticNet models offer a good balance between interpretability and flexibility. The balance that they give to both Lasso and Ridge regression methods allow them to handle complex relationships and multicollinearity. Given that it is also a linear regression technique, it also has high interpretability.

Similarly, KNN models are very interpretable. Its methods are easy to understand and explain--it simply relies on the proximity of nearest neighbours in the data to make predictions.Its ability to handle various data types and the customization in distance metrics and 'k' value makes it suitable for a wide range of problems, thereby having high flexibility.

Lastly, XGBoost is considered to have very good flexibility. It's immunity to outliers and noisy data, and the ability for its trees to "learn" from previous trees results in accurate predictions and adaptability. However, it is generally considered less interpretable than the other models we used. It is often considered a "black box" model due to the challenge of understanding its internal workings.


# Final thoughts

As you can see, our XGBoost model performed the best on the test set of all the other models, with an RMSE of about 3.12 days.

Given that, as mentioned previously, cycle lengths naturally vary by about 5 days month to month, a model whose predictions are on average only 3 days off from actual lengths is a strong model. Since cycle lengths are already not an exact science, and  no model can predit with 100% accuracy the *exact* date someone's next period will arrive, an average error rate of 3 days is very good.

We previously explored whether we should remove outliers from our data or not, and determined that we should wait until we find our optimal model before deciging. Since XGBoost models are generally robust to outliers, and we determined that the outliers present in our data are not implausible but just due to natural fluctuations in the population, we can confidently decide to *not* transform or remove the outliers in any way.  

Overall, XGBoost models are resistant to outliers, noisy data, and overfitting, all of which are concerns with our small dataset. Through comparing our models, we were able to determine that our XGBoost model was the least prone to overfitting and achieved the bias-variance tradeoff to the highest degree. Despite some concern regarding the interpretability of the model, since prediction is our main interest in the current project, interpretability is not that much of a concern, and we should use the most flexible model that is usually the most accurate.

We will use this model to make our predictor tool web app!




