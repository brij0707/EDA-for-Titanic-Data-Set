# EDA-for-Titanic-Data-Set
We will do the basic EDA on Titanic Dataset
---
<a name = Section1></a>
# **1. Introduction to Dataset**
---

- One of the most **popular disasters** in the history is the **sinking** of the **Titanic**.

- **Titanic** was a British passenger ship operated by the **White Star Line** that **sank** in the North Atlantic Ocean on 15 April 1912.

- The reason behind **sinking** of this beast was because of **striking** to an **iceberg** while travelling from **Southampton** to **New York City**.
- The popular event has **inspired** numerous **works of art**, the most prominient being the 1997 movie - Titanic. Don't tell me that you have not watched it yet.

![image](https://raw.githubusercontent.com/brij0707/EDA-for-Titanic-Data-Set/main/images/RMS_Titanic_3.jpg)

---
<a name = Section2></a>
# **2. Problem Statement**
---

- On **April 15, 1912** the **Titanic** **sank**, costing the lives of **1502 out of 2224** passengers and crew.

- Unfortunately, there **weren’t enough lifeboats** for everyone onboard, causing a **disproportionate** number of **deaths**.

- While there was some element of **luck** involved in surviving, it seems some **groups of people** were more likely to **survive** than others.
- So, the primary objectives are to:

  -	Do a statistical analysis of **how** some group of **people** were **survived** more than others.

  - Perform an Exploratory Data Analysis of titanic dataset with **visualizations** and **storytelling**.

![image](https://raw.githubusercontent.com/brij0707/EDA-for-Titanic-Data-Set/main/images/Titanic_lifeboat.jpg)

---
<a name = Section3></a>
# **3. Installing and Importing Libraries**
---
```python
!pip install -q datascience          # A package that is required by pandas-profiling library
!pip install -q pandas-profiling

#-------------------------------------------------------------------------------------------------------------------------------
import pandas as pd                                                 # Importing for panel data analysis
from pandas_profiling import ProfileReport                          # Importing Pandas Profiling (To generate Univariate Analysis) 
pd.set_option('display.max_columns', None)                          # Unfolding hidden features if the cardinality is high      
pd.set_option('display.max_colwidth', None)                         # Unfolding the max feature width for better clearity      
pd.set_option('display.max_rows', None)                             # Unfolding hidden data points if the cardinality is high
pd.set_option('mode.chained_assignment', None)                      # Removing restriction over chained assignments operations
pd.set_option('display.float_format', lambda x: '%.5f' % x)         # To suppress scientific notation over exponential values
#-------------------------------------------------------------------------------------------------------------------------------
import numpy as np                                                  # Importing package numpys (For Numerical Python)
#-------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt                                     # Importing pyplot interface of matplotlib                                             
import seaborn as sns                                               # Importing seaborn library for interactive visualization
%matplotlib inline
#-------------------------------------------------------------------------------------------------------------------------------
import warnings                                                     # Importing warning to disable runtime warnings
warnings.filterwarnings("ignore")                                   # Warnings will appear only once
```

---
<a name = Section4></a>
# **4. Data Acquisition & Description**
---


- The dataset consists of the information about people boarding the famous RMS Titanic.

| Records | Features | Dataset Size |
| :-- | :-- | :-- |
| 891 | 12 | 58.9 KB | 

| ID | Feature Name | Description of the feature |
| :-- | :--| :--| 
|01| **PassengerId**   | Identity of the passenger                                    |
|02| **Survived**      | Whether the passenger survived or not                 |
|03| **Pclass**        | Class of the ticket holder                            |
|04| **Name**          | Name of the passenger                                 |
|05| **Sex**           | Sex of the passenger                                  |
|06| **Age**           | Age of the passenger                                  |
|07| **SibSp**     | Siblings and/or spouse travelling with passenger |
|08| **Parch**     | Parents and/or children travelling with passenger|
|09| **Ticket**        | Ticket number                                         |
|10| **Fare**          | Price of the ticket                                   |
|11| **Cabin**         | Cabin number                                          |
|12| **Embarked**     | Port of Embarkation                                   |

```python
# Loading the data set
data = pd.read_csv(filepath_or_buffer='https://github.com/brij0707/EDA-for-Titanic-Data-Set/blob/df8618af38917ee4753fe711ef3a310d851bca52/titanic%20dataset.csv')
print('Shape of the data set:', data.shape)
data.head() 				# This is going to show top 5 Rows fromt he data frame
```

