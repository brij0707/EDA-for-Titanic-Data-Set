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

<a name = Section41></a>
### **4.1 Data Description**

- In this section we will get **information about the data** and see some observations.

```python
data.describe()     # this will give us statistical details of all numerical columns
```
![image](https://raw.githubusercontent.com/brij0707/EDA-for-Titanic-Data-Set/main/images/describe.jpg)

### **Observations:**

> **Survived**: 
- More than 50% did not survive the accident.

> **Pclass**:
- There are a lot **more 3rd class** passengers than **1st and 2nd class**.
- We can also see that there are **more 2nd class** passengers **than 1st class** passengers.

> **SibSp**:
- More than **50%** of passengers are **not travelling** with their **siblings** or a **spouse**.
- There are **some** passengers who are travelling with as **maximum** as **8 siblings and spouse**.

> **Parch**:
- More than **75%** passengers are not travelling with a **parent** or **children**
- But there are some passengers who have a **maximum** number of **6 children** and/or **parents** with them on the ship.
- We observe that a vast majority of passengers **are not travelling** with their family members.

> **Age**:
- The **average age** of passengers is around **29 years** while the **minimum** and **maximum** ages are **0.4 years** and **80 years** respectively.
- There is some **missing** data in the **Age** feature.

> **Fare**:
- The **average price** of ticket seems to be **£32.2**. **Minimum** price of the ticket is recorded as **£0** and **maximum** price recorded as high as **£512.32**.
- More than **50%** of the passengers have paid atleast **£14**
- More than **75%** passengers have paid atleast **£7** for their ticket whereas **less than 25%** have paid for **more than £31**.
- We have to replace the minimum value in the **Fare** feature with a reasonable value.

<a name = Section42></a>
### **4.2 Data Information**

- In this section we will see the **information about the types of the features**.

```python
data.info()		# to get the layout of table; complete structure of table
```

![image](https://github.com/brij0707/EDA-for-Titanic-Data-Set/blob/37f6b0579050c404e563ea6e301610a62df0b6ba/images/info.jpg)

### **Observations:**

- The **data types** of all the features look appropriate.

- There are **missing** values present in the **Age**, **Cabin** and **Embarked** features.

- **Age** and **Cabin** have a significant amount of **missing values** which **requires** **further investigation**.

---
<a name = Section5></a>
# **5. Data Pre-Profiling**
---

- For **quick analysis** pandas profiling is very handy.

- It generates profile reports from a pandas DataFrame.

- For each feature **statistics** are presented in an interactive HTML report.

- We can use Pandas Profiling or Sweetviz for this Profiling

```python
# profile = ProfileReport(df=data)
# profile.to_file(output_file='Pre Profiling Report.html')
# print('Accomplished!')
```
**OR**

'''

'''
