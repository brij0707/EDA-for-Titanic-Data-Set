# EDA-for-Titanic-Data-Set
We will do the basic EDA on Titanic Dataset
---

---
# **Table of Contents**
---

**1.** [**Introduction**](#Section1)<br>
**2.** [**Problem Statement**](#Section2)<br>
**3.** [**Installing & Importing Libraries**](#Section3)<br>
  
**4.** [**Data Acquisition & Description**](#Section4)<br>
  - **4.1** [**Data Description**](#Section41)
  - **4.2** [**Data Information**](#Section42)

**5.** [**Data Pre-Profiling**](#Section5)<br>
**6.** [**Data Cleaning**](#Section6)<br>
**7.** [**Data Post-Profiling**](#Section7)<br>
**8.** [**Exploratory Data Analysis**](#Section8)<br>
**9.** [**Conclusion**](#Section9)<br>
 
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
### Pandas Profiling
- For **quick analysis** pandas profiling is very handy.

- It generates profile reports from a pandas DataFrame.

- For each feature **statistics** are presented in an interactive HTML report.

- We can use Pandas Profiling or Sweetviz for this Profiling

```python

profile = ProfileReport(df=data)
profile.to_file(output_file='Pre Profiling Report.html')    # Saving the report
print('Accomplished!')                                  
```
![image](https://raw.githubusercontent.com/brij0707/EDA-for-Titanic-Data-Set/main/images/Pandas%20Profiling%20Report%20Overview.jpg)
Here you can see the overview of the data loaded. At the top in yellow collow highlighted texts are the various other tabs. Benefit of these is that you can understabd the data that you are using in one go. This report is best way to start working on your data. This is a next step deep dive to understand what could have been missed while you tried to udnerstand the data in first place.

![image](https://raw.githubusercontent.com/brij0707/EDA-for-Titanic-Data-Set/main/images/Pandas%20Profiling%20Report%20Alert.jpg)
Here you see alerts that can basically trigger the crucial findings from dataset that could possibly play a vital role in analysis and you have to take care of some to get a better result. For Example , High Cardinality refers to Numbers, Missing refers to data that is missing in the data set. Although other tabs are also there where you can look for missing values in each column.

**OR**

### SweetViz Profiling
This is similar to Pandas but a bit faster than it.
```python
!pip install sweetviz													# installing Sweetviz
import sweetviz as sv
sweet_report = sv.analyze(data)									# Analysing the dataframe
sweet_report.show_html('sweetviz_report.html')	  # saving Report
```
This is a complete view of data at one place something very similar to Pandas report.
![image](https://raw.githubusercontent.com/brij0707/EDA-for-Titanic-Data-Set/main/images/Sweetviz%20report.png)

### **Observations From Pandas Profiling:**

- There are **891** observations with **12** features. Most of the features have **categorical** data.

- Only **342** passengers out of **891** survived the accident.

- **Name**, **Ticket**, and **Cabin** features have high cardinality and are uniformly distributed.

- **PassengerId** is uniformly distributed

- A lot of **zeros** are present in **Fare**, **Sibsp** and **Parch** features.

- There are **no duplicate** rows in the dataset.

- We can observe that **8.1%** of data in cells is **missing**.

- **Fare** feature is highly skewed towards right.

- **Age** feature is faily symmetrical.

- We can observe that the **Age** feature **missing values** and **Cabin** feature has **missing values**.

- **Embarked** feature has just **2 missing values**.

- For detailed information, check the **Report file uploaded**.

- We will perform **cleaning** operations on our data based on the observations made from the profiling report.

---
<a name = Section6></a>
# **6. Data Cleaning**
---
|Feature|Data Type|Missing Proportion|Solution|
|:--:|:--:|:--:|:--|
|Age|float64|19.9%|Replace with median.|
|Embarked|object|0.2%|Replace with mode.|
|Cabin|object|77.1%|Drop the feature.|

```python
# Filling the missing values of Embarked feature with the mode of the feature.
data['Embarked'] = data['Embarked'].fillna(value=data['Embarked'].mode()[0])

# Filling the missing values of Age feature with the median age.
data['Age'].fillna(value=data['Age'].median(), inplace=True)

# Dropping the Cabin feature
data.drop(labels='Cabin', axis=1, inplace=True)

data.head()
```

---
<a name = Section7></a>
# **7. Data Post-Profiling**
---

- Now that we have cleansed the data, the dataset does not contain missing values.

- So, the profiling report which we have generated after preprocessing will give us more beneficial insights.

```python
sweet_report = sv.analyze(data)									# Analysing the dataframe
sweet_report.show_html('sweetviz_report_post_Profiling.html')	  # saving Report
```

### **Observations:**

- You can compare the two reports, i.e **Pre Profiling Report.html** and **Post Profiling Report.html**.

- Observations in Post Profiling Report.html:

  - In the Dataset info, **Total Missing** = **0.0%**

  - Number of **features** = **11**

  - You can see the difference in the **Age** feature in both the reports.

  - A lot of zeros are present in **Sibsp** and **Parch** features. They won't be removed as they are necessary.

  - We can observe that **Pclass** and **Fare** are highly **correlated** to each other **inversely**.

  - A lot of **inverse correlations** are observed among the features.

  - For detailed information, check the **Post Profiling Report.html** file.

- We can now begin the Exploratory Data Analysis.

---
<a name = Section8></a>
# **8. Exploratory Data Analysis**
---
**NOTE**:  

- Before diving further, we will **create** some **new features** that will be useful for analyzing the data.

- These features will be **FamilySize** and **Title**.

- The **FamilySize** will describe the frequency of family members.

- The **Title** will describe salutation of the passenger.

```python
# Creating a new feature FamilySize from Sibsp and Parch
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# Creating Title feature from Name
data['Title'] = data['Name'].str.extract(pat=' ([A-Za-z]+)\.', expand=False) # logic here is we are looking for the Alphabet characters followed by **.** 'full stop'

data.head()
```

```python
# CHecking how many Types of unique title do we have here
data.Title.nunique()
```

The crosstab() function is used to compute a simple cross tabulation of two (or more) factors. By default computes a frequency table of the factors unless an array of values and an aggregation function are passed.

```python
# Creating a crosstab between Sex and Title
pd.crosstab(index=data['Sex'], columns=data['Title'])
```

- There are **a lot of titles** for passengers. We will **simplify** these into selected categories.

- We will arrange the Males and Females into Mr, Mrs, Master, and Miss and put the **neutral** titles as **Other**

```python
# Rearranging titles into common titles
data['Title'].replace(to_replace=['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don'],
                      value=['Miss', 'Miss', 'Miss', 'Other', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr'],
                      inplace=True)
```

```python
# Now Checking it again
# Creating a crosstab between Sex and Title again
pd.crosstab(index=data['Sex'], columns=data['Title'])
```
![image](https://raw.githubusercontent.com/brij0707/EDA-for-Titanic-Data-Set/main/images/Cross%20tab.jpg)

- We can see that **gender** and **titles** **match** with each other (**Mr** and **Master** for **males**, **Miss** and **Mrs** for **females**)

- We can use this feature to check if a **title** of a person played an important role in their **survival**.

## **Question**: What is the relationship between age and the title of the passengers?

We will use standard definitions for titles according to Google and Wikipedia:

- **Mr** denotes an **adult man** (Age>18) (regardless of marital status)

- **Mrs** denotes an **adult woman** (Age>18) who is **married**.

- **Master** is an English honorific for **boys** and **young** men.

- **Miss** is an English language honorific used only for an **unmarried** **woman**.

- We will now **visualize** and **compare** the various titles based on this **standard**.

```python
# Plot a catplot for Age comparing the title of passengers
sns.catplot(x="Age", y="Title", data=data, size=7, aspect=3)

# Doing Beautification
plt.xticks(size=12)
plt.yticks(size=12)
plt.xlabel(xlabel='Age', size=14)
plt.ylabel(ylabel='Title', size=14)
plt.title(label='Age concerning Title of Passengers', size=16)
# plt.grid(b=True)

# Display the plot
plt.show()
```
![image](https://raw.githubusercontent.com/brij0707/EDA-for-Titanic-Data-Set/main/images/Title%20VS%20Age.jpg)

### **Observations**:

- **Titles** of passengers **don't match** the standard considered by us.

- This is because there are some **Masters** with **age** around **25 years** and **Misters** as young as **10 years** old.

- **Mrs** can be debated based on the **age** of the female gender.

- Some **males** and **females** (**Age<18**) can have marital status as **married** that explains their title as Mr and Mrs but such **data** is **not available** to us.

## **Question**: Does the title play an important role in the survival of the passengers?

```python
# Instantiate a figure of size of 15 x 7 inches
fig = plt.figure(figsize=[15, 7])

# Creating countplot of title vs survive
ax = sns.countplot(y='Title', hue='Survived', data=data, palette='hls')

# Adding percentages to the bars
total = data.shape[0]
for p in ax.patches:
    percentage = '{:.2f}%'.format(100 * p.get_width() / total)
    x = p.get_x() + p.get_width()
    y = p.get_y() + p.get_height() / 2
    ax.annotate(percentage, (x, y))

# Adding some cosmetics
plt.yticks(size=12)
plt.xticks(size=12)
plt.xlabel(xlabel="Count", size=14)
plt.ylabel(ylabel='Title', size=14)
plt.title(label="Count plot for Title concerning with survival", size=16)
plt.legend(labels=["Didn't Survive", "Survive"])


# Display the figure
plt.show()
```

![image](https://raw.githubusercontent.com/brij0707/EDA-for-Titanic-Data-Set/main/images/Title%20VS%20Count%20of%20servived.jpg)

### **Observations**:

- **Mrs** and **Miss** have particularly **higher** **survival** rate as compared to the rest of the titles.

- We can see that the passengers with the title "**Mr**" **died** the **most**.

- There are rarely any passengers with the **Other** title. We can't conclude if they were given more priority during the rescue.


## **Question**: Which gender category is more likely to survive, Male or Female?

```python
# Instantiate a figure of size of 15 x 7 inches
fig = plt.figure(figsize=(7, 7))

# Creating countplot of Sex vs Survived
ax = sns.countplot(x='Sex', hue='Survived', data=data, palette='Dark2')

# Adding percentages to the bars
total = data.shape[0]
for p in ax.patches:
    percentage = '{:.2f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width()/5
    y = p.get_y() + p.get_height()
    ax.annotate(percentage, (x, y))
    
# Adding some cosmetics - ticks, labels, title, legend and grid.
plt.xticks(ticks=[0, 1], labels=["Male", "Female"], fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(xlabel="Sex", size=14)
plt.ylabel(ylabel='Count', size=14)
plt.title(label='Comparison of male and female survivors', size=16)
plt.legend(labels=["Didn't Survive", "Survived"])

# Display the figure
plt.show()
```
![image](https://raw.githubusercontent.com/brij0707/EDA-for-Titanic-Data-Set/main/images/Count%20VS%20Sex.jpg)

### **Observations**:

- We can observe that a **significant** number of **males**  **didn't survive** the accident (**468/577 males died**).

- On contrary, the **female** **survival** rate is noticibly **higher** than the males (Only **81/314 females died**).
## **Question**: What is the rate of survival among different classes of passengers?
```python
# Instantiate a figure of size of 7 x 7 inches
f = plt.figure(figsize=(7, 7))

# Plotting countplot of Pclass vs survived
ax = sns.countplot(x='Pclass', hue='Survived', data=data, palette='rocket')

# Adding percentages to the bars
for p in ax.patches:
    percentage = '{:.2f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 5
    y = p.get_y() + p.get_height()
    ax.annotate(percentage, (x, y))

# Adding some cosmetics - ticks, labels, title, legend and grid.
plt.xticks(ticks=[0, 1, 2], labels=["1st", "2nd", "3rd"], fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(xlabel="Pclass", size=14)
plt.ylabel(ylabel='Count', size=14)
plt.title(label='Comparison of survival of classes of passengers', size=16)
plt.legend(labels=["Didn't Survive", "Survived"])
plt.grid(b=True)

# Display the figure
plt.show()
```

![image](https://raw.githubusercontent.com/brij0707/EDA-for-Titanic-Data-Set/main/images/Count%20VS%20PClass.jpg)
### **Observations**:

- **Majority** of passengers are from **3rd** class.

- A significant amount of **3rd** class passengers **did not survive** during the shipwreck.

- This creates a concern that **3rd** class passengers were given **less** **priority** to the rescue than the rest of the passengers.

- Passengers from the **1st** **class** have the **highest** **survival rate** than the **2nd class** than the **3rd class** passengers.

## **Question**: What is the survival rate considering the Embarked variable?

```python
# Entering port names in a list - ports.
ports = ["Southampton", "Cherbourg", "Queenstown"]

# Instantiate a figure of size of 15 x 7 inches with 2 subplots.
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,7))

# Plotting a countplot of embarked concerning survival in the first subplot.
sns.countplot(x='Embarked', data=data, hue='Survived', palette='summer', ax=ax[0])

# Adding percentages to the bars.
for p in ax[0].patches:
    percentage = '{:.2f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width()/5
    y = p.get_y() + p.get_height()
    ax[0].annotate(percentage, (x, y))

# Adding some cosmetics - ticks, labels, title, legend and grid to the countplot.
ax[0].set_title(label='Frequency distribution of Survived vs Embarked', size=16)
ax[0].set_ylabel(ylabel='Number of Passengers', size=14)
ax[0].set_xlabel(xlabel='Port of Embarkment', size=14)
ax[0].set_xticklabels(labels=ports, fontsize=12)
ax[0].set_yticklabels(labels=np.arange(0,500,50), fontsize=12)
ax[0].legend(labels=["Didnt Survive", 'Survived'])

# CPlotting a pointplot of embarked concerning survival in the second subplot.
sns.pointplot(x='Embarked', y='Survived', data=data, color='green', ax=ax[1])

# Adding some cosmetics - ticks, labels, title, and grid to the pointplot.
ax[1].set_title(label='Proportion of Survived concerning Embarked', size=16)
ax[1].set_ylabel(ylabel='Survived', size=14)
ax[1].set_xlabel(xlabel='Port of Embarkment', size=14)
ax[1].set_xticklabels(labels=ports, fontsize=12)
ax[1].set_yticklabels(labels=np.round(np.arange(0.25,0.7,0.05),2), fontsize=12)


# Setting a super title for Surival vs Embarked
plt.suptitle(t='Frequency and proportion distribution of Survived concerning Embarked', size=18)

# Display the output
plt.show()

```
![image](https://raw.githubusercontent.com/brij0707/EDA-for-Titanic-Data-Set/main/images/Count%20VS%20Embarked.jpg)

## **Question**: How does Age play an important role for the survival of a passenger?.

```python
# Instantiate a figure of size of 20 x 7 inches with 
# 2 subplots with 2:1 width ratio
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 7), gridspec_kw={'width_ratios': [2, 1]})

# Creating a kdeplot of Age against survived in subplot 1
sns.kdeplot(x='Age', shade=True, hue='Survived', data=data, ax=ax[0])

# Adding some cosmetics - ticks, labels, title, legend and grid to the kdeplot.
ax[0].set_title(label='Distribution of Age feature against Survival', size=16)
ax[0].set_ylabel(ylabel='Survived', size=14)
ax[0].set_xlabel(xlabel='Age', size=14)
ax[0].set_xticklabels(labels=np.arange(-20,100,20), fontsize=12)
ax[0].set_yticklabels(labels=np.arange(0, 0.035, 0.005), fontsize=12)
ax[0].legend(labels=['Survived', "Didn't Survive"])
#ax[0].grid(b=True)

# Creating a boxplot of Age against survived in subplot 2
sns.boxplot(y='Age', x='Survived', data=data, palette='muted', ax=ax[1])

# Adding some cosmetics - ticks, labels, title, and grid to the boxplot.
ax[1].set_title(label='Boxplot of Age feature against Survival', size=16)
ax[1].set_ylabel(ylabel='Age', size=14)
ax[1].set_xlabel(xlabel='Survived', size=14)
ax[1].set_xticklabels(labels=["Didn't Survive", "Survived"], fontsize=12)
ax[1].set_yticklabels(labels=np.arange(-10, 100, 10), fontsize=12)
#ax[1].grid(b=True)

# Setting a super title for the Age concerning survival plots
plt.suptitle(t='Influence of Age on Survival', size=18, y=1.0)

# Display the figures
plt.show()
```
![image](https://raw.githubusercontent.com/brij0707/EDA-for-Titanic-Data-Set/main/images/Age%20vs%20survival%20.jpg)

### **Observations**:

- From the left graph, we can see that a lot of **senior citizens** (Age>60) died in the **accident**.

- From both the graphs, we observe that the loss of children with **Age less than 10** is not in significant amount.

- From the right graph, we see that the **eldest** and the **youngest** person on the journey **survived** the shipwreck.

- From both the graphs, we observe that **majority** of **victims** were from the **20-40 years** age group.
#### More analysis can done as per your requirement

### **Observation On Hypothesis**
- On studying the previous questions, we observe that an **overwhelming** percentage of **women** & **children** have **survived** the titanic disaster.

- This reminds us of the infamous line from the Titanic movie - ***Women and Children first.***

- But we should also take note that there were a **significant amount** of **men** present during the voyage.

- **76%** of **females** **survived** whereas only **16%** of males **survived**.

- Also the **survival** rate for a **male** is **very low** **irrespective** of the **class**.

- Almost **all women in Pclass 1 and Pclass 2 survived** and nearly **all men in Pclass 2 and Pclass 3 died**.


<a name = Section9></a>
### **9 Conclusion**

- We have seen the **impact** of various factors such as **Gender**, **Age**, **Port of Embarkment**, **FamilySize** on the **rate of survival**.

- **Women** have a **higher** chances of **survival** than men.

- Passengers from **20-40 years** of age had a **very low survival** rate.

- But since a lot of **Age** data was **missing**, we **can't conclude** how much impact Age really had on survival.

- The **class** of the **passenger** seems to have played an **important** role in the rescue operation.

- Passengers who were from the **1st class** were given **more priority** during the **rescue** than **rest** of the **passengers**.

- Passengers who boarded from the **Cherbourg** port had a **higher survival** rate in contrast to the other two ports.

- A lot of **3rd class passengers** from the **Southampton** port **died** in the accident.
