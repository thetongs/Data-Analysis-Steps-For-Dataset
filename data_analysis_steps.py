# ## Step 1 : Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Step 2 : Load Dataset
dataset = pd.read_csv("dataset.csv")
dataset.head()


# ## Step 3 : General Information Of Dataset
# Information of type of object, column names, data type of each column, index information,
# memory usage and non-null count
dataset.info()

# Less Information
dataset.info(verbose = False)


# ## Step 4 : Check Data Type Of Features and Change If Required
# Data types of each column
print(dataset.dtypes)

# Change object datatype into category if number 
# of categories are less than 5 percent of the total number of values
cols = dataset.select_dtypes(include='object').columns

for col in cols:
    ratio = len(dataset[col].value_counts()) / len(dataset)
    if ratio < 0.05:
        dataset.loc[:, col] = dataset.loc[:, col].astype('category')

dataset.info()

# Change datatype if you think 
dataset.Gender = dataset.Gender.astype('category')
dataset.dtypes


# ## Step 5 : Missing Data Management
# Check missing percentage of each column
NAN = [(clm_name, dataset[clm_name].isna().mean()*100) for clm_name in dataset]
NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])
NAN

## Columns that has 30 or more than 30 percent missing values
# will be drop from dataset.
threshold = len(dataset) * 0.7
dataset = dataset.dropna(axis = 1, thresh = threshold)

NAN = [(clm_name, dataset[clm_name].isna().mean()*100) for clm_name in dataset]
NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])
NAN

# Missing data visualization
plt.figure(figsize = (10, 6))
sns.heatmap(dataset.isna(), 
        cbar = 'viridis',
        yticklabels = False)
plt.show()
plt.close()

# Handle missing values
dataset.Gender = dataset['Gender'].fillna(dataset.Gender.mode()[0])
dataset.isna().sum()


# ## Step 6 : Find and Deal With Duplicate Data
# Total count of duplicate rows
dataset.duplicated().sum()

# Drop rows which are duplicate
dataset.drop_duplicates(inplace = True)
dataset.duplicated().sum()


# ## Step 7 : Find Outliers and Deal
# Using Box Plot
dataset.plot(kind = 'box',
                subplots = True,
                layout = (7, 2),
                figsize = (15, 20))

plt.show()
plt.close()

# Check and Set Range
low = np.quantile(dataset.Salary, 0.05)
high = np.quantile(dataset.Salary, 0.95)
dataset = dataset[dataset.Salary.between(low, high)]
dataset

# Using Box Plot
dataset.plot(kind = 'box',
                subplots = True,
                layout = (7, 2),
                figsize = (15, 20))

plt.show()
plt.close()


# ## Step 8 : Check Imbalance dataset
import warnings
warnings.filterwarnings('ignore')

sns.countplot(dataset.Gender)
plt.show()
plt.close()


## For getting ratio of classes
# 
from sklearn.utils import compute_class_weight

## 
#
class_weight = compute_class_weight('balanced', 
                    dataset['Gender'].unique() , 
                    dataset['Gender'])

## Get in ratio
#
print("Classes : {}".format(dataset['Gender'].unique()))
print("Ratio : {}".format(class_weight))


# Handle Imbalanced
from imblearn.over_sampling import RandomOverSampler
over_sampler = RandomOverSampler(sampling_strategy='minority')


X, Y = over_sampler.fit_resample(dataset[['Year_Of_Experience', 'Gender']], 
                            dataset[['Salary']])
dataset = pd.concat([X, Y], axis=1)
sns.countplot(pd.concat([X, Y], axis=1).Gender)
plt.show()
plt.close()


# ## Step 9 : Statistical Information
# Information like count, mean, std, min, 25%, 50%, 75% and max
dataset.describe()

# To get statistical information of string column 
dataset.describe(include = 'all')


# ## Step 10 : Heatmap to check correlation
# Using heatmap
features = dataset.columns.to_list()
sns.heatmap(dataset[features].corr(), 
                cmap = 'coolwarm',
                annot = True)

plt.title("Heat map to show any correlation between features")
plt.xlabel("")
plt.ylabel("")
plt.show()
plt.close()


# ## Step 11 : Summarization and filter dataset
# Value count
dataset.Gender.value_counts()

# Filter
condition = dataset.Salary > 30000
dataset[condition]

# Filter
condition = dataset.Year_Of_Experience.between(3, 10)
dataset[condition]

# Sorting
dataset.sort_values(by = "Salary")


# ## Step 12 : Measure of Central Dependency
# Mean, Median and Mode
from scipy import stats

np.mean(dataset.Salary)
np.median(dataset.Salary)
stats.mode(dataset.Gender)[0]


# ## Step 13 : Measure of dispersion
# Variance and Standard Deviation
np.var(dataset.Salary)
# Variance is large - Data points are spread-out.

np.std(dataset.Salary)
# Standard deviation is large - Data are widely dispersed.


# ## Step 14 : Measure of position
# Percentile
percen = np.percentile(a = dataset.Salary, q = 10)
print("10th Percentile is at : {}".format(percen))

# Quartile
quar = np.percentile(dataset.Salary, 25, interpolation = 'midpoint')   # 1st quartile
print("1st quartile : {}".format(quar))


# ## Step 15 : Moments
# Skewness
from scipy.stats import skew

skew(dataset.Salary)
# Skewness by value → Skewness value > 0

# Kurtosis
from scipy.stats import kurtosis

kurtosis(dataset.Salary)
# Platykurtic → Kurtosis > 0 


# ## Step 16 : Problem statement