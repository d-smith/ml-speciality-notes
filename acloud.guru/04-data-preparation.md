# Data Preparation

## Intro and Concepts

Machine Learning Cycle

* We want to Clean and Prepare Out Data
* First, we must understand our data

Data prep

* the process of transforming a dataset using different techniques to prepare it for model training and testing
* changing our dataset so it is ready for machine learning

Typical Activities

* Categorical Encoding
* Feature Engineering
* Handling Missing Values

Options for Data Prep

* SageMaker and Jupyter Notebooks (adhoc)
* ETL Jobs in AWS Glue (reusable)

## Categorical Encoding

Categorical Encoding

* The process of manipulating categorical variables when ML algorithms expect numerical values as inputs
* Changing category values in our dataset to numbers

When to encode

| Problem | Algorithm | Encoding |
| -- | -- | -- |
| Predicting the price of a home | linear regression | encoding necessary |
| Determine whether given text is about sports or not | Naive Bayes | encoding not necessary |
| Detecting malignancy in radiology images | CNN | encoding necessary |

Pick the algorithm, then figure out if you need to encode your variables.

Examples:

* Color: { green, purple, blue} - multi-categorical, nominal
* Evil: {true, false} - binary categorical, nominal
* Size: { L > M > S } - ordinal (order does matter)
* Note all values are discrete, not continuous

More examples:

* Binary label - can transform 0 and 1 to N and Y
* Ordinal - S,M,L need to make sure numeric mapping is ordered, may have to determine appropriate values
* Nominal values (condo, house, apartment)
    * Encoding variables into integers is a bad idea (algorithm might treat them as ordered)
    * Better to one-hot encode

One-Hot Encoding

* Transform nominal categorical features and creates new binary columns for each observation
* Know your data before you encode
    * One hot encoding is not always a good choice when there are many, many categories
    * Using techniques like grouping by similarity could create fewer overall categories before encoding
    * Mapping rare values to "other" can help reduce overall number of new columns created

Summary

* Encoding needs are ML algorithm specific
* Text into numbers
* No single rule that universally applies for transformatopm
* Many differnt approach, each of which can impact on the outcode of your analysis
