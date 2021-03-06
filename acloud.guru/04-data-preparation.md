# Data Preparation

## Intro and Concepts

Machine Learning Cycle

* We want to Clean and Prepare Our Data
* First, we must understand our data

Data prep

* The process of transforming a dataset using different techniques to prepare it for model training and testing
* Changing our dataset so it is ready for machine learning

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
* No single rule that universally applies for transformation
* Many different approaches, each of which can impact on the outcode of your analysis


## Text Feature Engineering

Feature engineering

* Transforming attributes within our data to make them more useful within our model for the problen at hand. Feature engineering is often compared to an art.

Text feature engineering: transforming text within our data so machine learning can better analyze it. Splitting text into bite size pieces.

* Uses include natural language processing, speech recognition, text-to-speech
* Data like words and phrases, spoken languages, dialogue between two people
* Sources like magazines, newspapers, academic papers

Bag of Words

* Tokenizes raw text and create a statistical representation of text
* Break up text by white space into single words

N-Gram

* An extension of bag-of-words which produces groups of words of n size
* Breaks up text by white space into groups of words
* 1-gram same as bag of words
* Think of it as a sliding window (no overlap)
* 1 unigram, bigram 2, trigram 3

Orthogonal Sparse Bigram (OSB)

* Creates groups of words of size n and outputs every pair of words that includes the first word
* Creates groups of words that always include the first word
    * Orthogonal - when two things are independent of each other  
    * Sparse - scattered or thinly distributed
    * Bigram - 2 gram or two words  
* Used a delimiter of some sort

OSB in action - OSB, size 4

* Raw: "he is a jedi and he will save us"
* OSB size 4: {"he_is", "he__a", "he___jedi"}{"is_a", "is__jedi, etc} - note the use of underscores as place holders.

TF-IDF: Term frequency - invese document frequency

* Represents how important a word or words are to a given set of text by providing appropriate weights to terms that are common and less common in the text.
* Shows us the popularity of a word or words by making common words like "the" or "and" less important
* Term frequency - how frequent does a word appear?
* Document frequency - number of documents in which the which terms appear
* Inverse - makes common words less meaningful

Vectorized tf-idf

* (number of documents, number of unique n-grams)
* See [this resource](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

| Problem | Transformation | Why |
| -- | -- | -- |
| Matching phrases in spam emails | n-gram | easier to compare whole phrases like "click here now" or "you are a winner" |
| Determine the subject matter of multiple PDFs | tf-idf, orthogonal sparse bigram | filter less important words in PDFs. Find common word combinations repeated throughout PDFs |

Standard Transformations

* Remove punctuation: sometimes removing punctuation is a good idea if we do not need them (but leave punc inside words)
* Lowercase transformation: using lowercase transformation can help standardize raw text

Cartesian Product Transformation

* Creates a new feature from the combination of two or more text or categorical values.
* Combining sets of words together
    * Multiply a set of words by another set of words

Feature Engineering Dates

* Feature engineer dates so we can extract more information from them for our models
* Extracted information answers...
    * Was it a weekend or week day?
    * Was the date the end of a quarter?
    * What was the season?
    * Was the date a holiday?
    * Was it during busines  hours?
    * Was the world cup taking place on this date?
* Can include columns that include the extracted info, e.g. is_weekend, day_of_week, etc

## Numeric Feature Engineering

Numeric Feature Engineering

* Transforming numeric values within our data so Machine Learning algorithms can better analyze them.

Common Techniques

* Feature Scaling - changes numeric values so all values are on the same scale
    * Normalization
    * Standardization
* Binning - changes numeric values into groups or buckets of similar values
    * Quantile binning aims to assign the same number of features to each bin

Scaling Example

* Home price
    * Conceptually place the prices along the number line
    * Assign largest value to 1, smallest to 0, the rest placed using x'    = x - min(x) / max(x) - min(x)
    


Standardization

* Outliers can throw off normalization
* Put the average at 0, and offset the rest of the values using the Z score
    * Smooths out the standard deviation
    * Calculate z score as:
        * xbar = mean
        * sigma - standard deviation
        * x is the observation
        z = x - xbar / sigma

Scaling Summary

* Scaling features
    * Required for many algorithms like linear/non-linear regression, clustering, neural networks, and more. 
    * Scaling features depends on the algoritms you use
* Normalization
    * rescales values from 0 to 1
    * does not handle outliers
* Standardization
    * Rescales values by making the values of each feature in the data have zero mean and is less affected by outliers
* You can always translate back to the original scale

Binning

* Takes numeric values and sets ranges to group values by
* Use when the feature does not have a linear relationship with the target attribute / prediction
* Can use the bins as categorical variables
* Can end up with irregular bins if there's an uneven distribution

Quantile Binning

* Equal parts - group
    * Create groups such that there are even distribution between the bis

Binning Summary

* Binning is used to group together values to reduce the effects of minor observation errors
* Quantile binning is used to group together values to reduce the effects of minor observation errors
* Optimum number of bins - depends on the characteristics of the variables and its relationshiop to the target. This is best determined through experimentation.

## Other Feature Engineering

Image Feature Engineering

* Extracting useful information from images before using them in ML algorithms

Problem: does an image of a handwritten represent a number

* Features - break image into a grid, black in the grid: 1, otherwise 0
* Compare them, NIST datset

Audio Feature Engineering

* Extrating useful information from sounds and audio before processing them with ML algorithms
* Audio stream - amplitude and time, sample the audio stream to get amplitude at time

Data Input - Built-In Algorithms

* File
    * Loads all of the data from s3 directly onto the training instance volumes
    * CSV, JSON, Parquet, Image files
* Pipe
    * Datasets are streamed directly from Amazon S3
        * faster start times
        * better throughput
    * recordIO-protobuf (creates tensor)


Resource:

* [Common dataset formats](https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html)

Layering

* Usually there are multiple layers of transformations done to properly prepare your data.

## Handling Missing Data

Motivation: missing values in your dataset can interfere with analysis and model predictions.

* Can be represented in many different ways - null, NaN, NA, None, etc
* Handling missing values is an important data preparation step

Why are the values missing in the first place?

* Missing values correlated with another feature?
* Missing at Random (MAR)
    * Propensity for a data point to be missing is not related to the missing data, but is related to some of the observed data
* Missing Completely at Random (MCAR)
    * The fact that a certain value is missing has nothing to do with its hypothetical value and with the values of other variables
* Missing not at Random (MNAR)
    * Two possible reasons are that missing value depends on the hypothetical value or missing value is dependent on some other variable's value.

Techniques to Handle Missing Values

| Technique | Why this works | Ease of use |
| -- | -- | -- |
| Supervised learning | Predicts missing values based on the values of other features | Most difficult, can yield best results |
| Mean | The average value | Quick and easy, results can vary |
| Median | Orders values then chooses value in the middle | Quick and easy, results can vary |
| Mode | Most common value | Quick and easy, results can vary |
| Dropping rows | Remove rows missing values | Easiest but can dramatically change datasets |

Replacing data in known as imputation.

## Feature Selection

Selecting the most relevant features from your data to prevent over-complicating the analysis, resolving potential inaccuracies, and removes irrelevant features or repeated information.

* An intuitive step that humans take to reduce the number of features

Principal Component Analysis

* Unsupervised learning algorithm that reduces the number of features while still retaining as much information as possible
* Reduces the number of features in a dataset

Feature Selection Use Case

| Problem | Technique | Why |
| -- | -- | -- |
|  Data is too large due to the large number of features | Principle component analysis (PCA) | Algorithm that reduces the total number of features |
| Useless features that do not help solve ML problem | Remove features that do not help solve the problem |

## AWS Data Preparation Helper Tools

Data Preparation

* AWS Glue
* SageMaker
* EMR
* Athena
* Data Pipeline

Glue

* Input Data Source
    * S3, DynamoDB, RDS, Redshift, Database on EC2
* Crawler to glean data types, schema or structure of dataset
* Data Catalog - metadata with data types and info about your data set
* Set up jobs to run python or scala code to transform our data, do feature selection, etc
* Can upload code directly or edit generated code in the console
* Run on demand or on schedule or when another service is triggered
* Output to Athena, EMR, S3, Redshift

Data Catalog

* Databases and Crawlers

ETL

* Jobs to transform the data
    * Input from a table
    * Good for reusable jobs you run over and over or on a schedule
    * Jobs type
        * Spark jobs - runs on managed cluster spun up in the background
            * Pick python or scala
                * Python uses PySpark
                * Generate script for us
                * Provide our own
                * Start from scratch
        * Python shell scripts
            * More freedom to use traditional python libraries
* Can do Apache Zeppelin and Jupyter notebook transformations too
    * Ad hoc, not run as a job
    * Hosted in SageMaker

SageMaker

* Can create Jupyter notebooks on a fully managed server
* Use common python libraries or via package managers
* More for ad hoc

EMR

* Managed hadoop, run standard hadoop ecosystem tools
    * spark, presto, mahout, hive, jupyter, tensorflow, mxnet, etc.
* Can run the entire ML process in EMR, but more cumbersome than SageMaker
* Can integrate the SageMaker SDK for spark into the EMR environment, can use together
    * Train models in EMR
    * Host model endpoints in SageMaker

Athena

* Run SQL queries on your S3 data
    * Uses data catalog

Data Pipeline

* Move data between data sources
    * DB -> pipeline -> etl -> target data source


Which Service To Use?

| Datasource | Data preparation tool | Why |
| -- | -- | -- |
| S3, Redshift, RDS, DynamoDB, On premise DB | AWS Glue | Use Python or Scala to transform data and output data into s3 |
| S3 | Athena | Query data and output results into s3, which can transform |
| EMR | PySpark/Hive in EMR | Transform petabyyes of distributed data and output data into s3 |
| RDS, EMR, DynamoDB, Redshift | Data pipeline | Setup ec2 instances to transform data and output data into s3, can use languages other than python and scala |

## Data Prep Lab

* Data source - set of random data from [randomuser.me](https://randomuser.me/)
* From this data file...
    * Which percentage of users are male vs female?
    * What are the ages of most users?
    * Of the users, how many are in their 20s, 30s, 40s, etc?
    * Convert the data to CSV and store it in s3
    * Transform gender feature to a binary value - male 1, female 0.
 

## Exam Tips

Data Prep

* Know why it is important
* Understand the different techniques

Categorical Encoding

* Know why CE is used for certain ML algorithms
* Understand diff between ordinal and nominal categorical features
* Understand categorical data is qualitative and continuous data in quantitative
* Know what one-hot encoding is and when to use it

Numeric Feature Engineering

* Know what numeric feature engineering is any why it is important
* Know different techniques used for feature engineering numeric data
* Know the different types of feature scaling and when they should be used
    * Normalization
    * Standarization
* Know what binning is and when it should be used (when small diffs in value are not important)

Text Feature Engineering

* Know what it is and why it is important
* Know different techniques 
    * N-gram
    * Othogonal sparse bigram
    * Term frequency - invest document frequency (tf-idf)
    * Removing punctuation
    * Lowercase transformatio
    * Cartesian product
* Understand why feature engineering dates is important
* Know the questions we can answer when dates are transformed
    * weekday, weekend, etc

Other Feature Engineering

* audio, video, etc

Handling Missing Values

* Know why handling missing values is an important step in data prep
* Know the different techniques used for handling missing values
* Understand implications of dropping rows
* Understand what data imputation is

Feature Selection

* Know what it is and why it is important
* Understand diff between feature selection and PCA

Data Preparation Tools

* Know the service that allow you to transform data
* Know what a data catalog, crawlers, and jobs are in AWS glue
* Be abel to identify different AWS services and when to use one transformation tool over another.
