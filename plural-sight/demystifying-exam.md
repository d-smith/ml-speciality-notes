# Demystifying the AWS Certified Machine Learning Specialty Exam

## Understanding the Exam

Specialty exams - goes very deep in a specific area

* Also need to have knowledge of the domain associated with the area, as well as common tools and frameworks, etc.

Content Domains

* Data Engineering - 20%
* Exploratory Data Analysis - 24%
* Modeling - 36%
* Machine Learning & Operations


## Data Engineering Domain

Data repository approaches

 * Database
* Data warehouse
* Data lake

 Amazon FSx for Lustre - efficient fetching of data from s3 into a file system that acts as if it was local

* [SageMaker integration with lustre and efs](https://aws.amazon.com/about-aws/whats-new/2019/08/amazon-sagemaker-works-with-amazon-fsx-lustre-amazon-efs-model-training/)

Data ingestion approaches

 * Batch processing
    * Glue
    * AWS Batch
    * Database Migration Service
* Stream processing
 

## Exploratory Data Analysis

 
Subdomains

* Santitize and prepare data
    * Understand data terminology
    * AWS services for data prep
        * glue, kinesis, lambda, sagemaker (notebooks, groundtruth)
    * Model effect of correlation
        * effect of high correlation on model accuracy (exp regression)
    * Positive and negative correlation
    * Use of PCA to deal with feature correlation
    * Univariate statistics (min, max, mean, std dev, quartiles, etc)
    * Approaches for missing data
    * Impact of outliers
    * Standardize data scale
* Feature engineering
    * Categorical encoding
    * One-hot encoding and the problem it solves (magnitude problem with integers)
    * Different scaling techniques and when to use them
    * Data nornalization
* Analyze and prepare data for machine learning
    * Visualizations
        * Comparisons
            * Bar charts , line charts
        * Relationships
            * scatterplots
            * heat maps
        * Composition
            * Compositions based on category = pie charts
        * Distribution
            * Histogram
    * Visualization use cases
        * scatter plots to visualize correlation between two numerical values
        * Correlation matrices can visualize linear relationships across numerical features (heat maps)
        * Box plots give an overview of the univariate statistics for a feature - see outliers
        * Histograms can visualize the skewness of the dataset for a feature – Can be useful to see outliers
        * Heat maps can help to identify and rank features
 

Other things

* Variance inflation factor (VIF) - values over 5 are high correlated
* Detect outliers using Z score, range from IQR

