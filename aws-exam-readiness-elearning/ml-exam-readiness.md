# Exam Readiness: AWS Certified Machine Learning - Speciality

## Intro

Focus is on how to navigate and prepare to take the exam.

## Exam Overview and Test Taking Strategies

The AWS Certified Machine Learning – Specialty exam is designed to validate your ability to build, train, tune, and deploy ML models using the AWS Cloud. This includes your ability to:

* Choose the right approach for your business problem
* Use the right AWS services to implement ML solutions
* Design and implement scalable, cost-optimized, reliable, and secure ML solutions

Exam Breakdown

* 36% Modeling
* 20% Data Engineering
* 24% Exploratory data analysis
* 20 % ML Implementation and Operations

Use these strategies when answering questions:


Read and understand the question before reading answer options (pretend the answer options aren't even there at first).


* Identify the key phrases and qualifiers in the question.
* Try to answer the question before even looking at the answer choices, then see if any of those answer choices match your original answer.
* Eliminate answer options based on what you know about the question, including the key phrases and qualifiers you highlighted earlier.
* If you still don't know the answer, consider flagging the question and moving on to easier questions. But remember to answer all questions before the time is up on the exam, as there are no penalties for guessing.

## Domain 1: Data Engineering

### Domain 1.1 Create data repositories for machine learning

* Store data in a central repository
* Data lake is a key solution to this strategy. Store structured and unstructured data.
    * AWS lake formation is your data lake solution
    * S3 is your storage solution

* S3
    * Use storage classes to manage solution cost
    * Integrated with SageMaker
    * Amazon FSX for Lustre to speed up data access from s3 for multi SageMaker runs, builds filesystem from s3 data

Topics:

* AWS Lake Formation
* Amazon S3 (as storage for a data lake)
* Amazon FSx for Lustre
* Amazon EFS
* Amazon EBS volumes
* Amazon S3 lifecycle configuration
* Amazon S3 data storage options

### Domain 1.2 Identify and implement a data ingestion solution

* Batch processing and stream processing are two different kinds of data ingestion
* Kinesis is the platform for streaming data on AWS

Topics:

* Amazon Kinesis Data Streams
* Amazon Kinesis Data Firehose
* Amazon Kinesis Data Analytics
* Amazon Kinesis Video Streams
* AWS Glue
* Apache Kafka

### Domain 1.3 Identify and implement a data transformation solution

* Raw ingested data is often not ML ready
    * Needs to be cleansed (including deduplication), incomplete data must be handled, attributes must be standardized, etc.
* Apache Spark on Amazon EMR useful for transformation
* Glue for batch data prep
* Can use a single source of data on s3 for ad hoc analysis with athena, integrate with data warehouse using redshift spectrum, visualize with quicksight, and build ML models using sagemaker.

Topics:

* Apache Spark on Amazon EMR
* Apache Spark and Amazon SageMaker
* AWS Glue


## Domain 2: Exploratory Data Analysis

### Domain 2.1 Sanatize and prepare data for modeling

* Use descriptive statistics to better understand your data 
    * Overall stats (number of features, number of on=bservations)
    * Multivariate statistics (corellations and relationships between your attributes)
    * Attribute statistics (for numeric attributes no the mean, standard deviation, variance, min and max values)

* More on multivariate
    * Correlated features can affect model performance
    * Use scatter plots to visualize relationships between numerical variables
    * Correlation matrices help you quantify the linear relationships between variables

* Sanitize data
    * Standardize lamguage and grammar
    * Make sure the data is on the same scale
    * Make sure a column does not include multiple features

* Deal with outliers
* Deal with missing data
    * Remove rows or columns with missing data
    * Impute the missing value

Related Topics:

* Dataset generation
    * Amazon SageMaker Ground Truth
    * Amazon Mechanical Turk 
    * Amazon Kinesis Data Analytics
    * Amazon Kinesis Video Streams

* Data augmentation
* Descriptive statistics
* Informative statistics
* Handling missing values and outliers


### Domain 2.2 Perform Feature Engineering

Feature engineering

* Dimensionality reduction
* Transformation of numerical features (multinomial or polynomial transformation)
* Handle categorical data
    * Binary - transform to 1 or 0
    * Ordinal categoricals (e.g. garden size) - provide a mapping function, use business insight to determine scale to map to
    * One hot encode nominal ordinals
* Handle numerical data
    * Scale numeric features to prevent specific features to have an outsized influence based on scale alone

Topics

* Scaling
* Normalizing
* Dimensionality reduction
* Date formatting
* One-hot encoding

### Domain 2.3 Analyze and Visualize Data for ML

Visualization - answers questions like...

* What's the range of the data?
* What's the peak of the data?
* Are there any outliers?
* Are there any interesting patterns in the data?

Techniques

* Scatter plot - visualize the relationship between two features (plot feature 1 vs feature 2)
* Histogram - distribution of values for a single feature


Topics:

* Scatter plots
* Box plots
* Histograms
* Scatter matrix
* Correlation matrix
* Heatmaps
* Confusion matrix

