# Demystifying the AWS Certified Machine Learning Specialty Exam

## Understanding the Exam

Specialty exams - goes very deep in a specific area

* Also need to have knowledge of the domain associated with the area, as well as common tools and frameworks, etc.

Content Domains

* Data Engineering - 20%
* Exploratory Data Analysis - 24%
* Modeling - 36%
* Machine Learning & Operations - 20%


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

## Modeling Domain

Core concepts

Modeling

* Frame business problems as machine learning problems
* Select the appropriate models for a given machine learning problem
* Train machine learning models
* Perform hyperparameter optimization
* Evaluate machine learning models

High level approaches

Supervised Learning

* Classification - binary, multiclass
* Regression - predict a value

Unsupervised Learning

* Clustering
* Diminsionality reduction
* Anamoly Detection

SageMaker Algorithms

* Linear learner
    * supervised, classification or regression
    * optional data normalization step
    * based onSGD
* XGBoost
    * classification, regression, and ranking problems
    * can be leveraged as both built in algorithm or as a local framework
    * Based on gradient boosted trees algorithm
* K-nearest neightbors (kNN)
    * Supervised
    * Solves classification and regression
    * Sampling, diminsionality reduction, index building
* Random cut forest
    * Unsupervised learning
    * Identifying anonolies and outliers
* Image classification
    * Supervised, multilabel classification porblems in computer vision
    * CNN architecture
    * Can use transfer learning
* Object detection
* Semantic segmentation
    * Supervised approach, computer vision segmentationEm3rg3nt
     at a pixel level
    * Identify object and their shapes withing an image
* BlazingText
    * Supports word2vec (unsupervised)
        * Supports continuous bag of words, skip-gram, and batch skip gram
        * Solves sentiment analysis, machine translation, and entity recognition
    * Text classification
        * Supervised
        * Web searches, information retrieval, ranking, and document classification
* Sequence to sequence 
    * Supervised
    * Input, outputs - sequence of tokens
    * Text and audio
    * Machine translation, text summarization, speech to text
    * Uses CNNs and RNNs
* Object2Vec
    * Provides diminsionality reduction
    * Recommentation engine, multi-label document classiciation
    * Uses embedding approach from work2vec

Evaluation Models

* Overfitting
    * Improve via using fewer features, dim reduction, adding additional training samples
* Underfitting
    * Improve via more features, reduce regularization parameters

Classication model metrics

* accuracy
* precision
* recall
* F1
* AUC

Regression

* RMSE
* Mean absolute percentage error (MAPE)


## Machine Learning Implementation and Operations

* Recommend and implement the apprpriate machine learning services and features for a given problem
    * Data ingestion services
    * Data transformation
        * Amazon EMR
        * AWS Glue
        * Firehose transformation feature
    * Pretrained AI services (...most efficiently implement...)
        * Know the inputs and outputs
    * Data set analysis AI services
        * Forecast, fraud detector, personalize
    * Amazon SageMaker Service 
        * Augmented AI
    * Other compute approaches
        * EC2, ECS, EKS, EMR
* Deploy and operationalize machine learning solutions
    * SageMaker
        * SageMaker Hosting service
        * SageMaker Batch transform
    * Use multiple production variant for A/B testing of production hosting
    * Deployment steps for hosting services
        * Create model
        * Create endpoint configuration
        * Create endpoint
* Apply basic AWS security practices to machine learning solutions
    * Securing SageMaker - how to use in your own VPC
    * Find compliance standards
        * Artifacts service
* Build ML learning solutions for scalability, performance, availability, resiliency, and fault tolerance.
    * High availability 
        * Deplouments can be configured to autoscale
        * Endpoints should span multiple AZs
        * Loose coupling promites fault tolerance
        * AWS services to monitor and audit



