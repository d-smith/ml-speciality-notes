# CRISP-DM on AWS

## Introduction

* CRISP-DM: CRoss Industry Standard Process - Data Mining
* Framework for data science projects
* Created in 1996

The phases

* Business understanding - User story/business objective, frame problem - decide if not ML problem or suitable for ML. If suitable continue with CRISP-DM
* Data understanding - collection, verifying, exploring the data. Identify the data to use, where it is housed, how to access it. Examine the data, looking for patterns and applying standard statistical techniques to understand the properties of the data set. Understand the quality of the dataset and decide if it can be used.
* Data prep - prepare the data set, do feature engineering. 
* Modeling - select and create ML model. Tune the model, if more data or diff features iterate back to the data prep stage.
* Evaluation - what are the outcomes, what are the false positives, false negatives, examine the performance in the context of the business objectives. Decide to deploy or to start the entire cycle again.
* Deployment - planning the deployment, maint and monitoring, final report, project review.

Cyclical endeavor

## Phase 1 Business Understanding

* Understanding business requirements
    * form a business question or problem that must be solved
    * highlight the critical features of the project (people, resources, etc)
* Analyzing supporting information
    * list required resources and assumptions
    * analyze associated risks
    * plan for contingincies
    * compare costs and benefits
* Convert to ML objective
    * review machine learning question
    * creat technical data mining objective
    * define criteria for successful project outcome
* Create a project plan
    * number and duration of stages
    * dependencies
    * risks
    * goals
    * evaluation methods
    * tools and techniques

## Phase 2 - Data Understanding

* Data collection
    * detail various sources and steps to extract data 
    * analyze data for additional requirements - do i select all the data, specific fields, are there missing values, do i need to encode or decode
    * consider other data sources - non-electronic, business owner insights and domain knowledge
* Data properties
    * describe the data (structured vs unstructured), amount of data, metadata properties, complexity of the data
    * find key features and realtionships in the data
    * use tool and techniques to explore data properties - descriptive statistics, meanings of the properties, determine main attributes, check for correlations between them
    * Data quaity
        * verify attributes
        * identify missing data, errors, how to impute missing data
        * reveal inconsistencies
        * report all the problems in the data quality task and how to solve the.

    * AWS tools for data quality and visualization - amazon athena, amazon quicksight, aws glue
* Consider other data sources

Glue

* Managed ETL service
* 3 components
    * Build your data catalog
    * Generate and edit transformations
    * schedule and run jobs

Athena

* Interactive query service
* Run interactive sql quaries on s3 data
* schema on read 
* suports ansi sql operators and functions
* serverless

Amazon QuickSIght

* Cloud powered bi service
* scale to hundreds of thousands of users
* 1/10th the cost of traditional BI solutions
* Secure sharing and collaboration (storyboard)

## Phases 3 - Data Preparation 

Consists of two tasks

* Final dataset selection
    * Decide the consolidated, raw data to use for your project
    * Understand your constraints - total size, included and excluded columns (using previous analysis of the important and less important attributes), record selection, data types
* Preparing the data
    * Cleaning
    * Transforming
    * Merging
    * FOrmatted

Cleaning

* How is missing data handled?
    * Drop rows with missing values
    * Adding a default value or mean value for missing data
    * Using statistical methods to calculate the value (e.g. regression)
* Clean attributes with corrupt data or variable noise

Transformation

* Derive additional attributes from the original attributes
    * use hours, months, years as separate attributes instead of data string, or provide an ordered array based on timestamp for sequential models
    * One hot encoding to convert strings to numerical values (or vice versa)
* Normalization
* Attribute transformation

Merging

* May want to merge before cleaning/transforming
* May want to revisit data understanding after merging

Formatting

* Format your dataset to accomodate your modeling tool needs
    * Rearrange attributes
    * Randomly shuffle data
    * Remove constraints of the modeling tools, for example remove unicode characters

## Phase 4 - Modeling

phase 3 and 4 must be balanced using an iterative approach,
 think of these as a single phase

Modeling phases 

* Model selection and creation
    
* Model testing plan
* Parameter tuning/testing

Model Selection and Creation - Identify

* Modeling technique
    * regression for numeric values, random forests for multi-class classification, RNN to predict sequences
    * Also depends on your framework
* Constraints of modeling technique and tools
* Ways in which constraints tie back to data preparation phase

Generating a Model Testing Plan

* Before training your model, define how to test your model's accuracy.
* Split into training and test set
* Determine split of model training set - 30% test/70% traing
* Model evaluation training criterion
    * regression - mean squared
    * classification - compare the output class and true class, evaluate precision and recall

Building the Model

* Traing the model
* Tweak the model for better performance
* Build multiple models with different parameter settings - goal is to train faster and obtain better performance
* Describe the trained model and report on findings

### Tools for Prep and Modeling

Amazon EMR + Spark

* Managed hadoop
* Use with spark mlib dataframe based API for machine learning
* Use ipython notebooks, zeppelin notebooks, or r studio
* scala, python, r, java, SQL supported
* cost savings - leverage spot instances for task nodes

Amazon EC2 + Deep Learning AMI

* Deep learning AMI
    * preinstalled gpu cuda support for training
    * preinstalled deep learning frameworks lik MXNet, TendorFlow, Caffe2, Torch, Keras, Theano, etc
    * Includes python anaconda data science platform
* Can also [install R studio on aws](https://aws.amazon.com/blogs/big-data/running-r-on-aws/)