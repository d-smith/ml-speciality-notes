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

## Data Understanding

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