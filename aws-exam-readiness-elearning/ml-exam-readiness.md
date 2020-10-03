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


