# Data Engineering with AWS for Machine Learning

## Data Characteristics

* Structured - relational, predefined schema, relationships
* Semi-structured - json, xml, key-value, document
* Unstructure - not structure at all, heterogenous, object storage

Batch and Stream Processing Characteristics

* Batch - data scope is querying or processing over all the data in the data set
    * Complex analysis
    * Data size is large batches
    * Performance is latencies in minutes to hours

* Stream
    * Higher velocity data
    * Continous data, processed over a small time window
    * Queries done on most recent data
    * Latencies in seconds or subsecobds

Application Characteristics

* Number of users
* Data volume
* Locality
* Performance
* Request rate
* Access
* Scale
* Economics
* Developer access

For the Exam

* Data characteristics help you decide on the repository services to us
* Batch and stream processing characteristics help you decide the services to use for ingestion
* Application characteristics help you decide the storage services to use

## Typical Dataflow for ML on AWS

Modern data analytics pipelines

![](./data-analytics-pipeline.png)

Data movement - extract from the source and ingest into a destination

Modern data workflow architecture

![](./data-workflow-arch.png)

## Data Storage Options on AWS

Amazon S3

* Amazon S3 Data Lakes
* Query in place services for big data analytics
* Primary data repository for machine learning tools
* S3 access points simplifies data sharing
* S3 batch process s3 objects in a simple straight forward manner
* S3 event notifications

Partitioning Data in Amazon S3

* Objects in buckets
* Object key is full path to object
* Partitioning is done on the path
    * By date is column
    * Tailor to your data access patterns
* Up to 5 TB object size, up to 10 tags

S3 Storage Classes

~[](./storage-classes.png)

* Object lifecycle management
    * Transition actions - change storage class
    * Expiration actions - once expired AWS will delete the object

S3 Security

* Encryption - client-side, server-side
    * Server Side
        * SSE-S3: encrypts s3 objects using keys handled and managed by AWS
        * SSE-C: you manage encryption keys and provide them on upload and download, amazon uses the key to encrypt and decrypt
        * SSE-KMS: customer manages the data key using KMS, higher security, audit train

Security

* User based or resource based
    * User: IAM user policies
    * Resource
        * Bucket policies
        * Object ACLs
        * Bucket ACLs

VPC Endpoints for S3

* Keep traffic from the VPC off the public internet, plus no need for gateway, etc.
* Access points only via virtual host style endpoints

EFS for Machine Learning

* Personalized environments: notebook files, training data, model artifact
* SageMaker integrates for training jobs

Amazon FSx for Lustre

* S3 integration: file system linked to an S3 bucket
* Amazon sagemaker - traiing jobs
* AWS Batch integration
* AWS parallel cluster integration

Elastic Block Store

* Multi-attach capablity, use for distributed training on up to 16 instances, 128 GPUs

Cost Effectiveness

* produce, class or tier, how much data, what region
* use the calculator to compare

Exam Tips

* S3 - stores objects, globally unique bucket names, max 5 TB, object keys = path, partitioning on the path, life cycle rules, vpc endpoints
* EFS - speeds training jobs
* EBS - higher availability

## Database Options for Machine Learning on AWS

### Amazon RDS

Fully managed scalable relational database service

* 6 engines
* Can connect to RDS databases from Jupyter Note using pymysql, etc.

### Amazon Aurora

Relational database built for the cloud

* Natively integrated with SageMaker and Amazon Comprehend
* Use SQL to apply machine learning to data in Aurora
    * Amazon Aurora Machine Learning

### DynamoDB

Serverless non-relatoinal, keu value store with single-digit millisecond response data

Analyze data in DDB using SageMaker for realtime predictions

![](./ddb-realtime-predictions.png)


### Amazon Redshift

Multi-parallel processing data warehouse

Can do JDBC connections to redshift from Jupyter notebooks

* Use psycopg2, e.g. !conda install --y -c anaconda psycopg2

### Amazon Document DB

Fast, scalable non-relational schema-free data for mongodb workloads

* Unique distributed data storage on s3

Amazon Translate and Transcibe output JSON

* Makes document DB a natural database to integrate with

### Exam Tips

Know

* The different database offerings from Amazon at a high level
* Different integration scenarios for machine learning use cases

## Using a Data Warehouse or Data Lake for ML on AWS

Data warehouse vs Data Lake

![](./dwdl.png)

Data Lake Architecture

![](./dlarch.png)

### Immutable Logs and Materialized

Immutable logs and materialized views are the foundation for s3 data lakes

Realtime analytics architecture

![](./realtime.png)

Batch and interactive analytics AI'd

![](./batch.png)


Batch, Interactive, Stream, and Real-time Analytics on an S3 Data Lake

![](./unified.png)

### Amazon Lake Formation

Ingest and cleaning

* What data sources
* S3 locations
* Map data to s3 locations
* ETL to load and clean the data

Security

* Configure access policies
* Metadata access policies
* Access from analytical services

Analtics

* set up the analytics capabilities, etc.

AWS Lake Formation Solution Stack

* Instantiate a common architecture
* Ingest and clean
    * Blueprints simplify ingest
    * ML transforms for data cleaning
* Security
    * Real time monitoring and integrated auditing
    * Centralized permissions
* Analytics  & ML
    * Comprehensive portfolio of integrated tools
    * Right tool for the right job

Lake formation built on top of glue

![](./lakeglue.png)

### Redshift

Some features...

* AQUA - accelerated query optimizer cache for redshift
* Spectrum
* Federated Query (against postgres and aurora in addition to redshift)

### Data Warehouse vs Data Lake

Data warehouse

* Relational data
* Schema on write
* Curated data
* Batch reporting, BI, visualizations

Data Lake

* Non-relational
* Schema on read
* Raw data
* ML, predictive analytics, data discovery/profiling

### Exam Tips

* DL and DW are the primary way to get data into machine learning services
* Differences between DW and DL
* Data lake advantages


From AWS

> A data lake is a centralized repository that allows you to store all your structured and unstructured data at any scale. You can store your data as-is, without having to first structure the data, and run different types of analytics—from dashboards and visualizations to big data processing, real-time analytics, and machine learning to guide better decisions

