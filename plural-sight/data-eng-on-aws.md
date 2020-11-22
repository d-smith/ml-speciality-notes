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