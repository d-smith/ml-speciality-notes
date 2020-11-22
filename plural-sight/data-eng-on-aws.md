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

