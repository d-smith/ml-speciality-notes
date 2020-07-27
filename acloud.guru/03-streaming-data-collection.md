# Streaming Data Collection

## Intro

ML cycle: part of the fetch, clean, prepare part of the cycle

## Streaming Data Collection Concepts

Sources of open source datasets

* Kaggle
* UCI machine learning repository
* Registry of open data on AWS
* Google Big Query

Streaming Data

* From sensors and iot devices
* Stock prices
* Click stream data
* Game player interaction

Processing

* Via stream processing
* Via batches

Kinesis Family

* Kinesis Data Streams
* Kinesis Firehose
* Kinesis Video Streams
* Kinesis Data Analytics

## Kinesis Data Streams

Data Producers -> Kinesis Streams -> Data Consumers

* Data carried in shards
* Consumers can be EC2 apps, Lambda, Kinesis data analytics, EMR with Apache Spark

Shards

* Partition keys used to map data to shards
* Sequence numbers used to order the data in each shard
* Consists of a sequence of data records, ingested at 1000 records per seconds
* Default limit of 500
* Data record: seq no, partition key, data blob up to 1 MB
* Transient data store - retained from 24 hours up to 7 days

Interacting with Kinesis Data Stream

* KPL
* KCL
* Kinesis API (AWS SDK)

KPL

* provides layer of abstraction for ingesting data
* automatic and configurable retry mechanism
* additional processing delay can occur for higher packing effeciencies and better performance
* java wrapper

Kinesis API

* low level api calls
* stream creations, resharding, and putting and getting records handled by user
* no delays in procesing
* any aws sdk

When should you use kinesis data streams?

* Needs to be processed by consumers
* real time analytics
* feed into other services in real time
* some actions need to occur on your data
* storing data is optional
* data retention is important

Use Cases

* process and evaluate log files immediately
* real time data analytics

## Kinesis Firehose

Data Producers -> Processing Tools (lambda, optional) -> Storage

* Delivery service for streaming data to storage
* Can also add in s3 events to lambda, then push the data somewhere else too like dynamodb
* Redshift, ElasticSearch, Splunk, S3 storage supported

When should you use Firehose?

* Want to easily collect streaming data
* Processing is optioal
* Final destination is s3 or one of the others supported
* Data retention is not important

Use Cases

* Stream and store data from devices
* Create ETL jobs on streaming data

## Kinesis Video Streams

For streaming video into the AWS cloud

* Producers - video streams, cameras, audiostream, rader
* Data consumers - ec2 continous and batch consumers
* Can store in s3 after processing or store first then process

When to use?

* Need to process real-time streaming video data (audio, images, radar)
* Batch process and store streaming video
* Feed streaming data into other AWS services

Use Case Example

* Amazon cloud camera
* Detects movement, sends out alert

## Kinesis Data Analytics

Continously read and process streaming data in real time

* Can use SQL to analyze
* Streaming input from kinesis data stream and kinesis firehose
* Can store and visualize

When to use?

* When you want to run sql queries on streaming data
* Construct applications that provide insighe on your data (using Apache flink for example)
* Create metrics. dashboards, monitoring, notifications, and alarms
* Output query results into s3 (other AWS datasources)

Use cases

* Responsive real-time analytics
* Stream ETL jobs - clean, enrich, organize, transform before it lands into data warehouse or data lake

| Task at hand | which kinesis services to use? | why? |
| ---- | ---- | ---- |
| Need to stream apache log files directly from ec2 instances and store them in Redshift | Firehose | Allows direct storage route to FIrehose  |
| Stream live video coverage of a sporting event to distribute to customers in near real-time | Kinesis Video Streams | Video Streams used to process real-time video streaming data |
| Need to transform real-time streaming data and immediately feed into a custom ML application | Kinesis Streams | Built to allow streaming huge amounts of data to be processed then stored or fed into custom applications or other AWS services |
| Need to query real time data, create metric graphs, and store output into s3 | Kinesis data analytics | built in support for sql queries on streaming data, then store or feed output into other AWS services |

Shards and data retention - only a concern for Kinesis Streams

## Exam Tips

Streaming data collection

* Understand how to get data from public or in house data sets and load it into AWS
* Know the different ways to upload into s3

Kinesis family

* Know what each service is and how it processes/handles streaming data
* Know what shards are, what a data record is,time retention period for a shard
* Know the difference in the KPL, KCL, and Kinesis API
* For a given scenario know which streaming Kinesis service to use


Lab Notes

* https://randomuser.me


Resources:

* https://www.youtube.com/watch?v=jKPlGznbfZ0
* https://www.youtube.com/watch?v=0AGNcZfYkzw
* https://www.youtube.com/watch?v=EzxRtfSKlUA
* https://aws.amazon.com/blogs/machine-learning/analyze-live-video-at-scale-in-real-time-using-amazon-kinesis-video-streams-and-amazon-sagemaker/
* https://www.youtube.com/watch?v=dNp1emFFGbU
* https://aws.amazon.com/blogs/big-data/create-real-time-clickstream-sessions-and-run-analytics-with-amazon-kinesis-data-analytics-aws-glue-and-amazon-athena/
* https://aws.amazon.com/blogs/big-data/joining-and-enriching-streaming-data-on-amazon-kinesis/
* https://d0.awsstatic.com/whitepapers/whitepaper-streaming-data-solutions-on-aws-with-amazon-kinesis.pdf
* https://www.youtube.com/watch?v=M8jVTI0wHFM
