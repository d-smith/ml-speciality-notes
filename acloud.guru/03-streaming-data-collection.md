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

