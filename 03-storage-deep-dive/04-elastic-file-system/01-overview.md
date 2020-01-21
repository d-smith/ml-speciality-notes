# Deep Dive into Amazon Elastic File System - Overview

Cloud migration - what to do for on-premises NAS servers, used to host content repositories, development environments. media storage, user home directories. The file systems must be secure, scalable, and highly available.

# EFS Overview

Block storage - collection of data on a device. No native metadata, greate for high performance. E.g. EBS

File storage - collection of data in a file system. Metadata live in file system. E.g. EFS

Object storage - collection of data with built-in metadata. Defined by object IDs. Flexible, customizable meta data. E.g. S3

## EFS

* Fully managed file system service
* Provides availability, durability, and performance through a web interface
* Manages file system infrastructure
* Works with EC2
* Stores data and metadata across AZs
* Elastic capacity
* Billed only for capabity used

## When to Use EFS

* Apps running on EC2 that need a file system
* On premises file systems with multi-host attachments
* High-throughput application requirements (e.g. multi gigabit per second throughput rates)
* Applications that require high availability and durability

## Common Use Cases

* Content management
* Web serving
* Big data analytics
* Media and entertainment workflows
* Workflow management
* Container storage
* Database backups
* Home directories
* Software development tools