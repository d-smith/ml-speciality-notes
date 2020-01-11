# Ingesting Data into Amazon Glacier

Multiple entry points into amazon glacier

* lifecycle
* glacier API via SDKs
* console does not provide direct upload tool
* Can open a secure SSL tunnel
* Can trasfer data via direct connect

AWS Snow Family

* Data transfer appliances
* Snowball 50 tb (US) and 80 TB (everywhere)
    * Data encrypted end to end 
    * S3 compatible endpoint
    * 10 GB network interfaces
* Snowball Edge
    * 100 TB storage
    * Local compute equal to M4.4xlarge
    * Run lambda functions or process data
    * Amazon s3, hdfs, and nfs endpoints
* Snowmobile
    * 100 Petabyte shipping container
    * Interfaces up to 1 Tb/s (100PB in a few weeks)

Snow Family Use Cases

* Cloud Migration
* Disaster recovery
* Data center decommission
* Content distribution

Storage Gateway

* Hybrid storage service that enables on premises applications to seamlessly use AWS storage through conventional interfaces like iSCSI and NFS

Multipart Uploads

* Uploads objects 5GB or larger, use when object size is 100 MB or larger
* Part of the glacier API
* Uploads parts in any order, retry, etc.
* Initiate, upload parts, complete

Media Archive Use Case

* Move assests to AWS as part of offsite archive
* Next, on site assests like LTO archives move to AWS
* Then, direct connect to S3 and Glacier, and use lambdas to trigger workflows, AI services, etc.

