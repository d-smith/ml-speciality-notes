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