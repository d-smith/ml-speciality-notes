# AWS Storage Gateway Types Deep Dive

## Storage Gateway Architecture

![architecture](./architecture.jpg)

### Data Writes

1. Application writes data using iSCSI or NFS
2. Gateway appliance stores blocks locally, compresses it asynchronously and uploads the changed data securely
3. Storage gateway service stored the compressed data in s3, glacier, or EBS with AES-256 encryption

### Data Reads

1. Reads from the gateway appliance vis iSCSI or NFS protocol
2. Gateway returns the requested data from local storage (if stored locally)
3. If not in the local store, the data is requested from the gateway service. The gateway service retrieves the compressed data from s3, glacier, or EBS
4. The gateway appliance receives data from the storage gateway service, decompresses it, stores it locally, and responds to the application

On premises storage needs to provide local disk for cache and upload buffer. Size recommendations [here](https://docs.aws.amazon.com/storagegateway/latest/userguide/resource-gateway-limits.html)

![cache and buffer sizes](./cache-and-buffer-sizes.jpg)

Cached Data States

* Dirty - written but not yet uploaded. Cannot be evicted.
* Hot data - recently read or written
* Cold data - less recently used data

Upload Buffer

* Staging area for the gateway before it uploads the data to AWS

## Storage Gateways: File

![file gateway](./file-gateway.jpg)

* One to one mapping from files to objects.
    * Enables native access to S3 capabilities on the objects
* Controls on how your data is accessed
    * NFS/files options export - allowed clients, read-only/read-write exports, user squasing, default object ownship (uid/gid), default object permissions
    * S3 options per buckets - IAM role for access, storage class, object encryption with KMS, guess mime type
* Metadata and worload automation
    * Cached metadata and object inventory maintained on the applicance
    * RefreshCache API
    * CloudWatch upload events from gateway
* Reads
    * Reads byte ranges, has hueristics to do read ahead, etc. 
* Writes
    * Optimizes uploads via multipart uploads and similar techniques, dedup of writes, etc
    * Write opt example - appending data to a file: just appended data is uploaded to gateway service, gateway service uses the data in the cloud and the newly uploaded append data to create a new object
* Refresh Cache API Use Cases
    * Sync in cloud workloads to on premise view of files
    * S3 cross region replication - allow active active, surface replicated data to on prem
    * AWS snowball use

## Storage Gateways: Volume

