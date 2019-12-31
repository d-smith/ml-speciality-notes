# Using Amazon S3

Data Consistency Model

* Read-after-write consistency for puts of new objects
    * Eventually consistent data model for read-after-write if a HEAD or GET request is made before the object exists
* Overwrite or delete and existing model - it will take some time for this to be observed by all clients
    * Eventually consistent data model for read-after-write

Accessing Your Data

* S3 console - max file size for upload is 73GB
* AWS CLI
* AWS SDK

Data Transfer

* Direct connect (higher throughput, more secure by not going over public internet)
* AWS Storage Gateway (file gateway mode, supports NFS, smb)
* Amazon Kinesis Data Firehose, Kinesis Video Streams, Kinesis Data Streams
* Amazon S3 Transfer Acceleration - leverages CloudFront
* AWS Snowball, AWS Snowball Edge, AWS Snowmobile
* Third party connectors

Accessing Buckets and Objects

* Buckets and objects are resources with unique URI
* Two types - path style URLs and Virtual hosted style URL
    * Path style: http://region-specific-endpoint/bucket-name/object-key
        * For example http://s3-eu-west-1.amazonaws.com/mybucket/image1.jpg
        * Exception: us-east-1 endpoint s3.amazonaws.com
    * Virtual-sosted style: http://bucketname.s3.amazonaws.com/object key
        * Example: http://mybucket.s3.amazonaws.com/image1.jpg

TODO: get details on virtual hosted style and https - bucket names with periods won't work due to CNAME alias, for example www.example.com is CNAME alias for www.example.com.s3.amazonaws.com

How a Request is Routed

* DNS is used to route s3 requests
    * Name  resolved to one or more IP addresses that represent s3 facilities that can process the request
    * Requests arriving at the wrong region receive redirects
    * Bad requests - 4xx responses

Operations on Objects

* PUT
    * Upload objects in single operation (up to 5 GB)
    * Multipart upload for objects up to 5 TB
        * Recommended if size > 100 MB
        * All parts retained until upload completed or aborted
        * Enable clean up of multipart uploads in bucket settings

* COPY
    * Create copies of objects
    * Rename objects by copying original object and deleting the original
    * Move objects across Amazon S3 locations
    * Update object metadata

* GET
    * Retrieve the whole object at once or in parts
        * Parts via range of bytes in the header

* DELETE
    * Single or multiple
    * No versioning - permanent removal
    * Versioning enable - permanently or create a delete marker (only object name), can recover the delete marker from the object key. Permanent delete via deleting each version (key plus version in delete request)

Listing Objects

* Can enumerate the keys in a bucket, can filter by key prefix, can list keys by prefixes, can list common key prefixes

Listing objects examples

* `aws s3api list-objects-v2 --bucket mybucket --query "Contents[].{Key: Key}"`
* `aws s3api list-objects-v2 --bucket mybucket --prefix 2017/ --query "Contents[].{Key: Key}"`
*`aws s3api list-objects-v2 --bucket mybucket --prefix 2017/scores/ --delimiter /  --query "Contents[].{Key: Key}"`

Presigned URLs: PUT

* Grant upload permission for an object without giving the uploader AWS permissions
* To create a presigned url you provide:
    * Security credentials
    * Bucket name
    * Object key
    * HTTP method
    * Expiration
* Can also provide presigned URLs to get an object

Cross-Origin Resource Sharing

* Allow client applications in one domain to work with resources in another domain
* e.g. web font hosted in one domain loaded from another domain
* In s3, can create a CORS config xml file in your bucket to grant access to other domains
* See [here](https://docs.aws.amazon.com/sdk-for-javascript/v2/developer-guide/cors.html) for more details




