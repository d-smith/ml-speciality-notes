# Optimizing Performance in Amazon S3

## Best Practices - Performance

* For faster uploads over long distances, use transfer acceleration
* For faster uploads for large objects, use multipart upload
* optimize get performance with range gets and integrate with amazon cloudfront
* To optimize listing objects, use Amazon S3 inventory

Request rate performance

* Amazon s3 scales a single partition to 3,500 put/post/delete combined requests and 5,500 requests per second
* You can increase performance by using multiple prefixes to help scale the performance if needed
    * Initial bucket create - single partition
    * Amazon will create new partition if request rates not satisfied by single partition
    * If you know if advance you can contact support to partition the bucket before hitting 5xx errors

Parallelization

* Parallel uploads
* Multipart uploads
    * Upload parts independently and in any order
    * Retransmit only failed parts
    * Use when object size reaches 100 MB, or over error-prone networks
    * Remove to use AbortIncompleteMultipartUpload and consider configuring 'clean up incomplete multipart uploads' in your lifecycle config
    * Consider TransferManager in AWS SDK for Java
* Multipart Downloads
    * Use TransferManager
    * Use byte range in get requests

Multipart Upload Advantages

* Improved throughput
* Quick recovery from network issues
* Pause and resume object uploads
* Begin an upload before you know the final object size

## Amazon S3 Select

Retrieve only a subset of data from an object based on a SQL expression

* Faster performance compared to retrieving the entire object
* Available as an API - no infrastructure or administration
* Pay as you go, only paying for what you retrieve
* Works and scales like GET requests
* Perform SQL queries using the presto connector, AWS SDK, API, AWS CLI, or S3 console

Formats

* Input
    * Format
        * Delimited text (CSV, TSV)
        * JSON
        * Apache Parquet
    * Compression
        * GZIP
        * BZIP2

* Output
    * delimited text or JSON

Supports a [subset of SQL](https://docs.aws.amazon.com/AmazonS3/latest/API/API_SelectObjectContent.html)

Tools

* AWS Tools and SDKs
* AWS CLI
* S3 console (but data limited to 40MB)

## CloudFront

Faster downloads via CDN caching and geographic distribution

## Transfer Acceleration

* Change your endpoint, not your code.
* Takes advantage of global edge location to route to s3 over optimize network paths
* Only charges if it will speed you up

## List Inventory Optimization

* Inventory tool can provide lists of objects and metadata
* Faster to parse, cheaper