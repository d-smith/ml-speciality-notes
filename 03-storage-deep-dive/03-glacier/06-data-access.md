# Data Access on Amazon Glacier

Retrieval

* Expidited - minutes, like on site tape
* Standard - 3 to 5 hours, good for DR
* Bulk - large amounts of data inexpensively, like offsite tape, 5 to 12 hours

Retrievals

* Reads are asynchronous
* Initiate the request, then access the object when it has been restored
    * You are given 24 hours when restoring via the Glacier API to access you archive
    * Intiate via S3 restore, you can assign a TTL to specify how many days the restored archive will live. Stored in S3 reduced reduncancy storage classs

Provisioned Capacity Units (PCUs)

* A way to reserve I/O capacity for expidited retrieves and guarantees restore requests will not be throttled, Note expidited retrievals are only accepted if there is sufficient capacity.
* One PCU can deliver up to 3 expidited retrievals and up to 250 MB in 5 minutes, and provide up to 150 MS/s of retrieval throughput for best case sequential reads.
* No size limit on archives
* 25 GB archive retrieval should complete in under 10 minutes

Retrieval Costs

* Low percentage of your total cost, retrieval of 5% or less very small cost, extreme use still roughly 20% of total cost.

Restores

* Move to glacier via lifecycle policy - can execute a restrore job 
* Console provides the ability to manually specify objects to restore to s3

Amazon Glacier Select

* Lets you directly filter delimited glacier objects using simple SQL expressions without having to retrieve the full objects
* Accelerate bulk ad hoc analytics while reducing overall cost
* Operates like a get request and integrates with the glacier SDK and the CLI
* Use cases
    * Pattern matching
    * Auditing
    * Big data integration
* Objects must be formatted as uncompressed delimited files, you must has a bucket with write permissions in the same region, you must have permissions to call get job output
* Similar to get request, can use the 3 retrieval options too
* Pricing
    * Data scanned
    * Data return
    * Request - initiation of the retrieval request
* Job kicked off, SNS notified when complete