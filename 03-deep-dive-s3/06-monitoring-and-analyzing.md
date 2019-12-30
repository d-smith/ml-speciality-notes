# Monitoring and Analyzing Your Data in Amazon S3

## Storage Class Analytics

* What data is hot? What is infrequently accessed?
* Storage class analysis - data driven storage management of Amazon S3
    * Daily update report of object access patterns to help you understand what's hot, warm, cold.
    * Storage access pattern analysis and visualization

## Amazon QuickSight

* Analyze and visualize s3 analytics data using prebuild visualizations


## Explore using Amazon CloudWatch with Amazon S3

* Understand and improve the performance of applications that use Amazon S3
* Two types of metrics: Storage, and Request

Storage Metrics

* Reported once per day (no cose): BucketSizeBytes, NumberOfObjects
* Monitor the number of bytes stored in your bucket
* Can alert on these

Request Metrics

* One minute intervals, additional costs
* AllRequests, Put/Get/List/Delete/Head/Post Requests
* BytesDownload/Uploaded, 4xx/5xxErrors, FirstByteLatency, TotalRequestLatency

CLoudWatch Logs

* Integrate with CloudTrail, combine with CloudWatch alarms
* For example, configure a notification if someone changes a bucket policy or lifecycle rule