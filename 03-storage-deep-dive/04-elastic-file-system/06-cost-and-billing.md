# EFS Cost and Billing

Pricing dimensions

* Storage (GB-month)
* Throughput
    * Bursting throughput (default)
    * Provisioned throughput (MB/s-Month)
* EFS file sync (Per GB into EFS)

Pay only for data stored, no minimums, up front fees, etc.

Example: store 500GB of data and ensure high availability and durability

* Provision two m4 ec2 instances with two 600 DB capacity EBS volumes copied to multiple AZs. Cost per month is $120 storage + $350 compute + $129 data transfer equals $599 per month
* EFS at $0.30/GB a month at 500 GB is $150 a month 