# Glacier Overview

## Amazon Glacier

* Object storage
* Optimized for infrequently used data, cold data
* Common workloads
    * Backup
    * Archival
    * Compliance
    * Data laks

Features:

* 11 nines durability
* Data filtering for data lakes
* Storage costs start at $0.004 per GB per month
* Several data retrieval options for quick access
* IAM Permissions
* Vault access policies
* Vault lock - WORM via lockable policy
* Retrieve 5% of your data per day free of charge - can use range retrieval to optimize your 5% use
* Globally availability (multiple regions and AZs)

Data Model


* Archive
    * Can be any data
    * Single file or aggregate several into a zip
    * Single archive can be as large as 40 TB when stored via the glacier native API
    * Can store an unlimited number of archives
    * Archives are immutable
   * Upload, download, delete
* Vault
    * Container for storing archives
    * Name + region 
    * https://<region specific endpoint> /<account id>/vaults/<vault name>   
    * An unlimited number of archives can be stored in a vault

Glacier Entry Points

* AWS CLI, Amazon glacier SDK
* Use AWS and 3rd party ingestion tools
    * Direct transfer via storage gateway, direct connect, AWS snow family
* Store objects in S3, use lifecycle policies to migrate data to glacier

Benefits

* Durable, Available, Scalable
    * 99.999999999% durability
    * Data distributed across three geographically separated facilities
* Security and Compliance
    * AES-256 encryption and built in key mangement
    * Managed your own keys
    * PCS-DSS, HIPPA, FedRamp, SEC Rule 17-a-4(f)
* Query in Place
    * Run sophisticated analytics without extracting/moving data for analysis
    * Analyze using SQL
    * Glacier allows direct analysis on objects
* Flexible management
    * Classify, report, and visualize usage trends
    * Tag objects with custom metadata for efficient storage management
    * Analyze access patterns to design tiering and retention
    * Integrated with lambda for logging, alerts, and workflow automation
* Large ecosystem
    * ISVs and consulting partners
    * AWS market place

Glacier vs Tape

* Deliver a tape like customer experience with features, performance and cost model similar to on premises tape storage without all the hassles (such as media handling, complex technology refreshes, mass migrations, capacity planning)

Retrieval Options via Policy

* Expidated 1 - 5 minutes
* Standard 3 - 5 hours
* Bulk - 5 - 12 hours

Can set a policy of free tier, max allowed cost, unrestricted

Provisioned capacity - gaurantees of available capacity for expidited retrieval.