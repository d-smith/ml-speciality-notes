# Managing EBS Snapshots

An EBS-snapshot is a point in time backup of an EBS volume that is stored in Amazon S3.

* While EBS volumes are associated with a single AZ, Amazon S3 is a regional service the spans across AZs to provide 11 9's of durability.

Snapshots are incremental backups

* Only bytes that have changed from the last snapshot are changed.
    * First snapshot: all blocks on the volume that have been used are marked as a snapshot copied to s3, emptry blocks are not copied.
    * Volume is usable again once the create snapshot api returns
    * Second snapshot contains only new blocks or modified blocks that were part of the earlier snapshot.
    * Third snapshot - contains only new or changed blocks since snapshot two, only has references to unchanged blocks in snapshots one and two.
* When deleting snapshots, only the data not needed by any other snapshot is removed.
* Each snapshot contains all the information needed to restore your data (from the moment the snapshot was taken) to a new EBS volume.

Creating EBS Volumes from Snapshots

* The new volume begins as an exact replica of the original volume that was used to create the snapshot. The replicated volume loads data lazily, in the background, so you can being using it immediately.
* Data is loaded on demand, if you access data that hasn't been loaded yet, that data moves to the front of the queue of data to be copied to the new volume.

Sharing Snapshots

* Can copy to other regions
* Can grant other accounts access to snapshots, other account can then create EBS volumes from them.

Amazon Data Lifecycle Manager

* DLM autonates the backup of EBS volumes and the retention of those backups for as long as needd.
* This feature uses policies to definte backup schedules that specify when you want a snapshot to be taken and how often.
* You can also specify how many snapshots you want to retain - how long to retain, or how many copies.
* Specify tags to identify which EBS volumes should be backed up.
* DLM lets you snapshot every 12 or 24 hours. Use multiple tags to stagger backup start time using multiple policies to get more snapshots.

Considerations

* A lifecycle policy applies to any of the tags specified
* You can  use a volume tags in only one amazon DLM policy
* Snapshots are taken within one hour of the set start time
* Amazon DLM applies AWS tags when snapshots are created (snapshots are tagged when created, you can specify custom tags that are applied)

Tagging EBS Snapshots

* Tagged on create - atomic operation, tag failure means no snapshot
* Resources are properly tracked, monitored and enforced
* No need for building scripts

Tagging EBS Snapshots: Controlling Access

* AWS provides resource-level permissions to control access to EBS snapshots through IAM policies
* Tag support for API action:
    * CreateSnapshot
    * DeleteSnapshot
    * ModifySnapshotAttribute
* Use Cases
    * Require use of specific tags
    * Specify which users can take snapshots for a set of volumes
    * Restrict who can delete snapshots