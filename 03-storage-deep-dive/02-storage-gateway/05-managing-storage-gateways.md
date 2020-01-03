# Managing Storage Gateways

## Managing a File Gateway

* Adding file shares
    * Specify bucket name, storage class, select options (guess mime type, give bucket owner full control, enable requestor pays), pick or create role for bucket access
    * Specify allowed clients, set mount options and metadata

* refresh-cache option: gateway caches inventory, refresh cache updates the inventory based on the state of the bucket
* Mounting a file share - done for clients once the share is available
* Upload notifications
    * Storage gateway can send a notification when all written files are uploaded to S3. Can distinguish folder based workloads from object based workloads.
    * Includes files written to the NFS share
    * Use NotifyWhenUploaded API

* Best practice: configure S3 bucket so only one file share can write to it. Create an S3 bucket policy to deny all roles except the role associated with the file share to put or delete objects to the bucket. Single writer - direct s3 or file gateway. Multiple writers ok. Can also export file shares as read only.



## Managing a Volume Gateway

## Managing a Tape Gateway

## Automating Gateway Management