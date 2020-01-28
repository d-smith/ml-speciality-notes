# Managing EBS Volumes

Volume Actions:

* Create - create in a region
* Attach - attach volumes that have status 'available' to ec2 instances in the same region
* Create Snapshot
* Detach
* Delete

Volume States:

* Creating
* Deleting
* Available
* In-Use

Commands:

```console
ec2 create-volume
ec2 attach volume
ec2 create-snapshot

# Must terminate instance if root volume, otherwise unmount, then
ec2 detach-volume
ec2 delete-volume # volume must be in available state
```

Once attached, format it and use it, create a directory for mounting the new volume, then mount it.

