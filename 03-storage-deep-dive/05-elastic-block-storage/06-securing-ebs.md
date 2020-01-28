# Securing EBS

Access Control - via IAM

EBS Encryption

* When creating an encrypted volume and attaching it to certain instances types, data is protect at rest, and in transit between the EC2 instance and the encrypted volume, and from the encrypted volume to the encrypted snapshot.
* Supported by all volume types
* Same performance as unencrypted volumes, small additional latency
* Uses a CMK for both the volume and any snapshots
* More flexibility if you create and use your own keys
    * Define key rotation policy
    * Enable cloud trail auditing
    * Control who can use the key
    * Control who can administer the key
* Envelope encryption
    * CMK encrypts the data key

Changing Encryption State

* No direct way to encrypt an unencrypted volume, or remove encryption from a volume or snapshot. 
* You can copy data from an unencrypted volume to an encrypted volume
* You can apply a new encryption status when copying snapshots, e.g. un to e, or e to e with a new key

Encrypted Snapshots

* Snapshots of encrypted volumes are encrypted
* Volumes created from encrypted snapshots are encrypted
* Copying an unencryped snapshot to an encrypted snapshot will always be a full copy.
* Copying someone's encrypted snapshot? Use you own CMK so if you lose access to the original CMK you can still access the data. All copies will be a full copy however.

Sharing

* Unencrypted - public and private
* Encrypted - not publically shared, have to give key access

Cross-Account Copy

1. Share the CMK associated with the snapshot with the target account
2. Share the encrypted EBS snapshot with the target account
3. From the target account located the shared snapshot and make a copy of it
4. Verify the copy
