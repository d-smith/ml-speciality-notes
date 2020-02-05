# EBS Volume Encryption

Encryption done server side.

How it works:

1. KMS provisions a unique data key for each volume that requests encryption, never reused.
    * Key always associated with the volume, doesn't matter which instance it is attached to.
2. Volume passes encrypted version of that key to the instance when it is attached to the instance.
3. Instance requests that KMS decrypt the encrypted key.
    * Master key never leaves the KMS system.
4. Decrypted version of the data key is returned to the instance, never stored on disk, kept in memory only.