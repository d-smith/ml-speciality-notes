# Encryption in S3

Data at Rest Encryption

* Server side encryption
    * Amazon s3 encrypts when saving, and decrypts it when it's retrieved
    * Options
        * SSE with Amazon S3 managed keys (SSE-S3)
        * SSE with AWS KMS-managed keys (SSE-KMS)
        * SSE with customer provided keys SSE-C

* Client-side encryption - encrypted before upload

SSE-S3

* Each object encrypted with a unique key
* Per object key encrypted with master key (which can be rotated)
* AWS-256

SSE-KMS

* Similar to SSE-S3: KMS used to fully managed keys and encryption
* Per object keys is protected with a CMK
* Uses [envelope encryption](https://docs.aws.amazon.com/kms/latest/developerguide/concepts.html#enveloping)
* Can use your own CMK to give you the ability to rotate, disable, define access, and audit.
* Be aware of KMS API rate limitations

SSE-C

* Customer provided encryption keys
* You provide the key, amazon uses it to encrypt and decrypt using AWS-256
* Key provided both on upload and retrieved
* Key not stored, randomly salted hmac of key used to validate keys; key not stored, so if key lost you cannot decrypt the object

Client-Side Encryption

* Encrypt the object, upload the encrypted object
* Can use MKS managed customer master key, or use you own client-side master key

Default Encryption

* Can enable default encryption option - enable SS3-S3 (amazon s3-managed keys) or AWS-KMS for new objects not encrypted on upload
* You can then remove policies that dealt with lack of encryption

