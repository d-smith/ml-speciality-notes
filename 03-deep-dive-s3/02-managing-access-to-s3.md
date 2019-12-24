# Managing Access to S3

Share responsibility model

![shared responsibility](./shared-resp.jpg)

Access Determination

![access decision](./access-decision.jpg)

IAM Policies

* Assigned to IAM user, groups, and role
* Grant access to AWS resources

Bucket Policies

* Resource based policy assigned to an S3 buckets
* Incorporate user retrictions without using IAM - includes action, principal, and resource


IAM and Bucket Policies

* IAM - what can this user do in AWS. USe when controlling access to AWS services, policies in the IAM environment, use fewer more detailed policies
* Bucket policies - Who can access this bucket? 
    * Policies in the s3 environment.
    * Cross account access to your Amazon s3 resources
    * Size limits on IAM policies

Policy Language Elements

* Resource - elements the statement covers
* Action - resource operations to allow of deny
* Effect - Allow or Deny
* Principal - IAM user, federated user, assumed role user, AWS account, AWS service, or other principal entity
* Condition - expressions for when a policy is in effect. Match against information in the request for access to the resource

Some resources

* [Access policy langauge overview](https://docs.aws.amazon.com/AmazonS3/latest/dev/access-policy-language-overview.html)
* [Bucket policy examples](https://docs.aws.amazon.com/AmazonS3/latest/dev/example-bucket-policies.html)
* [IAM policy examples](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_examples.html)

Policy Elements

* NotPricipal
    * Denies all except the principle defined; do not use with policy statement "Effect":"Allow"
    * When using Deny the order in which AWS evaluates principals makes a difference
    * Does not explicitly grant access when used alone in a policy
    * Need to include the ARN for the AWS account in addition to the ARN of the user, a user cannot have more permissions than its parent account.
* NotAction
    * Matches everything except the list defined
    * Used with statement Effect Allow or Deny
        * Allow: provide access to all except listed
        * Deny: deny access to all except listed - does not explicitly grant access to other actions not listed


Example - deny access to all servcices except s3 unless the user is signed in using MFA

```
Statement: [{
    "Effect":"Deny",
    "NotAction":"s3:*",
    "Resource":"*",
    "Condition":{"BoolIfExists":{
        "aws:MultiFactorAuthPresent":"false"
    }}
}]
```

* NotResource
    * Matches everything except the list of defined resources
    * Use with statement effect - allow or deny
        * Allow: provide access to all except listed
        * Deny: deny access to all except listed

Example - you have a group named hrpayroll. Members of hr payroll should not be allowed to access any s3 resources except the payroll folder in the hr bucket

```
"Statement": {
    "Effect":"Deny",
    "Action":"s3:*",
    "NotResource":[
        "arn:aws:s3:::hrbucket/payroll",
        "arn:aws:s3:::hrbucket/payroll/*"
    ]
}
```

The above provides the restriction, additional policy elements are needed to provide the grant.
