# Authentication and Authorization with AWS IAM

IAM terminology

*User* - a permanent named operator, permanent set of credentials, authentication method
*Group* - collections of users
*Role* - authentication method, an operator, credentials associated with a role are temporary
*Policy Document* - permissions, attaches to a user, group of users, or a role

API call - present a set of credentials. API engine looks at the credentials, validates they are active, approves them. Then, look at the policy documents associated with the user to see if the API call is allowed. 

Explict deny trumps all allows.