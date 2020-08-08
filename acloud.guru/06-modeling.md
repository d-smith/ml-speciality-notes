# Modeling

From the machine learning cycle...

* We've fetched, cleaned, and prepared our data
* Now it is time to traing and evaluate models

## Modeling Concepts

Taking a problem or challenge as described by lots of data, adding a machine learning algorithm and through computation, tying to figure out a mathematical forumula that can accurately generatlize about that problem.

Components

* Model - want to produce generalization, e.g. based on my training here's what I think the outcome is given this new situation
* Data
* Algorithm - right algoritm
* Computational Power
* Feedback loop - for model, for suitability of data

Developing a Good Model

* What type of generalization are we seeking?
    * Forecasting a number? Deciding if customer more likely to choose option A or option B? Detect a quality defect in a machined part?
* Do we really need machine learning?
    * Can a simple heuristic work? IF THEN logic? REgression forumla or lokup function?
* How will my ML generalizations be consumed?
    * Do I need to return real-time results or can I process the inferences in batches? Weill consumers be applications via API calls or other systems which will perform additional processing on the data?
* What do we have to work with?
    * What sort of data accurately and fully captures the inputs and outputs of the target generalizations? Do we have enough data?
* How can I tell if the generalization is working?
    * What method can I use to test accuracy and effectiveness? Should my model have higher sensitivity to false positives or false negatives? How about accuracy, recall, and precision?

Types of Models

|  | Supervised Learning | Unsupervised Learning | Reinfocement Learning |
| -- | -- | -- | -- |
| Discrete | Classification | Clustering | Simulation-based optimization |
| Continuous | Regression | Reduction of Dimentionality | Autonomous |

Choosing the Rght Approach

| Problem | Approach | Why |
| -- | -- | -- |
| Detect whether a transaction if fraud | Binary classification | Only two possible outcomes: fraud or not fraud |
| Predict rate of deceleration of a car when brakes are applied | Heurisric approach (no ml needed) | Well known formulas involving speed, inertia and friction to predict this |
| Determine the most efficient path of surface travel for a robotic lunar rover | Simulation based reinforcement learning | Must figure out the optimal path via trial, error and improvement |
| Determine the breed of a dog in a photo | Multi-class classification | Which dog breed is the most associated with the picture among many breeds |

Cascading Algorithms

* Sometimes we have to stack algorithms
* Example: what is the estimated basket size of shoppers who respond to our email promotion?
    * Remove outliers - random cut forest
    * Identify relevant attributes - PCA
    * Cluster into groups - KMeans
    * Predict basket size - Linear learner

Confusion Matrix

| | Actual True | Actual False |
| -- | -- | -- |
| **Predicted True** | Predicted correctly | False Positive |
| **Predited False** | False negative | Predicted correctly | 

Problem: Is a financial transaction fradulant?

| | Fraud | Not Fraud|
| -- | -- | -- |
| **Predicted Fraud** | Happy bank. Happy Customer. No money lost.| Happy bank. Angry customer. No money lost.|
| **Predited Not Fraud** | Angry bank. Angry customer. **Money lost.**| Happy bank. Happy customer.  No money lost. |

Here is you are a bank you want to avoid where you predicted no fraud but there is fraude. Bank is ok with more false positive than false negatives as that reduces their exposure to fraud. We'll look closely at *recall*.

Example: is email spam?

| | Spam | Not Spam|
| -- | -- | -- |
| **Predicted Spam** | Spam blocked| Legitimate emails blocked|
| **Predited Not Spam** | Spam gets through | Legitimate emails get through |

Set the evaluation approach to error on the side of caution to ensure legitimate emails are not blocked. Watch *precision* of the model closely.





