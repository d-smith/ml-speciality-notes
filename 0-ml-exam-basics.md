# Machine Learning Exam Basics

## The ML Stack

### Overview

application services

* designed for app developers
* solution oriented pre build DL models available via APIs
* image analysis, langauge services, conversational ux

platform services

* designed for data scientists to address common and advanced needs
* fully managed platform for enterprise data service
* reduces heavy lifting in model training and dev

frameworks and interfaces

* designed for data scientists to address advanced/emerging needs
* provides max flexibility using leading ml frameworks
* enables expert ml systems to be developed and deployed

infrastructure

### Adoption Benefits

Make the best use of data scientist's time

* 80% of data scientists time spent perparing and managing data for analysis
* ...leaving only 20% of time used to derive insights and business value
* services like AWS glue and athena can be used to prepare and manage data where start up time is minutes not days/weeks/months

Converting the power of ML into business value

* improves business value by streamlining:
    * model training in the cloud
    * model deployent in the cloud and at the edge
* easy to invoke models in production by calling an API

Embedding ML into the business fabric

* value of ml relies on operationalizing models within business applications and processes
* 50% of predictive models don't get implemented
* improve process, minimize manual intervention, and make better decisions using one-click deployment

### Application Services 

* Amazon polly - text to speech service
    * use cases: text readers for web pages or podcases, public announcements, game characters, e-learning videos, interactive voice responses, contact centers
* Amazon Lex - natural language understanding (NLU)
    * build natural conversational interfaces
    * use cases: chatbot, feature phone bots and call centers
    * suited for intent based conversations
* Amazon Rekognition
    * deep-learning based image and video analysis
        * object, scene, and activity recognition
        * facial recognition and analysis
        * person tracking
        * unsafe content detaction
        * celebrity recognition
        * text in images
* Amazon Transcribe
    * Automatic conversion of speech into accurate, grammatically correct text
        * english and spanish support
        * intelligent punctuation and formatting
        * timestamp generation
        * support for telephony audio
        * recognize multiple speakers
        * custom vocabulary
    * use cases - call centers (recording transciption), subtitles for video on demand, transcribe meetings
    * can integrate with comprehend, and enhance transripts with human in the loop
* Amazon Translate
    * real time translation, batch analysis, automatic language recognition
* Amazon Comprehend
    * NKP NN developed by AWS
    * Discovers entities, key phrases, different languages, and sentiment
    * tag and label for data science
    * common use cases
        * voice of the customer analysis
        * semantic search
        * knowledge management/discovery
    * can use transcribe, translate, and comprehend together
    
### Platform Services

* Amazon SageMaker
    * Developed because ML is still too complicated for everyday developers
        * collect and prepare training data
        * choose and optimize your ML algoritm
        * set up and manage environments for training
        * train and tune model
        * deploy in production
        * scale and manage the production environment
    * SageMaker provides and environment to make ML easier
        * Build: pre-built networks, build in high performance algorithms
        * Train: one-click training, hyperparameter optimization
        * Deploy: one-click deployment, fully managed hosting with auto-scaling

* AWS DeepLens
    * HD video camera with on-board compute optimized for deep learning
    * integrated with sage maker and lambda
    * tutorials, examples, demos, pre-built models
    * unbox to inference in < 10 minutes

### Frameworks and Interfaces

AWS Deep Learning AMIs

* provide tools to develop deep leanring models in the cloud
* are scalable
* support managed auto-scaling cluster of GPU for large-scale training
* Supports MXNet, TensorFlow, Caffe, Caffe2, Keras, Theano, Torch, Microsoft Cognitive Toolkit

Gluon API

* improves speed, flexibility, accessibility of deep learning technology to developers
* supports multiple frameworks
* Provides...
    * Simple, easy-to-understand code
    * Flexible, imperative structure
    * High performance
* Open Neural Network Exchange (ONNX)
    * Developers can chhose the framework that best fits their needs
    * MXNet, PyTorch, Caffe2, Cognitive Toolkit (CNTK)

### Infrastructure

EC2 P3 Instances

* Offer up to 8 NVIDIA V100 GPUs
* Support the 61xlarge size - 128 GB GPU memory, more than 40,000 GPU cores, more than 125 teraflops single precision, > 62 teraflops double precision
* 14x faster than p2

IoT Edge Devices

* Greengrass - value from IoT devices at the edge, make them intelligent 
* response quickly ro local events, operate offline, simplified device programming, reduce cost of iot apps, asws-grade security

## ML for Business Challenges

### ML For Business Leaders

When is machine learning an appropriate tool to solve my problem?

What ML is not...

* A one stop shot to solve every problem
* Can't use to identify causaility
* Needs a lot of data to build viable solutions

What ML does...

* Opens the door to innovation, harnesses the power of collaboration, and adds deep levels of intelligence to diverse applications.

When is ML an option?

* if the problem is persistent
* if the problem challenges progress or growth
* if the solution needs to scale
* if the problem requires personalization

What does a successful ML solution require?

* People - ml scientists, data scientists, software engineers, etc
* Time - collect data, clean it, build solution, test, etc
* Cost

Limitations

* Models only as good as the data they are built on
    * Data cleansing is essential, determines your success
* Interpreting the results produced by the ML model
* Uncertainty based on their statistic nature

Leaders...

* Can identity sources of data
* Invest in data improvement strategies
* Acknowedge the time and effort needed to introduce ml
* Enable the team by asking the right questions
    * what are the made assumptions?
    * what is your learning target? (output variable/hypothesis)
    * what type of ml problem is it? 
    * why did you choose this algorithm?
    * how will you evaluate the model performance?
    * how confident are you that you can generalize the results?
* Attact and retain talent
    * Enable collab by allowing publishing papers, contributing open source software, etc.
    * Enable access to literature, attendance of workshops and conferences, etc.

### Intro to Machine Learning

ML - techniques to learn what output to produce based on input as opposed to a hard coded function. 

Big data and improved economics at scale have enabled to advancement and adoption of ML.

ML use at Amazon:

* Recommendations
* Robotics optimizations in fullfillment centers
* forecasting
* Search optimizations
* Delivery routes
* Alexa

```console
Flywheel: more data -> better predictions -> better recomendations
    -> more satisfied customers -> more sales -> more data
```

ML

* subfield of AI
* prevelance of large data sets and massve computational resources have made it the dominant subfield of AI
* enables computer programs to improve their performance without code changes

AI

* decribes machines capable of completing tasks that previously require humans

Example:

* Reviews - off topic review problem: is the review about the product, or about shipping problems, questions about the product, etc
* Traditional approach - select a blacklist, maybe add some more wores, try it out
    * context lost, interaction between words, etc
* ML approach - feed reviews directly into machine learning alg and build a classifier
    * Humans don't have to build domain word lists, optimize, etc
    * No need to write custom code

How to Define and Scope a Machine Learning Problem

* What is the specific busines problem you are trying to solve?
* What is the current state?
* What are the current pain points?
* What is causing the pain points?
* What is the problem's impact?
* How would the solution be used?
* What is out of scope?
* How do you define success?

Input Gathering

* Do we have sufficient data?
* Are there labeled examples? If not how difficult would it be to create or obtain them?
* What are our features?
* What are going to be the most useful inputs?
* Where is the data?
* What is the data quality?

Outputs

* What business metric is defining success?
* What are the trade offs?
* Are there existing baselines?
* If not, what is the simplist solution?
* Is there any data validation needed to greenlight the project?
* How important is runtime and performance?

With inputs and outputs defined you can then formalize the problem as an ML problem.

When is machine learning a good solution?

* Use ML when easier to learn behavior from data then it is to code the logic in software directly
    * high level tasks people can do effortlessly, like understanding speech
    * combine weak, link pieces of evidence
    * fit ASIN into the appropriate categories
* Use when behavior is adaptive and changes over time
* When the manual approach does not scale 
* Use when there is ample data to learn from
* Use when the problem is formalizable as an ML problem (reduce to well known ML problem)

When is machine learning not a good solution?

* ML is not as simple as traditional software

When is ML the wrong choice?

* No data
* No ground truth labels (can't use in supervised label)
* Need to launch quickly (hard to predict how long it will take)
* No tolerance for mistakes
    * Consider humans in the loop in some circumstances

Machine learning application

Type of machine learning applications

* Supervised learning
* Unsupervised learning
* Reinforcement learning

Supervised Learning

* Data set must have ground truth labels (e.g. label email - spam/not-spam)
* classification
    * examples: fraud detection - fraud/not fraud, face recognitions - face matches credential being used or not, e.g. amazon go stores)
* multiclass classification - multiple choices of output
* regression (continous valies)
    * demand forecasting - for each item sold on amazon, what's the future demand 
    * robot drive units 

Unsupervised Learning

* data does not have labels
* used to detect patterns in data
* clustering - how to partition a data set
    * e.g. customers based on buying habits, use the classifications to better recommend produces
    * topic modeling for e-books

ML business problem: gift wrap eligibility

* not fragile, max dim 24 inches
* some will understate fragility, some items greater than 24 can be gift wrapped
* associates can review rejected items and override the rule (dealing with false negatives)
* false positives can be identified by examining how long it takes an associate to gift wrap an item
* Build a classifie

First - select the example features (the feature vector)

* properties of the item that you feel are relevant to making an accurage prediction
* binary features, categorical, numeric
* need labels for each item

Want: function when given a feature vector can output a target label

Can we just code a function?

How can we learn a classifier how can we learn a classifier?

* decision tree?
* linear classifier?

When is machine learning a good solution?

* Difficult to directly code a solution
    * What is in the example we add manufacturer as a feature? It would be difficult to maintain as manufactureres are added over time...
* Difficult to scale a code-based solution
    * Too many items to consider in the gift wrap solution
* Personalized output
    * Different classifiers for different fullfillment centers
* Functions that change over time
    * Diminsions of the gift wrapping paper

Data, Data, Data

* Neural networks with large numbers of hidden layers can perform better by capturing non-linearities. What makes this possible?
    * Growing processing power
    * Large amounts of data

* Types of Data
    * Design matrix: 2D table structure, columns are features, rows are feature vector, example
    * Text - description, customer reviews, book text, etc. High-diminsonal (lots of words), sparse.
    * Image and video data
    * Sets - grouping of entities that occur together. For example items purchased/viewed together.
    * Sequence data - clickstreams, order of viewing, etc
    * Time series - time assocaited with each events, temporal component important part of model
    * Graph data - for example social networks

* ML Data Scope Questions
    * How much data is sufficient for building sucessful ML models?
    * Data quality - how to deal with data quality issues
    * Data preparation prior to model building

Image Classification: Vocabulary and Example

* Example: identify useless (nearly blank, etc photos)
    * To learn this function from data, we need a data set with labeled training instances
    * Feature engineering - deciding on a set of measurements to make from each input image.
        * Might compute the standard deviation of pixel values - pictures mostly the same color will have a low standard deviation
        * Range of brightness across the image
    * Select the learning algorithm
    * Train the model
    * Apply to each unlabaled image in the catalog

* Which algoritm and what features
    * Can try a variaty of algoritms and features to evaluate the performance of different approaches

Reinforcement Learning: Robot Programming Example

* Imagine a grid to move a robot on - we want to provide a function to help decide where to move on the grid.
    * Some squares have nutricious plants that provide energy to the robot, some have poison that sap energy, some are empty but cost energy to move to.
* Here we learn a mapping of the current state and a desired action, which is known as a policy.
* Data is captured as the agent interacts with the environment, and provides feedback with respect to the action, referred to as a reward.
* With reinforcement learning, the goal is to learn a policy that will maximize the accumulation of a reward in the future.
    * State: encoding of the agent and current (observable) environment
    * Action: {move up, move down, move left, move right, eat plant}
    * Past Data: where the robot has been, what the robot has eaten so far, corresponding rewards received
* Reinforcement learning differs from unsupervised learning in several ways:
    * No presentation of input/output pairs
    * Reward based
    * Agent needs to gather useful experiences about the possible system states, actions, transitions, and rewards to act optimally
    * Evaluation of the system is often concurrent with the learning.

Machine Learning in Action: The Pollexy Project

* The Pollexy Project is a Raspberry Pi and mobile-based special needs verbal assistant that lets caretakers schedule audio task prompts and messages both on a recurring schedule and/or on-demand.

# Machine Learning Terminology and Process

*Training* - how the machine uses data to build its prediction algorithms. The algorithms make up the *model*. The model is then used by the machine to take inputs and make a *prediction* (sometimes called the interence).

In a typical training process, the historical data used to build the model is split into two datasets. Most of the data is used for the training data set, and the rest is used for the test dataset.

## The Process

1. Business Problem
2. ML Problem Framing
3. Develop Your Dataset

### Step 1 - Business Problem

Goal: prediction

Form machine learning problem from the business problem

Questions to ask:

* Do we have all the data we need?
* What algorithm should we use?


### Step 2 -  ML Problem Framing

3 common types of ML algorithms

* Supervised - used where we have labeled historical data we used to train the machine to make predictions of future values
    * Classification - categorize objects into fixed categories (binary, multiclass)
    * Regression - 
* Unsupervised - answer not know ahead of time, let the algorithm quantify the data and give us the result
* Reinforcement - algoritm is rewarded on the choices it makes while learning

To frame our machine learning problem we need to formulate the output of the problem to evalue. Key elements: observations, lablels, features. Features can be numeric, features we derive, etc.

### Step 3 - Develop Your Dataset

Data Collection/Integration

* Can be collected from multiple sources, multiple data sets integrated, etc.

Three types of data:

* Structured - organized and stored in databases in rows and columns. Querying and analysis easy.
* Semistructured - organized in familiar formats but not stored in stables (CSV, JSON, etc)
* Unstructured - data that does not have any structure - app logs, text, video, etc.

### Step 4 - Data Preparation

Data found in the real world is very dirty and noisy - may have been improperly collected and formatted, may be incomplete, irrelevant, or misleading. 

May need to be converted into a format appropriate for the mode.

May need to add column headers, do type conversions, etc.

Deal with missing features and outliers.

Dealing with missing values

* Add indicator column to note rows with missing values
* Throw away rows with missing values
* Use a technique called imputation to fill in the missing values (uses best guess) For example, for missing numerical values use the mean or median value.

Shuffle Training Data

* We don't want the model to 'learn' anything related to the order the data is presented.
* Shuffling results in better model performance for certain algorithms
* Minimized risk of cross validation data under-representing the model data and model data not learning from all types of data.

Test-Validation-Train Split - 20% test, 10% validation, 70% train

Cross Validation

* 20/10/70 split
* leave-one-out: use one data point as our test, run training with the rest
* k-fold: randomly split the data into k folds, and for each fold train the model and record the error

### Step 5 - Data Visualization & Analysis

* Feature - an attribute in your training dataset
* Label - variable you are trying to predict, not a feature

Types of Viz

* Statistics
* Scatter plots
* Histograms - use to find outliers

Skew, feature and class distribution

Numerical - count, mean, max 
Categorical - counts via histograms

Feature-target correlation: scatter plots

### Step 6 - Feature Engineering

* Process of manipulating raw or original data into new useful features is called feature engineering.
* The most critical and time consuming part of model building
* Requires trial and error, domain knowledge, and ingenuity.
* Helps answer 'what am i using to make my decision'?

Converts raw data into a higher representation.

* e.g. preprocess image data to identify edges, other shapes

*Numeric value binning* - introduce non-linerarity into linear models but intelligently breaking up continuous values using binning.

Think age vs salary - increases with age, stabilizes at some point, maybe declines later.

*quadratic features* - derive new non-linear features by combining feature pairs, e.g. combine education and occupation

Other transformations like log, polynomical power, product/ration of feature values. Leaves of decision tree.

*domain-specific features*

Text features - stop-words removal/stemming, lowercasing, removing punctuation, cutting off very-high/low frequencies, term frequency inverse document frequency (tf-idf) normlization

Web pages - multiple fields of text: url, in/out anchor, title, frames, body, presence of certains elements like tables or images, relative style (italics/bold, font-size) & position


## Step 7 Model Training

Train model multiple times using different parameters 

### Parameter Tuning

Loss Function - distance from ground truth

* square: regression, classification
* hinge: classification only, more robust to outliers
* logistic: classification only, better for skewed class distributions

Regularization

* Prevent overfitting by constraining weights to be small

Learning Parameters (decay rate)

* Decaying too agressively - algoritm never reaches optimum
* Decaying too slowly - algoritm bounces aroundm never converges

## Step 8  Model Evaluation

Overfitting and Underfitting 

* Don't fit your training data to obtain max accuracy, want to generalize - look at evaluation accuracy
* Overfitting - models that don't generalize well as they are too specific to the training data
* Underfitting - models that don't generalize well, not using enough features

Bias-Variance Trade Off

Bias: difference between average model predictions and true target values
Variance: variation in model predictions across different training data samples

Evaluation Metrics:

* Regression  
    * Root mean square error (RMSE) - lower is better
    * Mean absolute precent error (MAPE) - lower is better
    * R2 - how much better is the model compared to just picking the best constant? R2 - 1 - (Model mean squared error / variance)

* Classification
    * Confusion matrix

|              | Actual +1      | Actual -1       |
|--------------|----------------|-----------------|
| Predicted +1 | True positive  | False  positive |
| Predicted -1 | False negative | True negative   |

    * ROC curve
    * Precision recall

Precision = TP/(TP + FP) - how correct we are on the ones we predicted as positive
Recall = TP/(TP + FN) - fraction of the negatives that were wrongly predicted

## Business Goal Evaluation

Business goal evaluations

* Evaluate how the model is performing related to business goals
* Make the final decision to deploy or not

Evaluation depends on:

* Accuracy
* Model generalization on unseen/unknown data
* Business success criteria

Augmenting Your Data

If we need more data or better data to prevent overfitting, we an add data augmentation or feature augmentation to our pipeline. These techniques increase the complexity of our data by adding information derived from internal and external data.

Goal is to deploy a model into production. To work successfully, the production data needs to have the same distribution as the training data. Since data distributions can drift over time, the deployment processs is an ongoind process. Monitor production data, trigger retraining if drift found. Or... train the model periodically.