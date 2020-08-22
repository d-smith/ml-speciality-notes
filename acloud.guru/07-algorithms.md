# Algorithms

Still in the train the model part of the machine learning cycle.

## Algorithm Concepts

Definition: unambiguous specification of how to solve a class of problems.

* An algorithm is a set of steps to follow to solve a specific problem intended to be repeatable with the same outcome.
* Contrast with heuristic, which is a mental short-cut or rule of thumb that provides guidance on doing a task but does not guarantee a consistent outcome.

Algorithms in Machine Learning

* Drive out bias by avoiding hueristics
* Still need to look out for bias - data we select for training or testing, or exclude an important chunk of sample data
* Feedback loop can introduce bias - might assume we'll see a certain set of results

SageMaker

* Use built in algorithms
* Purchase from AWS marketplace
* Build your own via docker image

## Regression

Linear Learner Algorithm

* Linear models are supervised learning algorithms for regression, binary classification or multiclass classification problems. You give the model labels (x,y) with x being a high dimensional vector and y as a numeric label. The algoritm learns a linear function, or, for classification problems, a linear threshold function, and maps a cector x to an approximation of  label y.
* To use this algorithm you need a number or list of numbers which yields some other number - the answer you are after. You can use it to predict a specific value or a threshold for grouping purposes.

Adjust to Minimize Error

* Algorithm wants the equation to be as good of a fit as possible, meaning the sum of all the distances from the training data point to the fitted line is a small as possible.
* Stochastic Gradient Descent is used to minimize error.
    * Local and global minimums

Linear Learner for Classification

* Map text values a number representation
* Convert data to vectors

Linear Learner Characteristics

* Very flexible. Linear learner can be used to explore differnt training objectives and chhose the best one. Well suited for discrete or continuous inferences.
* Built-in Tuning. Linear learner algorithm has an internal mechanism for tuning hyperparameters separate from the automatic model tuning feature.
* Good first choice. If your data and objective meets the requirement, linear learner is a good first choice to try for your model.

Usage

* Predict quantitative value based on given numeric input
    * Example: based on the last 5 year ROI from marketing spend, proedice this years ROI
* Discrete binary classification problem
    * Example: based on past customer response, should I email this particular customer? Yes or no..
* Discrete multiclass classification problems
    * Example: based on past customer response, how shouldI reac hthis customer? Email, direct mail, phone call?

Sparse Data

* Linear learner works well with a large amount of contiguous data
* Dealing with sparse data: factorization machines

Factorization Machines Algorithm

* General purpose supervised learning algorithm for both binary classification and regression. Captures interaction between features with high dimensional sparse datasets.
* To use this algorithm you need a number or list of numbers which yields some other number - the number you are after. You can use it to predict a specific value or a threshold for placing into one of two groups. It is a good choice when you have holes in your data.
* Use:
    * When you have high dimensional spare data sets
        * Example: click stream data on which ads on a webpage tend to be clicked given known information about the person viewing the page.
    * Recommendations
        * Example: what sort of movies should we recommend to a person who has watched and rated some other movies

Things to know about factorization machines

* Considers only pairwise features - SageMaker's implementation will only analyze relationships of two pairs of features at a time.
* CSV is not supported. File and pipe mode training is supported using recordio-protobuf format with Float32 tensors.
* Doesn't work for multi-class problems. Binary classification and linear regression modes only.
* Needs LOTS of data. Recommended dimension of the input feature space is between 10,000 and 10,000,000.
* AWS recommends CPUs with factorization machines for the most efficient experience.
* Don't perform well on dense data.

Example: Movie recommendations

* Think of a 1/0 per movie title - lots of movie titles means sparse populaton of review per user

### Notes from SageMaker Linear Learner

Docs are [here](https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html)

* Input: labeled examples (x,y) - x is a high-dimensional vector, y is the label
    * For binary classification, y is 0 or 1
    * For multiclass classification, y is 0 to num-classes - 1
    * For regression, y is a real number

* The best models optimizes either of the following:
    * Continuous objectives, such as mean square error, cross entropy loss, absolute error
    * Discrete objectives suited for classification, such as F1 measure, precision, recall, or accuracy

Linear Learner I/O Interface

* Supports 3 channels: train, validation (opt), test (opt)
* If validation data supplied, S3DataDistribution should be FullyReplicated
* Training - recordIO-wrapped protobuf and CSV formats supported
* Inference - application/json, application/x-recordio-protobuf, and text/csv formats
    * Format of interence response depends on the model
        * Regression is the prediction
        * Binary classification - oredicted label and score indicating how strongly the algorithm believes the label should be 1
        * Multiclass = predicted classs as number from 0 -- num-classes -1 plus score is an array of floating point numbers one per class.

EC2 Instance Recommendations

* Can use single or multi machine CPU and GPU instances
* No evidence multi-GPU computers are faster than single GPU computers

How it Works

* Step 1: preprocess
    * Shuffle data before training
    * Normalization (feature scaling) is important
        * Option available to let linear learner do it for you: normalize_data and normalize_label hyperparameters
* Step 2: Train
    * Uses distributed stochastic gradient descent (SGD) implementation
    * Can choose your optimization algorithm - Adam, AdaGrad, SGD, or others
    * Hyperparameters include momentum, learning rate, learning rate schedule
    * During training, simultaneously optimize multiple models, each with slightly different objectives - vary L1 and L2 regularization, etc.
* Step 3: Validate and set the threshold
    * Models evaluated against validation set to select most optimal model
        * For regression, pick model that achieves best loss
        * For classification, sample of the validation set is used to calibrate classification threshold. Select model with best perf based on selected evaluation criteria - F1 measure, accuracy, cross-entropy loss
* Step 4: Deploy trainined linear model

### Notes from SageMaker Factorization Machines Algorithm

Docs are [here](https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines.html)

* General purpose supervised learning algorithm for classification and regression that's an extension of a linear model designed to capture interactions between features within high diminsional sparse data sets.
    * Example usage: click prediction and item recommendation

I/O Interface

* Can be run in either binary classiciation mode or regression mode
* Can use train and test channels
* Scoring
    * Regression - root mean squared error (RMSE)
    * Classification - scored using Binary Cross Entropy (log loss), Accuracy (at theshold = 0.5), F1 score (at threshold = 0.5)
* Training - recordIO-protobuf format with Float32 tensors
* Inference - application/json and x-recordio-protobug formats
    * Output
        * binary classification - score and label (0 or 1). Score is a number that indicates how strongly the algorithm believes the label is 1. Score computed first, if >= 0.5 then label set to 1.
        * regression - just the score is returned - it is the predicted value

EC2 recommendations

* In general run training and inference using CPUs
* May be some benefits to one or more GPUs training on dense datasets

How it Works

* See [here](https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines-howitworks.html)
* Three terms in the equation are a global bias term, the linear terms model, and the factorization terms that model the pairwise interactions between the variables

## Clustering

Group objects based on similarities.

* Unsupervized algorithms that aim to group things such that they are with other things more similar than different.

K-Means

* Unsupervised algorithm that attempts to find discrete groupings within data, where members of a group are as similar to one another and as different as possible from members of other groups. The Euclidean distance between these points represents the similarity of the corresponding observations.

* K-Means will take in a list of things with attributes. You specify which attributes indicate similarity and the algorithm will attempt to group them together such that they are with other similar things. Similarity is calculated based on the distance between identifying attributes.

SageMaker K-Means

* Expects tabular data. Rows represent the observations tha you want to cluster and the columns represent attributes of the observations
* You must know enough about the data set to propose attributes that will define similarity. If you have no idea, there are ways around this too.
* SageMaker uses a modified k-means algorithm. AWS uses a modified version of the web-scale K-menas algorithm, which it claims to be more accurate.
* CPU instances recommended. GPU instances can be used but SageMaker's K-means can only use one GPU.
* Training is still a thing. You want to make sure you model is still accurate and using the best identifying attributes. Your data just doesn't have labels.
* You define number of features and clusters. You must define the number of features for the algorithm to analyze and the number of clusters you want.

Examples:

* PCM
* Handwritten digit recognition (MNIST data set)

### K-Means: SageMaker Documentation

* Expects tabular data
    * rows represent observations to cluster
    * The n attributes in each row represents a point in n-diminsional space
        * Euclidian distance between the points represents the similarity of the coressponding observations
* I/O interface
    * data provided in the train channel (recommended S3DataDistributionType=SHaredByS3Key), with optional test channel (S3DataDistributionType=FullyReplicated)
    * recordIO-wrapped-protobuf and CSV supported for training
    * Can use File mode or Pipe mode to train models on data that is formatted as recordIO-wrapped-protobuf or CSV
    * For inference, text/csv, application/json, and application/x-recordio-protobuf are supported. k-means returns a closest_cluster label and the distance_to_cluster for each observation.
* EC2 recommendation
    * training - CPU instances
    * Can train on GPU but limit use to p*.xlarge instances because only a single GPU per instance is used

How it works

* In SageMakers implementation you specify k (number of clusters) as well as a number of extra cluster centers (K=k*x), with the algorithm ultimately reducing to k clusters.
    * Step 1: determine the initial cluster centers choosen from the observations in a small, randomly sampled batch. Choose one of the following strategies:
        * random approach: randomly choose K observations in the input as cluster centers
        * k-means++ : pick observation at random, and choose the point in space assocaited with the observation as center 1. For center 2, pick an observetion at randon that is far awat from cluster center 1 by determining the distance to center 1 assigning a propability that is proportional to the square of the distance. Continue this method for the remaining center selection until you have K clusters.
    * Step 2: Iterate over the training data set and calculate cluster centers.
    * Step 3: reduce from K to k

Metrics

* test:msd - mean squared distances between each record in the test set and the closest center of the model
* test:ssd - sum of squared distances between each record in the test set and the closest center of the model

Tunable K-Means Hyperparameters

* epochs
* extra_center_factpr
* init_method (kmeans++, random)
* mini_batch_size



## Classification

K-Nearest Nieghbor

* An index-based, non-parametric method for classification or regression. For classification, the algorithm queries the k points that are closest to the sample point and returns the most frequently used label as the predicted label. For regression, the algoriym queries the k closest points to the sample point and returns the average of the feature values as predicted value.
* Predicts the value or classification based on that which you decide are closest. It can be used to classify or predict a value (average value of nearest neighbors)

Some details

* You choose the number of neighbors. You include a value for k, or in other words, the number of closest neighbors to use for classifying.
* KNN is lazy. Does not use training data points to generalize but rather use them to figure out who's nearby.
* The training data stays in memory. KNN doesn't learn but rather use the training dataset to decide on similar samples.

Use cases:

* Credit ratings - group people together for credit rick based on attributes they share with others of known credit usage
* Product recommendation - based on what someone likes, recommend similar items they might also like.

Beware

* Prone to biasing
    * Redlining - the practive of literally drawing lines around neighborhoods and classifying those as best, still desireable, definitely declining, hazardour. Public and private entities would then use those redline maps to deny home loans, insurance, and other services for those less desirable areas, regardless of the qualifications of the applicant.

### K-Means Nearest Neighbor - SageMaker Notes

SageMaker k-NN algorithm

* For classification, queries the k points closest to the sample and returns the most frequently used label.
* For regression, queries the k closest points to the sample point and returns the average of their feature values as the predicted values.

Training

* 3 steps: sampling, dimension reduction, and index building.
    * Sampling: reduces the size of the dataset to fit in memory
    * Diminsion reduction: reduce the feature diminsions of the data to reduce footprint of k-NN model in memory and inference latency
        * random projection
        * fast Johnson-Lindenstrauss transform
    * Index building: enable efficient lookups of distances between points who values/class is to be determined and the k nearest points used for inference

I/O Interface

* Use train channel for data to sample and construct into k-NN index
* Use a test channel to emit scores in log files
    * Scores listed as one line per mini-batch: accuracy for classifier, mean-squared error for refressor for score
* Training inputs: test/csv and application/x-recordio-protobuf data formats. For input text/csv, first label_size columns are interpreted as the label vector. Use either File node or Pipe mode for training.
* Inference inputs: application/json, application/x-recordio-protobuf, test/csv. Text/csv format accepts a label_size and encoding parameter, defaults are 0 and UTF-8
* Inference outputs: application/json and application/x-recordio-protobuf
    * For batch transform supoprts application/jsonlines for input and output

EC2 recommendations

* Can train on CPU or GPU
* Inference requests for CPUs generally have lower latency (avoids task on CPU-to-GPU communications). GPUs generally have higher throughput for batches.

Training metrics

* test:accuracy for classifier, test:mse for regressor

TUnable k-NN hyperparameters

* k, sample_size

## Image Analysis

* Can return a label (classiciation) and a confidence measure
* Can select a confidence threshold

Amazon Rekognition - amazon's image service

Image Analysis Algoritms

* Image classification - determine the classification of an image. It uses a convolutional neural network (ResNet) that can be trained from scratch or make use of transfer learning. Supervised learning.
    * Can use ImageNet as a resource when training
* Object detection - detects specific objects in an image and assigns a classification with a confidence score
* Semantic Segmentation - low level analysis of individual pixels and identifies shapes within an image (think edge detection)
    * Accepts PNG file input
    * Only supports GPU instances for training
    * Can deploy on either CPU or GPU instances. After training is done, model artifacts are output to S3. The model can be deployed as an endpoint with either CPU or GPU instances.
    * Can do transfer learning for example using the cityscapes data set for self driving models

* Image Analysis Use Cases
    * Image Metadata Extraction - extract scene metadata from an image provided and store it in a machine-readable format.
    * Computer Vision Systems - recognize orientation of a part on an assembly line and, if required, issue a command to a robotic arn to re-orient the part

## Anomaly Detection

  What's different? What's the same?

### Algorithm: Random Cut Forest

 * Detects anomalous data points within a set, like spikes in time series data, breaks in periodicity or unclassifiable points. Useful way to find outliers when it's not feasible to plot graphically. RCF is designed to wok with n-dimensional input.
 * Find occurences in the data that are significantly beyond normal (usually more than 3 standard deviations) that could mess up your model training.   

Random Cut Forest

* Gives an Anomaly Score to Data Points. Low scores indicate that a data point is considered normal while high scores indicate the presence of an anomaly.
* Scales well. RCF scales very well with respect to number of features, data set size, and number of instances.
* Does not benefit from GPU. AWS recommends using normal compute instances.

Use Cases:

* Quality control
    * Example: analyze an audio test pattern played by a high-end speaker system for any unusual frequencies.
* Fraud Detection
    * Example: if a financial transaction occurs for an unusual amount, unusual time or from an unusual place, flag the transaction for a closer look.

### Algorithm: IP Insights

IP Insights

* Learns usage patterns for IPv4 addresses by capturing associations between IPv4 addresses and various entities such as user IDs or account numbers.
* Can potentially flag odd online behaviour that might require closer review.

* Ingests Entity/IP address pairs. Histic data can be used to learn baseline patterns.
* Returns inferences via a score. When queried, the model will return a score that indidates how anomalous the entity/IP combination is, based on the baseline.
* Uses a neural network. Uses a NN to learn latent vector representation for entities and IP addresses.
* GPUs recommended for training. Generally GPUs are recommended but if the dataset is large, distributed CPU instances might be more cost effective.
* CPUs recommended for inference. Does not require costly GPU instances.

Use Cases

* Tiered Authentication Models.
    * Example: if a user tries to log into a website from an anomalous IP address, you might dynamically trigger an additional two-factor authentication routine.
* Fraud Detection:
    * Example: on a banking website, only permit certain activities in the IP address is unusual for a given user login.


## Text Analysis

### Latent Dirichlet Allocation (LDA)

LDA

* LDA algorithm is an  unsupervised learning algorithm that attempts to describe a set of observations as a mixture of distinct categories. LDA is most commonly used to disover a user-specified number of topics shared by documents within a text corpus. Here each observation is a document, the features are the presence (or occurrence count) of each word, and the categories are the topics.
* Used to figure out how similar documents are based on the frequency of similar words.

Use Cases

* Article Recommendation
    * Example: recommended articles on similar topics which you might have read or rated in the past
* Musical Influence Modeling
    * Example: Explore which musical artists over time were truely innovative and those who were influenced by those innovators

### Neural Topic Model (NTM)

Nueral Topic Model

* Unsupervised learning algorithm that is used to organize a corpus of documents into topics that contain word groupings based on their statistical distribution. Topic modeling can be used to classify or summarize documents based on teh tioucs detected or to retrieve information or recommend content based on topic similarities.
* Similar uses and function to LDA in that both NTM and LDA can perform topic modeling. However, NTW uses a difference algorithm which might yield different results than LDA.

### Sequence to Sequence

seq2seq

* Supervised learning algorithm where the input is a sequence of tokens (for example text, audio) and the output generated is another sequence of tokens.
* Think a language translation engine that can take in some text and predict what that text might be in another language. We must supply training data and vocabulary.

1. Steps consist of embedding, encoding, and decoding. Using a nueral network nodel (RNN and CNN), the algorithm uses layers for embedding, encoding, and decoding into targets.
2. Commonly initialize with pre-trained work libraries. A standard practive is initializing the embedding layer with a pre-trained word vector like FastText or Glove or to initialize it randomly and learn the parameters during training.
3. Only GPU instances are supported. SageMaker seq2seq is only supported on GPU instance types and is only set up to train on a single machine. But it does offer support for multiple GPUs on an instance.

Use cases:

* Language translations
    * Example: using a vocabulary, predict the translation of a sentence into another langauge.
* Speech to text
    * Given an audio vocabulary, predict the textual representation of spoken words

### BlazingText

BlazingText

* Highly optimized implementation of the Word2Vec and text classiciation algorithms. The Word2Vec algorithm is useful for many downstream natural language processing (NLP) tasks, such as sentiment analysis, named entity recognition, machine translation, etc.
* Really optimized way to determine contextual semantic relationships between words in a body text.


Can run in multiple modes

| Mode | Word2Vec (Unsupervised) | Text Classification (Supervised) |
| -- | -- | -- |
| Single CPU Instance | continuous bag of words, skip-gram, batch skip-gram | supervised |
| Single GPU instance (1 or more GPUs) | Continous bag of words, skip-gram | supervised with 1 GPU |
| Multiple CPU instances | batch skip-gram | None |

1. Expects single pre-processed text file. Each line in the file should contain a single sentence. If you need to traing on multipe text files, concatenate them into one file and upload the file in the appropriate channel.
2. Highly scalable. Improves on traditional Word2Vec algirithm by supporting scale-out for multiple CPU instances. FastText text classifier can leverage GPU accelaration.
3. Around 20x faster than FastText. Supports pre-trained FastText model but can also perform training about 20x faster than FastText.

Use cases:

* Sentiment Analysis
    * Example: evaluate customer comments in social media posts to evaluate whether they have a positive or negative sentiment
* Document classification
    * Example: review a large collection of documents and detect whether the document should be classified as containing sensitive data like personal information or trade secrets


## Object2Vec

Object2Vec

* General purposed neural embedding algorithm that can learn low-dimensional dense embeddings of high-dimensional objects while preserving the semantics of the relationship between the pairs in the original embedding space.
* A way to map out things in a d-dimensional space to figure out how similar they might be to one another.

1. Expects things in pairs. Looking for pairs of item and whether they are positive or negative from a relationship standpoint. Accepts categorical label or rating/score-based labels.
2. Feature engineering. Embedding can be used for downstream supervised tasks like classification or regression.
3. Traiing data is required. Officially, Object2Vec requires labeled data for training, but there are ways to generate the relationship labels from natural clustering.

Use cases:

* Movie rating prediction
    * Example: predict the rating a person is likely to give a movie based on similarity to other's movie ratings.
* Document classification
    * ExampleL determine which genre a book is based on its similarity to known genres (history, thriller, biography)
