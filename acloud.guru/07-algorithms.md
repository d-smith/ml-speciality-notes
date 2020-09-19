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

* Linear models are supervised learning algorithms for regression, binary classification or multiclass classification problems. You give the model labels (x,y) with x being a high dimensional vector and y as a numeric label. The algoritm learns a linear function, or, for classification problems, a linear threshold function, and maps a vector x to an approximation of  label y.
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
    * Example: based on past customer response, how should I reach this customer? Email, direct mail, phone call?

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

Linear Regression vs Logistic Regression

* Linear regression is used to predict the continuous dependent variable using a given set of independent variables. Logistic regression is used to predict the categorical dependent variable using a given set of independent variables. ... In logistic regression, we predict the values of categorical variables.

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

Other Use Cases:

* Customer grouping
* Document classification (group based on tags, topics, content, term frequency, etc)
* Crime localities
*  Call record detail analysis

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

## Image Analysis Algoritms

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

### Sage Maker Notes - Image Classification

Details [here](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html)

I/O Interface

* File mode training: recordIO, image
* Pipe mode training: recordIO
* Inference: image/png, image/jpeg, application/x-image

Training

* Train with recordIO - specify recordIO for both train and validation channels, with one file for train and one for validation.
* Train with image format - specify train, validation, train_lst, and validate_lst channels 
    * Training and validation images must be separated, e.g. stored in different folders
    * lst files - tab separated filewith three columns that contains a list of image file. First column is image index, second is class label index for the image, third is the relative path of the index file
* Train with augmented manifest image format - can train in pipe mode using image files without creating recordIO files
* Incremental training - seed the training of a new model with artifacts from a model you trained previously.
* Two training modes - full training and transfer learning. Full training is initialized with random weights, transfer is initialized with pre-trained weights and just the top fully connected layer is initialized with random weights.

Inference

* SageMaker resizes image automatically
* Output is the probability values for all classes encoded in JSON format

EC2 Recommendations

* GPU for training. Use instances with more memory for training with large batch sizes.
* Can use multi-GPU and multi-machines for distributed training
* Both CPU and GPU instances can be used for inference


### Object Detection Algorithm

The Amazon SageMaker Object Detection algorithm detects and classifies objects in images using a single deep neural network. It is a supervised learning algorithm that takes images as input and identifies all instances of objects within the image scene. The object is categorized into one of the classes in a specified collection with a confidence score that it belongs to the class. Its location and scale in the image are indicated by a rectangular bounding box. It uses the Single Shot multibox Detector (SSD) framework and supports two base networks: VGG and ResNet. The network can be trained from scratch, or trained with models that have been pre-trained on the ImageNet dataset.

### Semantic Segmentation

> The Amazon SageMaker semantic segmentation algorithm provides a fine-grained, pixel-level approach to developing computer vision applications. It tags every pixel in an image with a class label from a predefined set of classes. Tagging is fundamental for understanding scenes, which is critical to an increasing number of computer vision applications, such as self-driving vehicles, medical imaging diagnostics, and robot sensing.
> 
> For comparison, the Amazon SageMaker Image Classification Algorithm is a supervised learning algorithm that analyzes only whole images, classifying them into one of multiple output categories. The Object Detection Algorithm is a supervised learning algorithm that detects and classifies all instances of an object in an image. It indicates the location and scale of each object in the image with a rectangular bounding box.
> 
> Because the semantic segmentation algorithm classifies every pixel in an image, it also provides information about the shapes of the objects contained in the image. The segmentation output is represented as an RGB or grayscale image, called a segmentation mask. A segmentation mask is an RGB (or grayscale) image with the same shape as the input image.

## Anomaly Detection

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

#### SageMaker Random Cut Forest

From [here](https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html)

> Amazon SageMaker Random Cut Forest (RCF) is an unsupervised algorithm for detecting anomalous data points within a data set. These are observations which diverge from otherwise well-structured or .patterned data. Anomalies can manifest as unexpected spikes in time series data, breaks in periodicity, or unclassifiable data points. They are easy to describe in that, when viewed in a plot, they are often easily distinguishable from the "regular" data. Including these anomalies in a data set can drastically increase the complexity of a machine learning task since the "regular" data can often be described with a simple model.
>
> With each data point, RCF associates an anomaly score. Low score values indicate that the data point is considered "normal." High values indicate the presence of an anomaly in the data. The definitions of "low" and "high" depend on the application but common practice suggests that scores beyond three standard deviations from the mean score are considered anomalous.
>
> While there are many applications of anomaly detection algorithms to one-dimensional time series data such as traffic volume analysis or sound volume spike detection, RCF is designed to work with arbitrary-dimensional input. Amazon SageMaker RCF scales well with respect to number of features, data set size, and number of instances.

I/O

* Supports train and test channels
* Test channel (optional) computes accuracy, precision, recall, and F1-score metrics on labeled data
* Formats
    * recordIO or CSV for train and test data
    * For CSV, text/csv;label_size=1 where the first column of each row represents the anomaly label: "1" for an anomalous data point and "0" for a normal data point
    * Inference: application/x-recordio-protobuf, text/csv, application/json input. Output is recordio-protobuf or json

EC2 Recommendations

* CPU instance types - algorithm implementation does not use GPUs

Tunable Hyperparameters

* num_samples_per_tree, num_trees


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

#### SageMaker Notes 

From [here](https://docs.aws.amazon.com/sagemaker/latest/dg/ip-insights.html)

> Amazon SageMaker IP Insights is an unsupervised learning algorithm that learns the usage patterns for IPv4 addresses. It is designed to capture associations between IPv4 addresses and various entities, such as user IDs or account numbers. You can use it to identify a user attempting to log into a web service from an anomalous IP address, for example. Or you can use it to identify an account that is attempting to create computing resources from an unusual IP address. Trained IP Insight models can be hosted at an endpoint for making real-time predictions or used for processing batch transforms.
>
> Amazon SageMaker IP insights ingests historical data as (entity, IPv4 Address) pairs and learns the IP usage patterns of each entity. When queried with an (entity, IPv4 Address) event, an Amazon SageMaker IP Insights model returns a score that infers how anomalous the pattern of the event is. For example, when a user attempts to log in from an IP address, if the IP Insights score is high enough, a web login server might decide to trigger a multi-factor authentication system. In more advanced solutions, you can feed the IP Insights score into another machine learning model. For example, you can combine the IP Insight score with other features to rank the findings of another security system, such as those from Amazon GuardDuty.
>
> The Amazon SageMaker IP Insights algorithm can also learn vector representations of IP addresses, known as embeddings. You can use vector-encoded embeddings as features in downstream machine learning tasks that use the information observed in the IP addresses. For example, you can use them in tasks such as measuring similarities between IP addresses in clustering and visualization tasks.

I/O

* The Amazon SageMaker IP Insights algorithm supports training and validation data channels. It uses the optional validation channel to compute an area-under-curve (AUC) score on a predefined negative sampling strategy. The AUC metric validates how well the model discriminates between positive and negative samples.
* Training and validation data content types need to be in text/csv format.
* The first column of the CSV data is an opaque string that provides a unique identifier for the entity. The second column is an IPv4 address in decimal-dot notation. IP Insights currently supports only File mode. 
* For inference, IP Insights supports text/csv, application/json, and application/jsonlines data content types.
* IP Insights inference returns output formatted as either application/json or application/jsonlines. Each record in the output data contains the corresponding dot_product (or compatibility score) for each input data point.

EC2 Instance Recommendations

* Can train on both GPU and CPU instances. In general GPU recommended for training - in some cases with large datasets distribute training with CPUs might reduce training costs
* CPU recommended for inference

How it Works

> Amazon SageMaker IP Insights is an unsupervised algorithm that consumes observed data in the form of (entity, IPv4 address) pairs that associates entities with IP addresses. IP Insights determines how likely it is that an entity would use a particular IP address by learning latent vector representations for both entities and IP addresses. The distance between these two representations can then serve as the proxy for how likely this association is.

## Text Analysis

### Latent Dirichlet Allocation (LDA)

LDA

* LDA algorithm is an  unsupervised learning algorithm that attempts to describe a set of observations as a mixture of distinct categories. LDA is most commonly used to disover a user-specified number of topics shared by documents within a text corpus. Here each observation is a document, the features are the presence (or occurrence count) of each word, and the categories are the topics.
* Used to figure out how similar documents are based on the frequency of similar words.

From the SageMaker docs:

> Amazon SageMaker LDA is an unsupervised learning algorithm that attempts to describe a set of observations as a mixture of different categories. These categories are themselves a probability distribution over the features. LDA is a generative probability model, which means it attempts to provide a model for the distribution of outputs and inputs based on latent variables. This is opposed to discriminative models, which attempt to learn how inputs map to outputs.
> 
> You can use LDA for a variety of tasks, from clustering customers based on product purchases to automatic harmonic analysis in music. However, it is most commonly associated with topic modeling in text corpuses. Observations are referred to as documents. The feature set is referred to as vocabulary. A feature is referred to as a word. And the resulting categories are referred to as topics.



Use Cases

* Article Recommendation
    * Example: recommended articles on similar topics which you might have read or rated in the past
* Musical Influence Modeling
    * Example: Explore which musical artists over time were truely innovative and those who were influenced by those innovators

### Neural Topic Model (NTM)

Nueral Topic Model

* Unsupervised learning algorithm that is used to organize a corpus of documents into topics that contain word groupings based on their statistical distribution. Topic modeling can be used to classify or summarize documents based on the topics detected or to retrieve information or recommend content based on topic similarities.
* Similar uses and function to LDA in that both NTM and LDA can perform topic modeling. However, NTM uses a difference algorithm which might yield different results than LDA.

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

* The Amazon SageMaker Object2Vec algorithm is a general-purpose neural embedding algorithm that is highly customizable. It can learn low-dimensional dense embeddings of high-dimensional objects. The embeddings are learned in a way that preserves the semantics of the relationship between pairs of objects in the original space in the embedding space. You can use the learned embeddings to efficiently compute nearest neighbors of objects and to visualize natural clusters of related objects in low-dimensional space, for example. You can also use the embeddings as features of the corresponding objects in downstream supervised tasks, such as classification or regression.
* A way to map out things in a d-dimensional space to figure out how similar they might be to one another.

1. Expects things in pairs. Looking for pairs of item and whether they are positive or negative from a relationship standpoint. Accepts categorical label or rating/score-based labels.
2. Feature engineering. Embedding can be used for downstream supervised tasks like classification or regression.
3. Training data is required. Officially, Object2Vec requires labeled data for training, but there are ways to generate the relationship labels from natural clustering.

Use cases:

* Movie rating prediction
    * Example: predict the rating a person is likely to give a movie based on similarity to other's movie ratings.
* Document classification
    * ExampleL determine which genre a book is based on its similarity to known genres (history, thriller, biography)

## Reinforcement Learning

The carrot and the stick

* Positive - provide a positive reward thereby motivating the subject to repeat the behavior, presumably for another positive reward.
* Negative - provide a displeasurable experience or response thereby motivating the subject to not repeat the undesired behavior.

Reinforcement Learning

* RL is a machine learning technique that attempts to learn a stategy, called a policy, that optimizes for an agent acting in an environment. Well suited for solving problems where an agent can make autonomous decisions.
* Find the pat to the greatest reward.

Markov Decision Process (MDP)

* Agent - thing doing the activity. wants to max reward with the fewest steps
* Environment - real world or simulation
* Reward
* State - information about the environment and relevant history of past steps
* Action - action that can be performed
* Observation - info available to the agent at each state/step
* Episodes - iterations from start to finsh while agent is accumulating reward
* Policy- decision making part the agent learns to maximize reward

USe Cases

* Autonomous Vehicles
    * Example: a self-driving car model can learn to stay on the road through itertions of trial and error in a sumulation. Once the model is good enough, it can be tested in a real vehicle on a test track.
* Intelligent HVAC Control
    * Example: an RL model can learn patterns and routines of building occupants, impact of sunlight as it transitions across the sky and equipment efficiency to optimize the temperature control for lowest energy consumption.

## Forecasting

### DeepAR

* Forecasting algorithm for scalar time series using recurrent neural networks (RNN). DeepAR outperforms standard autoregressive integrated moving averages (ARIMA) and exponential smoothing (ETS) by training a single model over multiple time series as opposed to individual time series.
* Can predict both point in time values and estimated values over a timeframe by using multiple sets of historic data.

Cold Start Problem

* Little or history to use for building a forecasting model.
* Might want to combine datasets that includes charateristics of the thing with no history.

| Forecast type | Example |
| -- | -- |
| Point forecast | number of sneakers sold in a week is X |
| Probabilistic forecast | Number of sneakers sold in a week is between X and Y with Z% probability |

DeepAR characteristics

* Suport for various time series. Time series can be numbers, counts, or values in an interval (such as temperature readings between 0 and 100)
* More time series is better. Recommended training a model on as may time series as are available. DeepAR really shines with hundreds of related time series.
* Must supply at least 300 observations. DeepAR requires a minimum number of observations across all time series.
* You must supply some hyperparameters. Context length (number of time point model gets to see before making predictions), epochs (number of passed over the training data), prediction length (how many time steps to forecast), and time frequency are all required (granularity of time series).
* Automatic Evaluation of the model. Uses a backtest after training to evaluate the accuracy of the model automatically.

Use Cases

* Forecasting new product performance.
    * Example: incorporate historical data from other products to create a model that can predict performance of a newly released product.
* Predict labor needs for special events.
    * Example: use labor utilization rates at other distribution centers to predict the required leve lof staffing for a brand new distribution center.

## Ensemble Learning

Ensemble learning - using multiple algorithms and models collectively to hopefully improve the model accuracy.

### XGBoost - Extreme gradient boosting

* Open source implementation of the gradient boosted trees algorithm that attempts to accurately predict a target variable that attempts to accurately predict a target variable by combining the estimates of a set of simpler, weaker models.
* A virtual swiss army knife for all sorts of regression, classification (binary and multiclass) and ranking problems, with 2 required and 35 optional hyperparmeters to tune.
* Supervised learning technique

SageMaker implementation

* Accepts CSV and libsvm for training and inference. Uses tabluar data with rows representing observations, one column representing the target variable or label and the remaining columns representing features.
* Only trains on CPU and memory bound. Currently only trains on CPU instances and is memory bound as opposed to compute bound.
* AWS recommends lots of memory. AWS recommends using an instance with enough memory to hold the entire training data for optimal performance.
* Spark integration. Using the SageMaker Spark SDK you can call XGBoost direct from within the Spark environment.

Example: what price to see my house for 

* Overly simplistic: linear relationship based on size
* Want to consider other factors too - location, age, size, number of bedrooms, number of bathrooms, condition, walk-up or lift access, economic climate, lending cliate
* We can create a set of Classification and Regression Trees (CART)
* Different trees with different modifiers based on attributes that are built using
* Hierarchy of trees, sum the impact of all the trees

Use multiple trees to create a more realistic estimation model than any one tree could have provided.

Use cases

* Ranking
    * Example: On an e-commerce website, you can leverage data about search results, clicks,
    and successful purchases, and then use XGBoost to train a model that can return
    relevance scores for searched products.
* Fraud Detection
    * Example: When XGBoost is given a dataset of past transactions and whether or not they
    were fraudulent, it can learn a function that maps input transaction data to the probability that transaction was fraudulent.


## Principal Component Analysis (PCA)

> PCA is an unsupervised machine learning algorithm that attempts to reduce the dimensionality (number of features) within a dataset while still retaining as much information as possible. This is done by finding a new set of features called components, which are composites of the original features that are uncorrelated with one another. They are also constrained so that the first component accounts for the largest possible variability in the data, the second component the second most variability, and so on.
> 
> In Amazon SageMaker, PCA operates in two modes, depending on the scenario:
> 
> regular: For datasets with sparse data and a moderate number of observations and features.
> 
> randomized: For datasets with both a large number of observations and features. This mode uses an approximation algorithm.
>
> PCA uses tabular data.
>
> The rows represent observations you want to embed in a lower dimensional space. The columns represent features that you want to find a reduced approximation for. The algorithm calculates the covariance matrix (or an approximation thereof in a distributed manner), and then performs the singular value decomposition on this summary to produce the principal components.


## Exam Tips

Concepts

* Difference between an algorithm and a heuristic.
* Be aware of how bias can foul our models.
* Understand the difference between a discrete model and a continuous model.
* Understand the difference and characteristics of supervised learning, unsupervised learning and reinforcement learning.
* Know the options SageMaker provides for algorithms (built-in, buy from marketplace and bring-your-own)

Regression

* Understand the types of problems best suited for regression
* Linear Learner algorithm seeks to minimize error via Stochastic Gradient Descent (SGD) with regression problems.
* Linear Learner can also be used with classification problems too.
* Know that Factorization Machines are best suited for sparse datasets and don’t perform well on dense data at all.

Clustering

* Know that clustering algorithms are usually unsupervised.
* Understand that K-Means can perform clustering similar items based on identifying attributes.
* We must define the identifying attributes, number of features and number of clusters.

Classification

* Understand the difference between Classification and Clustering
* K-NN can be used for classification or regression problems based on the nearest K data points.
* K-NN considered lazy algorithm because it does not seek to generalize... rather looks for who’s nearest.

Image Analysis

* Know that image analysis services are usually classifier models which require training.
* Understand the difference between the SageMaker algorithms of Image Classification, Object Detection and Semantic Segmentation.
* Be familiar with the higher-level Amazon Rekogniton service.

Anomoly Detection

* Understand that Random Cut Forest is best used to detect unusual and out-of-the-ordinary events.
* Know that IP Insights is used to detect anomalies between IPv4 addresses and various entities such as user IDs or account numbers.

Text Analysis

* Latent Dirichlet Allocation (LDA) most commonly used to figure out similarity of documents but that it also has uses in other clustering problems.
* Know that a Neural Topic Model (NTM) and LDA can both perform topic modeling but use different algorithms.
* Sequence to Sequence (seq2seq) is often used in language translation and speech to text by using an embedding, encoding and decoding process.
* Understand BlazingText is highly optimized and can be used to cluster as well as classify text

Reinforcement Learning

* Know that RL seeks to find the policy that optimizes an agent acting in an environment.
* Understand the components of the Markov Decision Process (MDP)
* RL is best suited for situations where the agent can or must make autonomous decisions.

Forecasting
* Know why DeepAR is considered to outperform other regression methods of forecasting.
* Understand the Cold Start Problem and how DeepAR can help.
* Understand the difference between Point Forecasts and Probabilistic Forecasts

Ensemble Learning

* Understand Ensemble Learning from a conceptual standpoint.
* XGBoost can be used for regression, classification and ranking problems.
* Know how XGBoost uses decision trees to create an improvement over linear regression.
* XGBoost is “memory-bound” versus “compute-bound”

Know this stuff: [Use SageMaker built in algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)



## Lab Notes

* Create a model artifact, store on s3
* Model validation metrics and error rates
* Some options...
    * XGBoost as a multi-classification problem with researchOutcome as the target attribute. Goal is to minimize the training and validation error.
    Linear Learner as a multi-classification problem with researchOutcome as the attribute. Goal is to maximize the training accuracy (and other metrics)

