# Data Preparation

## Intro and Concepts

Machine Learning Cycle

* We want to Clean and Prepare Out Data
* First, we must understand our data

Data prep

* the process of transforming a dataset using different techniques to prepare it for model training and testing
* changing our dataset so it is ready for machine learning

Typical Activities

* Categorical Encoding
* Feature Engineering
* Handling Missing Values

Options for Data Prep

* SageMaker and Jupyter Notebooks (adhoc)
* ETL Jobs in AWS Glue (reusable)

## Categorical Encoding

Categorical Encoding

* The process of manipulating categorical variables when ML algorithms expect numerical values as inputs
* Changing category values in our dataset to numbers

When to encode

| Problem | Algorithm | Encoding |
| -- | -- | -- |
| Predicting the price of a home | linear regression | encoding necessary |
| Determine whether given text is about sports or not | Naive Bayes | encoding not necessary |
| Detecting malignancy in radiology images | CNN | encoding necessary |

Pick the algorithm, then figure out if you need to encode your variables.

Examples:

* Color: { green, purple, blue} - multi-categorical, nominal
* Evil: {true, false} - binary categorical, nominal
* Size: { L > M > S } - ordinal (order does matter)
* Note all values are discrete, not continuous

More examples:

* Binary label - can transform 0 and 1 to N and Y
* Ordinal - S,M,L need to make sure numeric mapping is ordered, may have to determine appropriate values
* Nominal values (condo, house, apartment)
    * Encoding variables into integers is a bad idea (algorithm might treat them as ordered)
    * Better to one-hot encode

One-Hot Encoding

* Transform nominal categorical features and creates new binary columns for each observation
* Know your data before you encode
    * One hot encoding is not always a good choice when there are many, many categories
    * Using techniques like grouping by similarity could create fewer overall categories before encoding
    * Mapping rare values to "other" can help reduce overall number of new columns created

Summary

* Encoding needs are ML algorithm specific
* Text into numbers
* No single rule that universally applies for transformatopm
* Many differnt approach, each of which can impact on the outcode of your analysis


## Text Feature Engineering

Feature engineering

* Transforming attributes within our data to make them more useful withing our model for the problen at hand. Feature engineering is often compared to an art.

Text feature engineering: transforming text within our data so machine learning can better analyze it. Splitting text into bite size pieces.

* Uses include natural language processing, speech recognition, text-to-speech
* data like words and phrases, spoken languages, dialogue between two people
* sources like magazines, newspapers, academic papers

Bag of Words

* Tokenizes raw text and create a statistical representation of text
* Break up text by white space into single words

N-Gram

* An extension of bag-of-words which produces groups of words of n size
* Breaks up text by white space into groups of words
* 1-gram same as bag of words
* Think of it as a sliding window (no overlap)
* 1 unigram, bigram 2, trigram 3

Orthogonal Sparse Bigram (OSB)

* Creates groups of words of size n and outputs every pair of words that includes the first word
* Creates groups of words that always include the first word
    * Orthogonal - when two things are independent of each other  
    * Sparse - scattered or thinly distributed
    * Bigram - 2 gram or two words  
* Used a delimiter of some sort

OSB in action - OSB, size 4

* Raw: "he is a jedi and he will save us"
* OSB size 4: {"he_is", "he__a", "he___jedi"}{"is_a", "is__jedi, etc} - note the use of underscores as place holders.

TF-IDF: Term frequency - invese document frequency

* Represents how important a word or wors are to a given set of text by providing appropriate weights to erms that are common and less common in the text.
* Shows us the popularity of a word or words by making common words like "the" or "and" less important
* Term frequency - how frequent does a word appear?
* Document frequency - number of documents in which the which terms appear
* Inverse - makes common works less meaningful

Vectorized tf-idf

* (number of documents, number of unique n-grams)
* See [this resource](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

| Problem | Transformation | Why |
| -- | -- | -- |
| Matching phrases in spam emails | n-gram | easier to compare whole phrases like "click here now" or "you are a winner" |
| Determine the subject matter of multiple PDFs | tf-idf, orthogonal sparse bigram | filter less important words in PDFs. Find common word combinations repeated throughout PDFs |

Standard Transformations

* Remove punctuation: sometimes removing punctuation is a good idea if we do not need them (but leave punc inside words)
* Lowercase transformation: using lowercase transformation can help standardize raw text

Cartesian Product Transformation

* Creates a new feature from the combination of two or more text or categorical values.
* Combining sets of words together
    * Multiply a set of words by another set of words

Feature Engineering Dates

* Feature engineer dates so we can extract more information from them for our models
* Extracted information answers...
    * Was it a weekend or week day?
    * Was the date the end of a quarter?
    * What was the season?
    * Was the date a holiday?
    * Was it during busines  hours?
    * Was the world cup taking place on this date?
* Can include columns that include the extracted info, e.g. is_weekend, day_of_week, etc

