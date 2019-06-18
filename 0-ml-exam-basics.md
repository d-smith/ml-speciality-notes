# Machine Learning Exam Basics

## The ML Stack

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

## Adoption Benefits

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

## Application Services 

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
    
## Platform Services

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

## Frameworks and Interfaces

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

## Infrastructure

EC2 P3 Instances

* Offer up to 8 NVIDIA V100 GPUs
* Support the 61xlarge size - 128 GB GPU memory, more than 40,000 GPU cores, more than 125 teraflops single precision, > 62 teraflops double precision
* 14x faster than p2

IoT Edge Devices

* Greengrass - value from IoT devices at the edge, make them intelligent 
* response quickly ro local events, operate offline, simplified device programming, reduce cost of iot apps, asws-grade security
