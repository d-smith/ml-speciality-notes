# Feature Extraction

Maps data into a smaller feature space that captures
the bulk of the information in the data

* a.k .a, data compression

Motivation- Improve computational efficiency

*  Reduce curse of diminsionality
Techniques

* Principle component analysis
* Linear discriminant analysis
* Kernel versions of these for fundamentally non - linear data,

Principle Component Analysis

* Unsupervised linear approach to feature extraction
*  Finds patterns based on corellations between features
*  Constructs principle components : orthoganal axes in directions of maximum variance

scikit-learn : sklearn.decomposition.PCA

Kernel PCA - uses a Kernel function to project a dataset into a higher dimnsional feature space where it is linearly seperable.
