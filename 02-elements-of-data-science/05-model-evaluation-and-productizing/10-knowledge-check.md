# Knowledge Check 5

Q1: Why is it reasonable to report the success of a classifier using only precision and recall, even through both these metrics ignore the TN (true negatives) entry of the confusion matrix.

A1: In classification problems, we usually set the interested responses as positive class and in many cases, the interested class is related to a rare situation (e.g. fradulant transaction) and the regular case is more common. TN cases usually dominate and if you include TN and focus on accuracy, then it may give us very high accuracy, but not much detail about the model performance for the positive class. FOr a balanced data where positive and negative are roughly equal, we can include TN in the analysis.

