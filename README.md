# Active LearningCompetition

Active learning is a special case of **semi-supervised machine learning** in which a learning algorithm is able to interactively query the user (or some other information source) to obtain the desired outputs at new data points.

The code for **Active Learning** Kaggle **regression** competition. 

The implemented algorithm is a combination of two **Active Learning** algorithms:
* Train regression model on using labeled data (with Cross-Validation). Predict values Y_hat for labeled data and calculate error E for each of the labeled objects. 
Then train labeled dataset, using prediction errors E as labels. Predict E_hat for unlabeled data and choice the object on which the model give the biggest E_hat. These objects are chosen for sampling.
* Train a zoo of algorithms on labeled data using random subspace method. Then, predict target values for unlabeled data. The objects with highest prediction variance are chosen for sampling.

During the learning process algorithms are being weighed in depends on the quality they give to the model and proportion of sampling changes respectively.
