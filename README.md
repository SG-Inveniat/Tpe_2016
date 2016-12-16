# Tpe_2016
Artificial Intelligence Tpe 2016


##medicine
Two folders "cancer_1" and "cancer_2" containing similar neural networks aiming at predicting breast cancer (tumour) nature - benign/malignant.


###/resources
Resource folder containing the datasets and their corresponding description as provided by the UCI Machine Learning repository at: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)

The data used was accordingly split into "train_data.txt" and "validation_data.txt".


###/src
Source folder containing the Artificial Neural Network (ANN) code written in python 3.5.

Requirements:

&nbsp;&nbsp;&nbsp;&nbsp;Numpy: http://www.numpy.org

&nbsp;&nbsp;&nbsp;&nbsp;Keras: https://keras.io

&nbsp;&nbsp;&nbsp;&nbsp;TensorFlow: https://www.tensorflow.org

&nbsp;&nbsp;&nbsp;&nbsp;Theano: http://deeplearning.net/software/theano/


###accuracies
current accuracies after training on the validation sets:

&nbsp;&nbsp;&nbsp;&nbsp;cancer_1: ~94%

&nbsp;&nbsp;&nbsp;&nbsp;cancer_2: 92-96%

Sample accuracies and test runs can be obtained in the "sample_results" folder under each of the cancer_ folders. The validation accuracies are displayed alongside their respective losses at the bottom left-hand corner (in %).
