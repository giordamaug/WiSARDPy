# WiSARDPy
WiSARD Classifier and Regression in Python

# Classifier
run the example command:

```
python test_classifier.py -i datasets/iris.csv -T species -M WNN -b 8 -z 256 
```

This command executes a train/test split of the dataset (25% of test) and
makes a classification with the WiSARD method with 8 bits and a 256 tic-sized termometer encoding.
The paramter `-T`specifies the name of the targte column in the dataset. 

If you want to apply another classfier (i.e. RandoForest):

```
python test_classifier.py -i datasets/iris.csv -T species -M RF
```


# Regressor
run the example command

```
python test_regression.py -i datasets/boston.csv -T medv -M WNN -b 63 -z 128
```
This command executes a train/test split of the dataset (10% of test) and
makes a regression with the WiSARD method with 63 bits and a 128 tic-sized termometer encoding.
The paramter `-T`specifies the name of the targte column in the dataset. 

If you want to apply another regressor (i.e. RandoForest):

```
python test_classifier.py -i datasets/boston.csv -T medv -M RF
```

