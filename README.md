# WiSARDPy
WiSARD Classifier and Regression in Python

# Classifier
run the example command:

```
python test_classifier.py -i datasets/iris.csv -T species -M WNN -b 8 -z 256 -m random 
```

If you want to apply another classfier (i.e. RandoForest):

```
python test_classifier.py -i datasets/iris.csv -T species -M RF
```


# Regressor
run the example command

```
python test_regression.py -i datasets/boston.csv -T medv -M WNN -b 63 -z 128
```

If you want to apply another regressor (i.e. RandoForest):

```
python test_classifier.py -i datasets/boston.csv -T medv -M RF
```

