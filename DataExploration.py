from FactChecker import *

if __name__ == "__main__":
    fc = FactChecker("bert", "DN")
    fc.trainClassifier()
    print(fc.get_training_data_accuracy())