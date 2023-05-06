from StatementEmbeddings import *
from StatementClassifer import *

class FactChecker:


    def __init__(self, embeddings_model, classifier_model):
        self.embeddings_model = embeddings_model
        self.classifier_model = classifier_model
        self.classifier = StatementClassifier(embeddings_model, data)
        self.data = StatementEmbeddings.retrieveEmbeddings(embeddings_model)

    def train(self):
        StatementClassifer.train()

    def predict(self, statements):
        StatementClassifier.predict(statements)

    def factCheck(self, statement):
        statementEmbedding = StatementEmbeddings.getEmbedding(statement, self.embeddings_model)
        prediction = self.predict([statement])[0]
        return FactChecker.num2label(prediction)

    
    @staticmethod
    def num2label(num: int) -> str:
        labels = ["pants-fire", "false", "mostly-false", "half-true", "mostly-true", "true"]
        return labels[num]

    @staticmethod
    def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float: 
        """
        Calculates accuracy 
        """
        return np.mean(y_pred == y_true)
    