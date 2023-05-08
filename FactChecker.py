from StatementEmbeddings import *
from StatementClassifier import *

class FactChecker:

    def __init__(self, embeddings_model, classifier_model):
        self.embeddings_model = embeddings_model
        self.classifier_model = classifier_model
        self.data = StatementEmbeddings.retrieveEmbeddings(embeddings_model)
        if embeddings_model == "bert":
            self.classifier = StatementClassifier(classifier_model, self.data, 768)
        elif embeddings_model == "t5-small":
            self.classifier = StatementClassifier(classifier_model, self.data, 512)
        elif embeddings_model == "t5-large":
            self.classifier = StatementClassifier(classifier_model, self.data, 1024)

    def trainEmbeddings(self):
            embeddingsGenerator = StatementEmbeddings('politifact_factcheck_data.json')
            embeddingsGenerator.storeAllEmbeddings("bert")
            embeddingsGenerator.storeAllEmbeddings("t5-small")
            embeddingsGenerator.storeAllEmbeddings("t5-large")
            self.data = StatementEmbeddings.retrieveEmbeddings(embeddings_model)
    
    def trainClassifier(self):
        print(f"Training {self.classifier_model} classifier")
        self.classifier.train()

    def get_training_data_accuracy(self):
        print("Predicting test set")
        predictions = self.classifier.predict(self.data["X_test"])
        self.predictions = predictions
        self.y = self.data["y_test"]
        return FactChecker.accuracy(predictions, self.data["y_test"])

    def factCheck(self, statements):
        statementEmbedding = StatementEmbeddings.getEmbeddings(statements, self.embeddings_model)
        predictions = self.classifier.predict(statementEmbedding)
        return [FactChecker.num2label(prediction) for prediction in predictions]

    
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
    