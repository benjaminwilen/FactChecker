from StatementEmbeddings import *
from StatementClassifier import *

class FactChecker:
    """
    Complete fact checking pipeline to determine the real-world 
    validity of statements
    """

    def __init__(self, embeddings_model, classifier_model):
        """
        Arguments: 
            - embeddings_model (str): Either "bert", "t5-small", or "t5-large" 
            - classifier_model (str): Either  "LR" for linear regression, 
                "OR" for ordinal regression, or "DN" for deep network
        """
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
        """     
        Generates embeddings for each model and updates self.data
        """
        embeddingsGenerator = StatementEmbeddings('politifact_factcheck_data.json')
        embeddingsGenerator.storeAllEmbeddings("bert")
        embeddingsGenerator.storeAllEmbeddings("t5-small")
        embeddingsGenerator.storeAllEmbeddings("t5-large")
        self.data = StatementEmbeddings.retrieveEmbeddings(embeddings_model)
    
    def trainClassifier(self):
        """
        Trains classifier
        """
        print(f"Training {self.classifier_model} classifier")
        self.classifier.train()

    def get_training_data_accuracy(self):
        """
        Gets accuracy of model calculated from training data
        
        Returns: (float) accuracy of model 
        
        """
        print("Predicting test set")
        predictions = self.classifier.predict(self.data["X_test"])
        self.predictions = predictions
        self.y = self.data["y_test"]
        return FactChecker.accuracy(predictions, self.data["y_test"])

    def factCheck(self, statements):
        """
        Arguments: 
            - statements (List): List of statements to be fact checked
            
        Returns: (List) Predictions for those statements 
        """
        statementEmbedding = StatementEmbeddings.getEmbeddings(statements, self.embeddings_model)
        predictions = self.classifier.predict(statementEmbedding)
        return [FactChecker.num2label(prediction) for prediction in predictions]


    def crossValidation(self):
        """
        Helper method used to compute cross validation, modified accordingly
        for linear regression, ordinal regression, and nueral network 
            
        Prints accuracies for each combination of hyper parameters 
        """
        
        # TAILOR TO EACH OF THE MODELS
        
        # Get data into 5 sections 
        X = self.data["X_train"]
        y = self.data["y_train"]
        coupled = [[X[i], y[i]] for i in range(len(X))]
        np.random.shuffle(coupled)

        X1 = []
        y1 = []
        for x,y in coupled[:1200]:
            X1.append(x)
            y1.append(y)
        X1 = np.array(X1)
        y1 = np.array(y1)

        X2 = []
        y2 = []
        for x,y in coupled[1200:2400]:
            X2.append(x)
            y2.append(y)
        X2 = np.array(X2)
        y2 = np.array(y2)

        X3 = []
        y3 = []
        for x,y in coupled[2400:3600]:
            X3.append(x)
            y3.append(y)
        X3 = np.array(X3)
        y3 = np.array(y3)

        X4 = []
        y4 = []
        for x,y in coupled[3600:4800]:
            X4.append(x)
            y4.append(y)
        X4 = np.array(X4)
        y4 = np.array(y4)

        X5 = []
        y5 = []
        for x,y in coupled[4800:]:
            X5.append(x)
            y5.append(y)
        X5 = np.array(X5)
        y5 = np.array(y5)

        # Training folds
        fold1X, fold1y = np.concatenate((X2, X3, X4, X5)), np.concatenate((y2, y3, y4, y5))
        fold2X, fold2y = np.concatenate((X1, X3, X4, X5)), np.concatenate((y1, y3, y4, y5))
        fold3X, fold3y = np.concatenate((X1, X2, X4, X5)), np.concatenate((y1, y2, y4, y5))
        fold4X, fold4y = np.concatenate((X1, X2, X3, X5)), np.concatenate((y1, y2, y3, y5))
        fold5X, fold5y = np.concatenate((X1, X2, X3, X4)), np.concatenate((y1, y2, y3, y4))


        training_X = [fold1X, fold2X, fold3X, fold4X, fold5X]
        training_y = [fold1y, fold2y, fold3y, fold4y, fold5y]
        testing_X = [X1, X2, X3, X4, X5]
        testing_y = [y1, y2, y3, y4, y5]

        
        # Iterate through each combination of hyper-parameters
        for learning_rate in [.001, .01, .1]:
            for dimensions in [(256, 128), (128, 64), (64, 32)]:
                for dropout_prob in [.1, .25, .5]:
                    model = DeepNetwork(6, 768, dimensions[0], dimensions[1], 0.01, dropout_prob)

                    loss_fn = nn.CrossEntropyLoss() 
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                    
                    accuracy = 0
                    for i in range(5):   
                        loss_history, train_accuracy, dev_accuracy = model.train_model(torch.Tensor(training_X[i][:1000]), 
                                                                        torch.LongTensor(training_y[i][:1000]), 
                                                                        torch.Tensor(training_X[i][1000:]), 
                                                                        torch.LongTensor(training_y[i][1000:]),
                                                                            loss_fn, optimizer, verbose=False)
                    
                        predictions, _ = model.predict(torch.Tensor(training_X[i]))

                        # Average accuracy
                        accuracy += DeepNetwork.accuracy(predictions, testing_y[i])
                    accuracy /= 5
                    print(f"Accuracy for LR: {learning_rate} Dim: {dimensions} Dropout: {dropout_prob} ::: {accuracy}")
    
    @staticmethod
    def num2label(num: int) -> str:
        """
        Converts label integer into word representation of label
        """
        labels = ["pants-fire", "false", "mostly-false", "half-true", "mostly-true", "true"]
        return labels[num]

    @staticmethod
    def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float: 
        """
        Calculates accuracy 
        """
        return np.mean(y_pred == y_true)
    