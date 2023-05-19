from sklearn.linear_model import LogisticRegression
from statsmodels.miscmodels.ordinal_model import OrderedModel

from DeepNetwork import *

class StatementClassifier:
    """ Classifies statement embeddings into corresponding truth classes"""
    
    def __init__(self, modelType, data, embeddings_dim):
        """
        Initializes classifier

        Arguments: 
            - modelType (str): either "LR", "OR", or "DN"
            - data (dict): All training, dev, and testing data
            - embeddings_dim (int): length of statements embeddings
            
        """
        self.modelType = modelType
        self.data = data
        
        self.model = None
        if modelType == "LR":
            self.model = LogisticRegression()
        elif modelType == "OR":
            self.model = OrderedModel(data["y_train"],
                        data["X_train"],
                        distr='logit')
        elif modelType == "DN":
            self.model = DeepNetwork(6, embeddings_dim, 128, 64, 0.01, 0.1)
    
    def train(self):
        """
        Trains the classifier on the training data from self.data
        """
        
        if self.modelType == "LR":
            self.model.fit(self.data["X_train"], self.data["y_train"])
        elif self.modelType == "OR":
            self.model = self.model.fit(method='bfgs')
        elif self.modelType == "DN":           
            LEARNING_RATE = 1e-1
            loss_fn = nn.CrossEntropyLoss() 
            optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE) #stochastic gradient descent 

            loss_history, train_accuracy, dev_accuracy = self.model.train_model(torch.Tensor(self.data["X_train"]), 
                                                               torch.LongTensor(self.data["y_train"]), 
                                                               torch.Tensor(self.data["X_dev"]), 
                                                               torch.LongTensor(self.data["y_dev"]),
                                                                loss_fn, optimizer)

                            
    def predict(self, X) -> list:
        """
        Predicts the class for embeddings
        
        Arguments: 
            - X (np.ndarray): All statement dense embeddings to predict a label for
            
        Returns: (List[int]): Predicted class for each embedding
            
        """
        if self.modelType == "LR":
            return self.model.predict(X)
        elif self.modelType == "OR":
            predicted_probs = self.model.model.predict(self.model.params, exog=X)
            return [np.argmax(row) for row in predicted_probs]
        elif self.modelType == "DN":
            predictions, _ = self.model.predict(torch.Tensor(X))
            return predictions

