from sklearn.linear_model import LogisticRegression
from statsmodels.miscmodels.ordinal_model import OrderedModel

from DeepNetwork import *

class StatementClassifier:
    def __init__(self, modelType, data):
        self.modelType = modelType
        self.data = data
        
        self.model = None
        if modelType == "LR":
            model = LogisticRegression()
        elif modelType == "OR":
            model = OrderedModel(data["y_train"],
                        data["X_train"],
                        distr='probit')
        elif modelType == "DN":
            model = DeepNetwork(6, 512, 128, 64, 0.01, 0.1)
    
    def train(self, loss_fn=None, optimizer=None, num_iterations=10000, batch_size = 100, check_every=1000, verbose=True):
        if modelType == "LR":
            model.fit(self.data["X_train"], self.data["y_train"])
        elif modelType == "OR":
            model.fit(method='bfgs')
        elif modelType == "DN":
            assert loss_fn != None
            assert optimizer != None
            
            LEARNING_RATE = 1e-1
HIDDEN_DIM1 = 200 
HIDDEN_DIM2 = 100
LEAKY_RELU_NEG_SLOPE = 0.01
DROPOUT_PROB = 0.4 
loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE) #stochastic gradient descent 
            model = model.