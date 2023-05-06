class DeepNetwork(nn.Module):
    """
    Pytorch implementation for Deep Averaging Network for classification 
    """
    def __init__(self, num_classes, #number of labels / y-values
                     embedding_dim: int, #architecture/pre-processing decision 
                     hidden_dim1: int, #architecture decision
                     hidden_dim2: int, #architecture decision 
                     leaky_relu_negative_slope: float, #hyperparameter
                     dropout_probability: float #hyperparameter
                ):
        """
        Create the network architecture. 
        
        Hints: 
        - Make sure all your dimesions of various layers work out correctly 
        """
        super().__init__()
        self.num_classes = num_classes 
        
        
        # Dropout layer (Used twice)
        self.dropout = nn.Dropout(p=dropout_probability)
        
        # Averaging layer will be completed in forward pass
        
        # Hidden layer 1
        self.hidden1 = nn.Linear(embedding_dim, hidden_dim1)
        
        # Hidden layer 2
        self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        
        # Output layer
        self.theta = nn.Linear(hidden_dim2, num_classes)
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(leaky_relu_negative_slope)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        
    def forward(self, X_batch: torch.Tensor) -> torch.Tensor:
        """
        Given X_batch, make the forward pass through the network. 
        
        The output should be the predicted *log probabilities*. 
        
        Returns: 
            - (torch.Tensor): the log probabilites after the forward pass 
                The shape of this tensor should be (X_batch.shape[0], 2)
                
        Hints: 
            - Look at Pytorch's implemenation of .mean()
            - There should be NO for-loops in this method 
        """
        
        # Step 4: Layer 2 - Hidden layer 1
        hid1 = self.hidden1(X_batch)
        hid1_out = self.leaky_relu(hid1)
        
        # Step 6: Dropout layer
        hid1_out_dropout = self.dropout(hid1_out)
        
        # Step 5: Layer 3 - Hidden layer 2
        hid2 = self.hidden2(hid1_out)
        hid2_out = self.leaky_relu(hid2)
        
        # Step 6: Dropout layer
        hid2_out_dropout = self.dropout(hid2_out)
        
        # Step 7: Layer 4 - Output Layer
        output = self.theta(hid2_out_dropout)
        log_probs = self.log_softmax(output)
        
        return log_probs
        
        
        
    def train(self, X_train, Y_train, X_dev, Y_dev, loss_fn, optimizer, num_iterations=10000, batch_size = 100, check_every=100, verbose=True): 
        """
        Method to train the model. 
        
        No need to modify this method. 
        """
        self.train() # tells nn.Module its in training mode 
                      # (important when we get to things like dropout)
            
        loss_history = [] #We'll record the loss for inspection
        train_accuracy = []
        dev_accuracy = []

        for t in range(num_iterations):
            if batch_size >= X_train.shape[0]: 
                X_batch = X_train
                Y_batch = Y_train
            else: #randomly choose batch_size number of examples 
                batch_indices = np.random.randint(X_train.shape[0], size=batch_size)
                X_batch = X_train[batch_indices]
                Y_batch = Y_train[batch_indices]

            
            # Forward pass 
            pred = self.forward(X_batch)
            loss = loss_fn(pred, Y_batch)

            #Backprop
            optimizer.zero_grad() # clears the gradients from the previous iteration
                                  # this step is important because otherwise Pytorch will 
                                  # *accumulate* gradients for all itereations (all backwards passes)
            loss.backward() # calculate gradients from forward step 
            optimizer.step() # gradient descent update equation 
            
            #Check the loss and train and dev accuracies every 
            if t % check_every == 0:
                loss_value = loss.item() # call .item() to detach from the tensor 
                loss_history.append(loss_value)
                
                #Check train accuracy (entire set, not just batch) 
                train_y_pred, _ = self.predict(X_train)
                train_acc = self.accuracy(train_y_pred, Y_train.detach().numpy()) 
                train_accuracy.append(train_acc)
                
                #Check dev accuracy (entire set, not just batch) 
                dev_y_pred, _ = self.predict(X_dev)
                dev_acc = self.accuracy(dev_y_pred, Y_dev.detach().numpy())
                dev_accuracy.append(dev_acc)
                
                if verbose: print(f"Iteration={t}, Loss={loss_value}")
                
        return loss_history, train_accuracy, dev_accuracy
    
    def predict(self, X): 
        """
        Method to make predictions given a trained model. 
        
        No need to modify this method. 
        """
        self.eval() # tells nn.Module its NOT in training mode 
                 # (important when we get to things like dropout)
    
        pred_log_probs = self.forward(X)
        prediction = np.array([np.argmax(probs.detach().numpy()) for probs in pred_log_probs])
        
        return prediction, pred_log_probs
    
        if self.num_classes == 2: 
            log_pred_pos_class = pred_log_probs[:,1].detach().numpy() #get only the positive class 
            pred_pos_class = np.exp(log_pred_pos_class) #exp to undo the log 
            # decision threshold
            y_pred = np.zeros(X.shape[0])
            y_pred[pred_pos_class>= 0.5] = 1
            return y_pred, pred_pos_class