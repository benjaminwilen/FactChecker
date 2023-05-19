import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5Model
import torch
from tqdm import tqdm

class StatementEmbeddings:
    """ Generates sentence embeddings"""
    
    def __init__(self, dataFile):
        """
        Reads the file containing raw data
       
        Arguments: 
            - dataFile (str): File path to json file containing training statements
            
        """
        self.dataFrame = data = pd.read_json(dataFile, lines=True)
        self.data = self.getDataSets(self.dataFrame, ["pants-fire", "false", "mostly-false", "half-true", "mostly-true", "true"] )

    def getDataSets(self, dataFrame, labels, numTrain=1000, numDev=200, numTest=200) -> dict():
        """
        Generates testing, dev, and training examples from dataframe, excluding 2022 data

        Arguments: 
            - dataFrame (pandas.df): Dataframe of data 
            - labels (List): Classses of data
            - numTrain (int): number of training examples for each label
            - numDev (int): number of dev examples for each label  
            - numTest (int): number of testing examples for each label  
            
        Returns: (dict): Keys are "X_train", "y_train", "X_dev", "y_dev", "X_test", "y_test",
                        and values are np.arrays of the corresponding data
            
        """
        X_train = []
        y_train = []
        X_dev = [] 
        y_dev = []
        X_test = []
        y_test = []
        for label_i, label in enumerate(labels):
            # Get examples with label of for loop label
            collection = np.array(dataFrame.loc[dataFrame['verdict'] == label])
            
            y_original = collection[:,0]
            X_original = collection[:,2]
            speakers = collection[:,1]
            dates = collection[:,3]

            correct_years_collection = []
            # Get rid of 2022
            for i in range(len(y_original)):
                if dates[i][-4:] != "2022":
                    correct_years_collection.append([StatementEmbeddings.formatStatement(speakers[i], X_original[i]), y_original[i]])
                
            correct_years_collection = np.array(correct_years_collection)
            
            # Shuffle data before generating splits
            np.random.shuffle(correct_years_collection) 
            X_train.extend(correct_years_collection[:numTrain, 0])
            y_train.extend([label_i for j in range(numTrain)])
            X_dev.extend(correct_years_collection[numTrain:numTrain+numDev, 0])
            y_dev.extend([label_i for j in range(numDev)])
            X_test.extend(correct_years_collection[numTrain+numDev:numTrain+numDev+numTest, 0])
            y_test.extend([label_i for j in range(numTest)])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_dev = np.array(X_dev)
        y_dev = np.array(y_dev)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        assert len(X_train) == len(labels) * numTrain
        assert len(y_train) == len(labels) * numTrain
        assert len(X_dev) == len(labels) * numDev
        assert len(y_dev) == len(labels) * numDev
        assert len(X_test) == len(labels) * numTest
        assert len(y_test) == len(labels) * numTest
        
        return {"X_train": X_train, 
                "y_train": y_train, 
                "X_dev": X_dev, 
                "y_dev": y_dev, 
                "X_test": X_test, 
                "y_test": y_test}


    def storeEmbeddings(self, modelType, dataSet, embeddings, y):
        """
        Arguments: 
            - modelType (str): either "bert", "t5-small", "t5-large"
            - dataSet (str): either "train", "dev", "test"  
            - embeddings (List[List]): statement embeddings for that model and dataset
            - y (List): labels for the statement embeddings

        This function stores embeddings in json file
        (make sure json file doesn't exist before function is run)
        
        """
        
        embeddings_data = [[embeddings[i], y[i]] for i in range(embeddings.shape[0])]
        # creating a list of index names
        index_values = np.arange(embeddings.shape[0])
        
        # creating a list of column names
        column_values = ['embeddings', "label"]
        
        # creating the dataframe
        data = pd.DataFrame(data = embeddings_data, 
                        index = index_values, 
                        columns = column_values)
        data.to_json(f'datasets/{modelType}-{dataSet}-data.json', orient = 'records', index = 'true')
    
    
    def storeAllEmbeddings(self, model):
        """
        Gets and stores all embeddings for that model in JSON files
        
        Arguments: 
            - model (str): either "bert", "t5-small", "t5-large"  
            
        """
        
        for dataSet in ["train", "dev", "test"]:
            X = self.data[f"X_{dataSet}"]
            y = self.data[f"y_{dataSet}"]
            
            embeddings = StatementEmbeddings.getEmbeddings(X)
            self.storeEmbeddings(model, dataSet, embeddings, y)

    
    @staticmethod
    def formatStatement(speaker, statement) -> str:
        """ Appends speaker to start of statement
            Used to handle cases where statement is specific to speaker          
        """
        return f"{speaker} said, '{statement}'"
   
   
    @staticmethod
    def getEmbeddings(statements, modelType) -> np.ndarray:
        """
        For each model, tokenizes the statement, passses it through model,
        and retrieves cls token embedding from hidden states
        
        Arguments: 
            - statements (List): Statements to retrieve embeddings for  
            - modelType (str): either "bert", "t5-small", "t5-large"
            
        Returns: (np.ndarray) Statement embedding for each statement         
        """
        if modelType == "bert":
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
            
            embeddings = np.empty((len(statements), 768))
            
            #print(f"Progress for {modelType} statement embeddings")
            for i, statement in enumerate(statements):
                input_ids = torch.tensor(tokenizer.encode(statement)).unsqueeze(0)
                outputs = model(input_ids, output_hidden_states=True)
                last_hidden_states = outputs.hidden_states[-1]
                cls_tok = last_hidden_states[0,0,:]
                embeddings[i] = cls_tok.detach()
    

        elif modelType == "t5-small":
            tokenizer = T5Tokenizer.from_pretrained("t5-small")
            model = T5Model.from_pretrained("t5-small")
        
            embeddings = np.empty((len(statements), 512))

            #print(f"Progress for {modelType} statement embeddings")
            for i, statement in enumerate(statements):
                input_ids = tokenizer.encode(statement, return_tensors="pt")  # Batch size 1
                outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)
                last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
                cls_tok = last_hidden_states[0,0,:]
                embeddings[i] = cls_tok.detach()

        elif modelType == "t5-large":
            tokenizer = T5Tokenizer.from_pretrained("t5-large")
            model = T5Model.from_pretrained("t5-large")
        
            embeddings = np.empty((len(statements), 1024))

            #print(f"Progress for {modelType} statement embeddings")
            for i, statement in enumerate(statements):
                input_ids = tokenizer.encode(statement, return_tensors="pt")  # Batch size 1
                outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)
                last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
                cls_tok = last_hidden_states[0,0,:]
                embeddings[i] = cls_tok.detach()
        
        return embeddings

    
    @staticmethod
    def retrieveEmbeddings(modelType) -> dict():
        """
        After statement embeddings have been generated and stored, this function
        can be called to retrieve them from their json files without having to pass the statements
        through the models again

        Arguments: 
            - modelType (str): either "bert", "t5-small", "t5-large"
            
        Returns: (dict): Keys are "X_train", "y_train", "X_dev", "y_dev", "X_test", "y_test",
                        and values are np.arrays of the corresponding data
        """
        train_df = pd.read_json(f'datasets/{modelType}-train-data.json', orient ='records')
        dev_df = pd.read_json(f'datasets/{modelType}-dev-data.json', orient ='records')
        test_df = pd.read_json(f'datasets/{modelType}-test-data.json', orient ='records')

        X_train = np.array([np.array(row) for row in train_df["embeddings"]])
        y_train = np.array(train_df["label"])
        X_dev = np.array([np.array(row) for row in dev_df["embeddings"]])
        y_dev = np.array(dev_df["label"])
        X_test = np.array([np.array(row) for row in test_df["embeddings"]])
        y_test = np.array(test_df["label"])

        vectorLength = 0
        if modelType == "bert":
            vectorLength = 768
        elif modelType == "t5-small":
            vectorLength = 512
        elif modelType == "t5-large":
            vectorLength = 1024
        assert X_train.shape == (6000, vectorLength)
        assert y_train.shape == (6000,)
        assert X_dev.shape == (1200, vectorLength)
        assert y_dev.shape == (1200,)
        assert X_test.shape == (1200, vectorLength)
        assert y_test.shape == (1200,)

        return {"X_train": X_train, 
                "y_train": y_train, 
                "X_dev": X_dev, 
                "y_dev": y_dev, 
                "X_test": X_test, 
                "y_test": y_test}
