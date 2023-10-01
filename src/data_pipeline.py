import random

import numpy as np

from nltk.stem import WordNetLemmatizer
from typing import List, Union


class DataLoader:
    """Loads and tokenize data into data and labels.

    Atrributes:
        - words: words in the dataset
        - classes: tags in the dataset
        - documents: tuple consists of bags of words and corresponding tags in 0 and 1s

    Methods:
        - preprocess
    """
    def __init__(self, words: List, classes: List, documnets: List) -> None:
        self.words = words
        self.classes = classes
        self.documents = documnets
        self.lemmatizer = WordNetLemmatizer()
    
    def create_train_data(self) -> Union[List, List]:
        """Prepares word and corresponding labels for training.
        
        Returns:
            train_x (list): train data
            train_y (list): label for data
        """
        training = list()
        output_empty = [0] * len(self.classes)

        for doc in self.documents:
            bag = list()
            word_patterns = doc[0]
            word_patterns = [self.lammetizer.lammetize(word.lower()) for word in word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)
                output_row = list(output_empty)
                output_row[self.classes.index(doc[1])] = 1
                training.append((bag, output_row))

        random.shuffle(training)
        training = np.array(training)
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        return train_x, train_y