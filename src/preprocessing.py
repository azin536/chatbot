import json
import nltk
import pickle

from nltk.stem import WordNetLemmatizer
from typing import List, Union

class Preprocessor:
    """Preprocesses the words.

        Attributes:
            - lemmatizer
            - words
            - classes
            - documents
            - ignore_letters
            - intents
        
        Method:
            - preprocess
    """
    def __init__(self) -> None:
        nltk.download('punkt')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.words = list()
        self.classes = list()
        self.documents = list()
        self.ignore_letters = ['!', '?', ',', '.']
        intents_file = open('intents.json').read()
        self.intents = json.loads(intents_file)

    def preprocess(self) -> Union[List, List, List]:
        """Preprocesses the words and tags so later they could convert to 0 and 1.

        Returns:
            words
            classes
            documents
        """
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word = nltk.word_tokenize(pattern)
                words.extend(word)
                self.documents.append((word, intent['tag']))
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])
        words = [self.lemmatizer.lemmatize(w.lower()) for w in words if w not in self.ignore_letters]
        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))
        pickle.dump(words, open('words.pkl', 'wb'))
        pickle.dump(classes, open('classes.pkl', 'wb'))
        return words, classes, self.documents