
import json
import nltk
import random
import pickle

import numpy as np
import tensorflow.keras as tfk

from tkinter import Text, Scrollbar, Button, Tk
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Callable

from src.model_building import ModelBuilder

nltk.download('punkt')
nltk.download('wordnet')


def load_model() -> tfk.Model:
    model_builder = ModelBuilder((87, ), 9)
    model = model_builder.build_model()
    model.load_weights('chatbot.h5')
    return model


def clean_up_sentence(sentence: str) -> List:
    """Cleanes sentences to lower words and drop duplicates.

    Args:
        sentence (str): sentence

    Returns:
        List: list of words of the sentences
    """
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def get_bag_of_words(sentence: str, words: List, show_details=True) -> np.ndarray:
    """Gets the words as 0 and 1.

    Args:
        sentence (str): sentence
        words (List): words
        show_details (bool, optional): Defaults to True.

    Returns:
        np.ndrray: a list of 0 and 1 which shows whether a word in the input was in
                   dataset or not. 
    """
    sentence_words = clean_up_sentence(sentence=sentence)
    bag = [0] * len(words)
    for s_word in sentence_words:
        for i, word in enumerate(words):
            if word == s_word:
                bag[i] = 1
                if show_details:
                    print(f'Fount {word} in bag')
    return np.array(bag)


def predict_class(sentence: str, words: List, classes: List, model: tfk.Model) -> List:
    """Predicts class.

    Args:
        sentence (str): inpute sentence
        words (List): words from database
        classes (List): classes
        model (tfk.Model): trained adn saved model

    Returns:
        List: list fo dictionaries with predicted class and probability.
    """
    bag = np.array(get_bag_of_words(sentence, words, show_details=False))
    bag = np.expand_dims(bag, axis=0)
    result = model.predict(bag)[0]
    threshold = 0.25
    results = [[i, r] for i, r in enumerate(result) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = list()
    for res in results:
        return_list.append({'intent': classes[res[0]], 'probability': str(res[1])})
    return return_list


def get_response(ints: List, intents_json: Dict) -> str:
    """Gets response to input sentence

    Args:
        ints (List): predicted class 
        intents_json (Dict): database

    Returns:
        str: responce to input sentence
    """
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for int_ in list_of_intents:
        if int_['tag'] == tag:
            result = random.choice(int_['responses'])
            break
    return result


def wrap_send_func(entry_box: Text, chat_box: Text, words: List, classes: List,
                   model: tfk.Model, intents: List) -> Callable:
    """Warps the send function.

    Args:
        entry_box (_type_): entry box
        chat_box (_type_): chat box
        words (_type_): words
        classes (_type_): classes
        model (tfk.Model): trained and saved model
        intents (_type_): intents in database

    Returns:
        Callable: send function
    """
    def send() -> None:
        """Sends the respond according to input.
        """
        message = entry_box.get("0.0", "end-1c").strip()
        entry_box.delete("0.0", 'end')
        if message != '':
            chat_box.config(state='normal')
            chat_box.insert('0.0', f'You:{message}\n\n')
            chat_box.config(foreground="#446665", font=("verdana", 12))
            ints = predict_class(message, words, classes, model)
            result = get_response(ints, intents)
            chat_box.insert('1.0', f'Bot:{result}\n\n')
            chat_box.config(state='disabled')
            chat_box.yview('0.0')
    return send

        
def main():
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model()
    root = Tk()
    root.title("Chatbot")
    root.geometry("400x500")
    root.resizable(width=False, height=False)
    entry_box = Text(root, bd=0, bg='pink', width='29', height='5', font='Arial')
    chat_box = Text(root, bd=0, bg='white', height='8', width='50', font='Arial')
    chat_box.config(state='disabled')
    scrollbar = Scrollbar(root, command=chat_box.yview, cursor='heart')
    chat_box['yscrollcommand'] = scrollbar.set
    entry_box.place(x=106, y=401, height=90, width=265)
    chat_box.place(x=6, y=6, height=386, width=370)
    scrollbar.place(x=376, y=6, height=386)
    send = wrap_send_func(entry_box, chat_box, words, classes, model, intents)
    send_button = Button(root, font=('Verdana', 12, 'bold'), text='send', width='12',
                         height=5, bd=0, bg='#f9a602', activebackground='#3c9d9b',
                         fg='#000000', command=send)
    send_button.place(x=6, y=401, height=90, width=100)
    root.mainloop()


if __name__ == '__main__':
    main()
# intents = json.loads(open('intents.json').read())
# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = load_model()
# root = Tk()
# root.title("Chatbot")
# root.geometry("400x500")
# root.resizable(width=False, height=False)
# entry_box = Text(root, bd=0, bg='white', width='29', height='5', font='Arial')
# chat_box = Text(root, bd=0, bg='white', height='8', width='50', font='Arial')
# chat_box.config(state='disabled')
# scrollbar = Scrollbar(root, command=chat_box.yview, cursor='heart')
# chat_box['yscrollcommand'] = scrollbar.set
# entry_box.place(x=128, y=401, height=90, width=265)
# chat_box.place(x=6, y=6, height=386, width=370)
# scrollbar.place(x=376, y=6, height=386)

# def send() -> None:
#     """Sends the respond according to input.
#     """
#     message = entry_box.get("1.0", "end-1c").strip()
#     entry_box.delete("0.0", 'end')
#     if message != '':
#         chat_box.config(state='normal')
#         chat_box.insert('0.0', f'You:{message}\n\n')
#         chat_box.config(foreground="#446665", font=("verdana", 12))
#         ints = predict_class(message, words, classes, model)
#         result = get_response(ints, intents)
#         chat_box.insert('1.0', f'Bot:{result}\n\n')
#         chat_box.config(state='disabled')
#         chat_box.yview('0.0')

# send_button = Button(root, font=('Verdana', 12, 'bold'), text='send', width='12',
#                         height=5, bd=0, bg='#f9a602', activebackground='#3c9d9b',
#                         fg='#000000', command=send)
# send_button.place(x=6, y=401, height=90)
# root.mainloop()