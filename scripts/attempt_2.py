import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class G:
    EPOCHS = 50
    DATA_SPLIT=0.9
    NUM_WORDS=100000
    OOV_TOKEN='<OOV>'
    PADDING='post'
    TRUNCATING='post'
    MAXLEN=120

all_sentences, all_labels = [], []
with open("\\wsl.localhost\Ubuntu-22.04\home\magellan\envs\tensorflow_cuda_1\data\archive\text.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)

    for col in reader:
        all_sentences.append(col[1])
        all_labels.append(col[2])

total_length=len(all_sentences)
split = int(total_length*G.DATA_SPLIT)

training_sentences=all_sentences[:split]
training_labels=all_labels[:split]
testing_sentences=all_sentences[split:]
testing_labels=all_labels[split:]

training_labels=np.array(training_labels)
testing_labels=np.array(testing_labels)

tokenizer=Tokenizer(oov_token=G.OOV_TOKEN, num_words=G.NUM_WORDS)
tokenizer.fit_on_texts(training_sentences)
sequences=tokenizer.texts_to_sequences(training_sentences)
padded_sequences=pad_sequences(sequences, maxlen=G.MAXLEN, padding=G.PADDING, truncating=G.TRUNCATING)

testing_sequences=tokenizer.texts_to_sequences(testing_sentences)
padded_testing_sequences=padded_sequences(testing_sequences, maxlen=G.MAXLEN, padding=G.PADDING, truncating=G.TRUNCATING)

