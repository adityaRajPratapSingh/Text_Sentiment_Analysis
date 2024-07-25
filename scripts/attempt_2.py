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
    OPTIMIZER=tf.keras.optimizers.Adam()
    LOSS=tf.keras.losses.sparse_categorical_crossentropy
    STOP_PATIENCE=20
    EMB_DIM=64

all_sentences, all_labels = [], []
with open("/home/magellan/envs/tensorflow_cuda_1/project/data/archive/text.csv") as csvfile:
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

lr_schedule = tf.keras.callbacks.LerarningRateScheduler(
    lambda epoch:1e-8 * 10**(epoch /20)
)

early_stop=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=G.STOP_PATIENCE)

model=tf.keras.Sequential([
    tf.keras.layers.Embedding(G.NUM_WORDS, G.EMB_DIM, input_length=G.MAXLEN),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(
    loss=G.LOSS,
    optimizer=G.OPTIMIZER,
    metrics=['accuracy']
)

model.fit(padded_sequences, training_labels, epochs=G.EPOCHS, validation_data=(padded_testing_sequences, testing_labels), callbacks=[lr_schedule, early_stop])
