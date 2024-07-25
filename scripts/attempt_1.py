import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

train_data, test_data=imdb['train'], imdb['test']
training_sentences, training_labels, testing_sentences, testing_labels=[],[],[],[]

for s,l in train_data:
    training_sentences.append(s.numpy().decode('utf8'))
    training_labels.append(l.numpy())
for s,l in test_data:
    testing_sentences.append(s.numpy().decode('utf8'))
    testing_labels.append(l.numpy())
training_labels=np.array(training_labels)
testing_labels=np.array(testing_labels)

tokenizer=Tokenizer(oov_token='<OOV>', num_words=10000)
tokenizer.fit_on_texts(training_sentences)
sequences=tokenizer.texts_to_sequences(training_sentences)
padded_sequences=pad_sequences(sequences, maxlen=120, padding='post', truncating='post')

testing_sequences=tokenizer.texts_to_sequences(testing_sentences)
padded_testing_sequences=pad_sequences(testing_sequences, maxlen=120,padding='post', truncating='post')

model=tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=120),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(units=1024, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

model.fit(padded_sequences, training_labels, epochs=20, validation_data=(padded_testing_sequences, testing_labels))