from midiparser import MidiData
from tensorflow.keras import layers
from util import showprogress
from matplotlib import pyplot as plt
import os
import random
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

embedding_size = 512
batch_size = 16
rnn_units = 1024

mididata = MidiData('midis/tracks.csv', open_files=['midis/smile.mid'], smoothing=32, transpose=[-4,-3,-1,-2,0,1,2,3,4,5,6,7])

trainds, devds, testds = mididata.get_instr_batched_dataset(sequence_length=60, batch_size=batch_size, ratio=[0.6,0.2])

def build_model(input_size, output_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_size, embedding_dim, batch_input_shape=[batch_size,None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    layers.BatchNormalization(),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(output_size)
  ])
  return model

model = build_model(mididata.get_vocabs_sizes()[1], mididata.get_vocabs_sizes()[1], embedding_size, rnn_units, batch_size)

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def chopinScore(labels, logits):
  return tf.py_function(mididata.chopin, [labels, logits], tf.float64)

model.compile(optimizer='adam', loss=loss, metrics=[chopinScore])

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_1')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

model.summary()

history = model.fit(trainds, epochs=200, callbacks=[checkpoint_callback], validation_data=devds)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
plt.plot(history.history['chopinScore'])
plt.plot(history.history['val_chopinScore'])
plt.title('Chopin Score')
plt.xlabel('epochs')
plt.ylabel('Chopin Score')
plt.show()

model = build_model(mididata.get_vocabs_sizes()[1], mididata.get_vocabs_sizes()[1], embedding_size, rnn_units, 1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

def generate_style(model, starting_instr):
    num_generate = 900
    input_eval = tf.expand_dims(starting_instr, 0)
    generated_instr = []
    #denominator = 1.0
    model.reset_states()
  
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        generated_instr.append(predicted_id)

    print(len(generated_instr))
    mididata.export_model_output_to_midi(generated_instr, 'generated.mid')

generate_style(model, starting_instr=[0])