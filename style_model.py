from midiparser import MidiData
from tensorflow.keras import layers
from util import showprogress
import os
import random
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

mididata = MidiData('midis/tracks.csv', open_files=['midis/smile.mid'], smoothing=16, transpose=[-4,-2,0,2,4])

dataset = mididata.get_instr_batched_dataset(sequence_length=100, batch_size=64)

def build_model(input_size, output_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_size, embedding_dim, batch_input_shape=[batch_size,None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(output_size)
  ])
  return model

# Call the 'build_model' function and store the output in 'model'
model = build_model(mididata.get_vocabs_sizes()[1], mididata.get_vocabs_sizes()[1], 512, 1024, 64)

# Define a loss function.
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# Put everything together and tell the model to compile with the chosen loss.
model.compile(optimizer='adam', loss=loss)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

model.summary()

history = model.fit(dataset, epochs=10, callbacks=[checkpoint_callback])

model = build_model(mididata.get_vocabs_sizes()[1], mididata.get_vocabs_sizes()[1], 512, 1024, 1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

def generate_text(model, starting_instr):
    # Number of characters we want to generate
    num_generate = 900
    # Turning the input string into its integer representation (because the model only understands
    # integer inputs)
    input_eval = tf.expand_dims(starting_instr, 0)
    generated_instr = []
    temperature = 1.0
    model.reset_states()
  
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        # [0,1,2,3,4]
        # [2.8888, -9.8, 0.8, 0]

        # Pick one character by sampling one value over the distribution that was returned (Note that it's
        # not actually a character but rather its integer representation, aka its index)
        # predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # Add the predicted id to the input for the next prediction
        input_eval = tf.expand_dims([predicted_id], 0)

        # Find the character corresponding to the predicted id and add it to 'text_generated'
        generated_instr.append(predicted_id)
    
    # Once the for is done, return the generated text
    print(len(generated_instr))
    mididata.export_model_output_to_midi(generated_instr, 'generated.mid')
    # return (starting_instr + ''.join(ge))

# This actually calls the method above and prints the output
generate_text(model, starting_instr=[0])