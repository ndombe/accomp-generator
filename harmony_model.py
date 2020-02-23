from midiparser import MidiData
from tensorflow.keras import layers
from util import showprogress
import os
import random
import tensorflow as tf

tf.enable_eager_execution()


mididata = MidiData('midis/tracks.csv', open_files=['midis/smile.mid'])

dataset = mididata.get_instr_batched_dataset(sequence_length=100, batch_size=64)
def build_model(input_size, output_size, embedding, batch_size, rnn_units):

    return tf.keras.Sequential([
        layers.Embedding(input_size, embedding, batch_input_shape=[batch_size,None]),
        layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True, stateful=True,
                            recurrent_initializer='glorot_uniform')),
        layers.Dropout(0.3),
        layers.Dense(output_size)
    ])

model = build_model(mididata.get_vocabs_sizes()[1], mididata.get_vocabs_sizes()[1], 512, 64, 1024)

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

checkpoint_dir = './models/harmony/checkpoints'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'ckpt_{epoch}'),
    save_weights_only=True
)

history = model.fit(dataset, epochs=80, callbacks=[checkpoint_callback])

model = build_model(mididata.get_vocabs_sizes()[0], mididata.get_vocabs_sizes()[1], 512, 1, 1024)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

melodydata = MidiData('midis/tracks.csv', open_files=['midis/traditional tidings.mid'])
melody = melodydata.get_frames_as_int[0]

def generate_instr(model, melody):
    predictions = []
    k = 10
    for idx in melody:
        mel_input = tf.expand_dims([idx], 0)
        predictions = model(mel_input)
        predictions = tf.squeeze(predictions, 0)

        predicted_id = tf.math.top_k(predictions[0],k=k)[1].numpy()[random.randint(0,k-1)]

        predictions.append(predicted_id)
    
    mididata.export_model_output_to_midi(predictions, 'generated.mid')
generate_instr(model, melody)