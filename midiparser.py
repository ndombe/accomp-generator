from ast import literal_eval
from copy import deepcopy
from itertools import product
from tensorflow.contrib.data import sliding_window_batch
from mido import Message, MidiFile, MidiTrack
from os import listdir
from random import randint
import csv
import numpy as np
import tensorflow as tf
import util

FILE_EXTENSION = '.dimp' # Digital Music Representation
NOTES_RANGE = 128 # a.k.a the width of one frame
DEFAULT_TEMPO = 120
DEFAULT_TICKS_PER_SECOND = 96
MELODY_TRACK = 'melody'
INSTRUMENT_TRACK = 'instrument'

class MidiData():
    # *********PUBLIC/IMPORTANT METHODS************
    def __init__(self, tracks_info_file, open_files=None, open_from_folder=None, smoothing=16,
            transpose=[0]):
        """
        Creates an instance of MidiData. This is called automatically when you call 'MidiData()',
        see example usage below.
        @param tracks_info: the path to the location of the cvs file containing information about
                the midi files
        @param open_files (recommanded): a list of the files to open in this MidiData object
        @param open_from_folder: use this argument if you want to open all midi files inside of
                a specific folder without listing them all.
                NOTE: Only one of 'open_files' or 'open_from_folder' must be used.
        @param smoothing (optional): an integer greater or equal to 1. It determines how much
                granularity we want the frames to have. A smoothing of 1 will have the frames as
                granular as they can be (96 frames per beat). A greater smoothing value means less
                granularity (useful for only retaining the most important note/chord changes and
                ignoring the potential small variations, especially when there are many such small
                variations). Default 16.
        @param transpose (optional): an array of at least one integer. This will decide if we want
                to transpose the midi data. If this argument is used, it should usually take a value
                like this [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]. With an argument like this one,
                the returned data would contain, in addition to the original data in the midi file,
                4 copies of the same data transposed respectively 4, 3, 2 and 1 times downward, then
                7 copies of the original data transposed respectively 1...7 times upward.
                Default [0].
        
        Example Usage:
        data = MidiData('path/to/csv/file/tracks.csv', open_files=['midi/folder/song1.mid',
                                                                    'midi/folder/song2.mid'])
        """

        assert open_from_folder == None or open_files == None
        assert smoothing >= 1
        assert len(transpose) > 0
        self.__transpose__ = transpose
        self.__read_tracks_info_file__(tracks_info_file)
        self.__smoothing__ = smoothing
        self.__generate_reduced_melody_vocab__()
        if open_from_folder: self.__open_from_folder__(open_from_folder)
        elif open_files: self.__open_files__(open_files)
        self.__smooth_frames__()
        print('Extracting vocabulary...')
        frame2str = lambda x: str(tuple(x))
        
        # NOTE(1): We're disabling unreduced melodies. A reduced melody vocabulary is generated
        # before even reading the files so that we don't need to load any 'new_melody' in advance
        # but can any such 'new_melody' file after training, and it will still read from the same
        # vocabulary.
        # mel_as_str = np.apply_along_axis(frame2str, 1, all_melodies).tolist()
        instr_as_str = np.apply_along_axis(frame2str, 1, self.__songs__[1]).tolist()
        # reduced_mel_as_str = np.apply_along_axis(
        #     frame2str, 1, self.extract_notes_only(self.__songs__[0])).tolist()
        reduced_instr_as_str = np.apply_along_axis(
            frame2str, 1, self.extract_notes_only(self.__songs__[1], with_root=True)).tolist()

        # See NOTE(1).
        # mel_vocab, self.__mel_as_int__ = np.unique(mel_as_str, return_inverse=True)

        self.__reduced_mel_as_int__ = np.apply_along_axis(
            lambda x: self.__reduced_mel_vocab__.index(str(tuple(x))), -1,\
            self.extract_notes_only(self.__songs__[0]))
        self.__instr_vocab__, self.__instr_as_int__ = np.unique(instr_as_str, return_inverse=True)
        self.__reduced_instr_vocab__, self.__reduced_instr_as_int__ = np.unique(
            reduced_instr_as_str, return_inverse=True)

        self.__vocab__ = (None, self.__instr_vocab__)
        self.__reduced_vocab__ = (self.__reduced_mel_vocab__, self.__reduced_instr_vocab__)
        print('Extracted vocabulary as int.')
        print('Extracting vocabulary as frames...')
        self.__np_instr_vocab__ = self.instr_idx_to_frames(np.arange(len(self.__instr_vocab__)))
        # TODO(ndombe): until we start using the mel_vocab_nonzero_count, it's wasteful to compute
        # it.
        # self.__mel_vocab_nonzero_count__ =\
        #     np.apply_along_axis(
        #         lambda x: len(x) if len(x) > 0 else 1, -1, np.nonzero(self.__np_mel_vocab__))
        self.__instr_vocab_nonzero_count__ =\
            np.apply_along_axis(
                lambda x: len(np.nonzero(x)) if len(np.nonzero(x)) > 0 else 1,
                -1, self.__np_instr_vocab__)
        print('Vocabularty size (reduced-melody: {}, instr: {}, reduced-instr: {})'\
            .format(len(self.__reduced_mel_vocab__), len(self.__instr_vocab__),\
            len(self.__reduced_instr_vocab__)))

        print('MidiData object ready!')


    def chopin(self, labels, logits, k=1, from_frames=False, cut_threshold=.75):
        """
        Compute the Chord Proxy Substitution (Chopin) score for a some given labels and logits. This
        method is meant to be called from within a metrics or loss function, with the labels and
        logits fromt the model.
        NOTE: We assume that this method is only called in reference to the instrumental part. Some
                implicit calls to the instrumental vocabulary are made.
        @param labels: Tensor corresponding to the labels. If `from_frames` is False, it is
                assumed that `labels` contains integers representing the index of frames in the
                vocabulary; if `from_frames` is True, it is assumed that `labels` contains frames.
        @param logits: Tensor corresponding to the logits. If `from_frames` is False, it is
                assumed that the model's last layer has as many neurons as there are frames in the
                vocabulary, implying that the last dimension of `logits` is as long as the
                vocabulary. If `from_frames` is True, it is assumed that the model has as many
                neurons as there are notes in a frame, implying that the last dimension of `logits`
                is as long as a frame, which is that the last dimension is a probability on each
                note of the frame for whether or not they should be played.
        @param k(optional): an integer greater or equal to 1. If `from_frames` is False, `k`
                represents the number of top logits value to aggregate in computing the chopin
                score. If `from_frames` is True, `k` represents the number of top most probable
                frames from the vocabulary we want to choose from each time. Default 1.
        @param cut_threshold(optional): a float ranging from 0 to 1 indicating where we want to cut
                the probabilities before fetching the frames.
                
        @return score: chopin score
        """
        labels = labels.numpy().astype(int)
        if from_frames:
            logits = self.probs_to_frames(logits.numpy(), k=k, threshold=threshold)
        else:
            logits = tf.math.top_k(logits, k=k)[1].numpy()
        logits = logits.astype(int)
        
        score = 0
        if from_frames:
            score = util.chopin(labels, logits)
        else:
            for i in range(k):
                logits_i = logits[:,:,i]
                get_frames = lambda x: self.instr_idx_to_frames(x)
                score += util.chopin(np.apply_along_axis(get_frames, -1, labels),
                            np.apply_along_axis(get_frames, -1, logits_i))
            score = score / k
        
        
        return score

    def probs_to_frames(self, probs, k=1, threshold=.6):
        """
        Retreive a frame from the vocabulary that is the most probable match given the provided
        notes probabilities argument.
        NOTE: This method is assumed to be used for the instrumental, implying that there is an
                implicit call to the instrumental vocabulary.
        
        @param probs: an nd-array. The last dimension is the length of frame (# notes in a frame).
                Each element of this last dimension represent the probability of the corresponding
                note to be played in that frame.
        @param k(optional): an integer greater or equal to 1 representing how many of the top
                likely frames we consider when picking the output frame.
        @param threshold(optional): a float number from 0 to 1 determining where to cut the
                probabilities, i.e. every probability below `threshold` will be set to 0 and will
                not be taken into account when looking for the most probable frame. Default 0.6
        """
        self.__probs_to_frames_k = k
        self.__probs_to_frames_threshold = threshold
        frames = np.apply_along_axis(self.__probs_to_k_frame__, -1, probs)
        idx = self.get_frame_index(frames)
        return frames

    def get_frame_index(self, frame):
        """
        Get the index of the given frame from the vocabulary.
        NOTE: This is assumed to be used for the instrumental, implying that we're fetching from the
                instrumental vocabulary.
        @param frame: a one dimensional list/array of the length of a frame legnth representing a
                frame.
        @return an integer representing the index of the frame in the vocabulary.
        """
        return np.where(self.__vocab__[1]==str(tuple(frame)))

    def compute_vocab_chopin_distribution(self, ref_frames, from_frame=False):
        """
        TODO: complete documentation
        NOTE (Assumption): this is used to fetch frames from the instrumental vocabulary only.
        TODO: add a flag for melody
        """
        if not from_frame:
            # `ref` is an integer representing the index of the desired reference frame
            # TODO: check for integer/float type
            ref_frames = np.array(ref_frames)
            ref_frames = self.instr_idx_to_frames(ref_frames)
        
        candidates = np.array(self.__np_instr_vocab__)
        def get_chopin(ref_frame):
            references = np.array([ref_frame,]*self.get_vocabs_sizes()[1])
            score = util.chopin(references, candidates)
            return score

        scores = np.apply_along_axis(get_chopin, -1, ref_frames)

        
        return scores

    def get_mel_vocab(self):
        # TODO: missing documentation

        # See NOTE(1)
        return self.__reduced_mel_vocab__

    def get_instr_vocab(self, reduced=False):
        # TODO: missing documentation
        return self.__reduced_instr_vocab__ if reduced else self.__instr_vocab__

    def get_vocabs(self, reduced=False):
        """
        DEPRECATED! Use `get_instr_vocab` or `get_mel_vocab` instead. This functino will
        be removed.

        Get the unique frames from the loaded data, aka the vocabulary. Note that this returns 2
        sets of frames, the first one corresponding to the melody and the second one to the
        instrumental.

        TODO(ndombe): update documentation with 'reduced'

        Example Usage:
        # Assuming you declared a MidiData object like so
        data = MidiData('path_to_tracks_info.csv', open_files=['file1.mid', 'file2.mid'])
        mel_vocab, instr_vocab = data.get_vocabs()

        This will return a tuple where the first element is a numpy array containing the unique
        frames for the melody, and the second element is a numpy array with the unique frames for
        the instrumental.
        For instance, if the list of frames for the instrumental looks something like this:
        [
            [0,1,1,0],
            [0,1,1,0],
            [0,1,1,0],
            [1,1,1,0]
        ]

        The unique frames array will be:
        [
            [0,1,1,0],
            [1,1,1,1]
        ]
        """
        vocab = self.__vocab__ if not reduced else self.__reduced_vocab__
        return None, vocab[1]

    def get_mel_vocab_size(self):
        # TODO: missing documentation
        return len(self.get_mel_vocab())

    def get_instr_vocab_size(self, reduced=False):
        # TODO: missing documentation
        return len(self.get_instr_vocab(reduced))

    def get_vocabs_sizes(self, reduced=False):
        """
        DEPRECATED! Use `get_instr_vocab_size` or `get_mel_vocab_size` instead. This functino will
        be removed.

        Get the number of unique frames, aka the size of the vocabulary. Note that this returns 2
        sizes, the first one corresponding to the size of unique frames in the melody, and the
        second one corresponding to the size of frames in the instrumental.

        TODO(ndombe): update documentation with 'reduced'

        Example Usage:
        # Assuming you declared a MidiData object like so
        data = MidiData('path_to_tracks_info.csv', open_files=['file1.mid', 'file2.mid'])
        mel_vocab_size, instr_vocab_size = data.get_vocabs_sizes()

        This is the same thing as calling "get_unique_frames()" then getting the length, like so:
        data = MidiData('path_to_tracks_info.csv', open_files=['file1.mid', 'file2.mid'])
        mel_vocab, instr_vocab = data.get_vocab()
        mel_vocab_size = len(mel_vocab)
        instr_vocab_size = len(instr_vocab)
        """
        vocab = self.__vocab__ if not reduced else self.__reduced_vocab__
        return None, len(vocab[1])

    def get_frames_as_int(self, reduced=False):
        """
        Get the original melody and instrumental where each frame is replaced by its integer
        representation according to the vocabulary.

        TODO(ndombe): update documentation with 'reduced'

        Example Usage:
        # Assuming you declared a MidiData object
        data = MidiData(...)
        mel_as_int, instr_as_int = data.get_frames_as_int()
        """

        # See NOTE(1)
        return (self.__reduced_mel_as_int__,
                self.__instr_as_int__ if not reduced else self.__reduced_instr_as_int__)

    def get_instr_batched_dataset(self, sequence_length=(96), batch_size=64, buffer_size=10000,
            output_frames=False, reduced=False, ratio=[.8,.1], stride=5):
        """
        Get the original frames expressed as their respective integer index according to the
        unique frames vocabulary. Note that this returns 2 arrays, the first one containing the
        indices corresponding to the frames of the original melody, and the second one containing
        the indices corresponding to the frames of the original instrumental.
        @param sequence_length: an integer indicating the window length of eacch sequence we feed
                into the model when learning. Deafult 96.
        @param batch_size: in integer that determines the number of sequecences we want to have in
                each batch that we'll feed to the model. Default 64.
        @param buffer_size: in integer used when shuffling the data. Default 10000.
        @param reduced: a boolean determining if we want the dataset to contain indices into the
                vocabulary of the original frames or indices into a vocabulary of the reduced
                frames. Reduced frames are frames that only contain information about the activated
                notes, regardless of what octaves they belong to. Reduced frames from the melody
                vocabular are of length 12, i.e they only contain information about the activation
                status of the 12 musical notes (C, C#, D, D#, E, F, F#, G, G#, A, A# and B). Whereas
                reduced frames from the instrumental vocabular are of length 24, i.e, on octave with
                exactly one activated note (the root note) and the other one with the activation
                status of any remaining note in the original frame. Default False.
        @param ratio: a list of 2 float numbers both between 0 and 1. The first number indicate the
                percentage of the final number of batches that we want to allocate as training data.
                The second is the percentage of total batch number to allocate as dev/validation
                data. The remaining will be allocated to the test data.
        @param stride: an integer. This represents the stride to be used when generating the
                sequences. Indeed, the sequences are going to be generated by taking each
                consecutive `sequence_length` frames and skipping `stride` frames from one sequence
                to the other. The stride has to be greater than 0. Defatul 5.

        @return a 3-tuple where each element represents respectively the training dataset, the
                dev/validation dataset and the test dataset. Each dataset contains 2 sequences: the
                first sequence corresponds to the input we would feed to the model and the second
                sequence corresponds to the output we expect the model to predict (true label). The
                output sequence is simply the input sequence minus its first frame and plus the
                frame that should come right after the last frame in the input sequence.

        Example Usage:
        # Assuming you declared a MidiData object like so
        data = MidiData('path_to_tracks_info.csv', open_files=['file1.mid', 'file2.mid'])
        instr_dataset = data.get_instr_batched_dataset(sequence_length=500, batch_size=256)
        """

        instr_seq = tf.data.Dataset.from_tensor_slices(
            self.__instr_as_int__ if not reduced else self.__reduced_instr_as_int__)\
            .apply(sliding_window_batch(window_size=sequence_length+1, stride=stride))
            # .batch(sequence_length+1, drop_remainder=True)#.shuffle(buffer_size)
        
        # NOTE: won't bother adding the 'reduced' case here because this case won't be used.
        instr_seq_as_frames = tf.data.Dataset.from_tensor_slices(self.instr_idx_to_frames(\
            self.__instr_as_int__)).batch(sequence_length+1, drop_remainder=True)

        split_input_target = lambda x: (x[:-1], x[1:])
        input_instr = instr_seq.map(lambda x: x[:-1])
        output_instr = instr_seq_as_frames.map(lambda x: x[1:])

        instr_dataset =\
            instr_seq.map(split_input_target).shuffle(buffer_size)\
                .batch(batch_size, drop_remainder=True) if not output_frames else\
            tf.data.Dataset.zip((input_instr, output_instr)).shuffle(buffer_size)\
            .batch(batch_size, drop_remainder=True)
        
        return self.__split_dataset__(instr_dataset, ratio)


    def get_mel_to_instr_batched_dataset(self, sequence_length=(96), batch_size=64,
            buffer_size=10000, reduced=False, ratio=[.8,.1], stride=5):
        """
        Get the original frames expressed as their respective integer index according to the
        unique frames vocabulary. Note that this returns 2 arrays, the first one containing the
        indices corresponding to the frames of the original melody, and the second one containing
        the indices corresponding to the frames of the original instrumental.
        @param sequence_length: an integer indicating the window length of eacch sequence we feed
                into the model when learning. Deafult 96.
        @param batch_size: in integer that determines the number of sequecences we want to have in
                each batch that we'll feed to the model. Default 64.
        @param buffer_size: in integer used when shuffling the data. Default 10000.
        @param reduced: a boolean determining if we want the dataset to contain indices into the
                vocabulary of the original frames or indices into a vocabulary of the reduced
                frames. Reduced frames are frames that only contain information about the activated
                notes, regardless of what octaves they belong to. Reduced frames from the melody
                vocabular are of length 12, i.e they only contain information about the activation
                status of the 12 musical notes (C, C#, D, D#, E, F, F#, G, G#, A, A# and B). Whereas
                reduced frames from the instrumental vocabular are of length 24, i.e, on octave with
                exactly one activated note (the root note) and the other one with the activation
                status of any remaining note in the original frame. Default False.
                NOTE: the melody will always be reduced.
        @param ratio: a list of 2 float numbers both between 0 and 1. The first number indicate the
                percentage of the final number of batches that we want to allocate as training data.
                The second is the percentage of total batch number to allocate as dev/validation
                data. The remaining will be allocated to the test data.
        @param stride: an integer. This represents the stride to be used when generating the
                sequences. Indeed, the sequences are going to be generated by taking each
                consecutive `sequence_length` frames and skipping `stride` frames from one sequence
                to the other. The stride has to be greater than 0. Defatul 5.

        @return a 3-tuple where each element represents respectively the training dataset, the
                dev/validation dataset and the test dataset. Each dataset contains 2 sequences: the
                first sequence corresponds to the input we would feed to the model and the second
                sequence corresponds to the output we expect the model to predict (true label). The
                output sequence is simply the input sequence minus its first frame and plus the
                frame that should come right after the last frame in the input sequence.

        Example Usage:
        # Assuming you declared a MidiData object like so
        data = MidiData('path_to_tracks_info.csv', open_files=['file1.mid', 'file2.mid'])
        mel_to_instr_dataset =
            data.get_mel_to_instr_batched_dataset(sequence_length=500, batch_size=256)
        """
        # See NOTE(1)
        mel_seq = tf.data.Dataset.from_tensor_slices(self.__reduced_mel_as_int__)\
            .apply(sliding_window_batch(window_size=sequence_length+1, stride=stride))
            # .batch(sequence_length+1, drop_remainder=True)
        instr_seq = tf.data.Dataset.from_tensor_slices(
            self.__instr_as_int__ if not reduced else self.__reduced_instr_as_int__)\
            .apply(sliding_window_batch(window_size=sequence_length+1, stride=stride))
            # .batch(sequence_length+1, drop_remainder=True)

        mel_dataset = mel_seq.map(lambda x: x[:-1])
        instr_dataset = instr_seq.map(lambda x: x[1:])

        dataset = tf.data.Dataset.zip((mel_dataset, instr_dataset)).shuffle(buffer_size)\
            .batch(batch_size, drop_remainder=True)

        return self.__split_dataset__(dataset, ratio)


    def export_model_output_to_midi(self, output, midi_file, from_frames=False):
        """
        Create a MIDI file from the output from a prediction model.
        Note that this method expects that the model output will be an array of integers. Those
        integers are supposed to be each mapped to frames from the vocabulary.

        Example Usage:
        # Assuming you declared a MidiData object like so
        data = MidiData('path_to_tracks_info.csv', open_files=['file1.mid', 'file2.mid'])
        model = ...
        predictions = []
        for i in range(...):
            ...
            predictions.append(<model_prediction>)
            ...
        data.export_model_output_to_midi(predictions, 'generated.mid')
        """

        midi_file = self.__force_midi_extension__(midi_file)
        frames = output if from_frames else self.instr_idx_to_frames(output)
        __Track__.toFile(
            frames, file=midi_file, smoothing=self.__smoothing__)    

    def mel_idx_to_frames(self, idx_array):
        """
        Get the list of frames corresponding to the melody from the list of indices returned by the
        model.
        Example Usage:
        data = MidiData(...)
        predictions = []
        model = ...
        for i in range(...):
            ...
            predictions.append(<model_prediction>)
            ...
        predicted_frames = data.mel_idx_to_frames(predictions)
        """

        # See NOTE(1)
        map_idx_vector_to_frames = lambda x: np.array(
            literal_eval(self.__reduced_mel_vocab__[int(x)]))
        map_all_idx_to_frames = lambda x: np.array(
            list(map(map_idx_vector_to_frames, x)))
        return np.apply_along_axis(map_all_idx_to_frames, -1, idx_array)

    def instr_idx_to_frames(self, idx_array):
        """
        Get the list of frames corresponding to the instrumental from the list of indices returned
        by the model.
        Example Usage:
        data = MidiData(...)
        predictions = []
        model = ...
        for i in range(...):
            ...
            predictions.append(<model_prediction>)
            ...
        predicted_frames = data.instr_idx_to_frames(predictions)
        """

        map_idx_vector_to_frames = lambda x: np.array(literal_eval(self.__instr_vocab__[int(x)]))
        map_all_idx_to_frames = lambda x: np.array(
            list(map(map_idx_vector_to_frames, x)))
        return np.apply_along_axis(map_all_idx_to_frames, -1, idx_array)

    def extract_notes_only(self, frames, with_root=False):
        """
        Extract from a frame of set of frames the notes information only without caring about the
        different octaves they were played in. Just the existence or non-existence of the 12 notes.
        @param frames: nd-array where the last dimension is the size of a frame.
        @param with_root(optional): a boolean representing whether or not we want to capture the
                'root note' information. If False, the new frames will only be 12 notes long; if
                True, the new frames will be 24 notes long, with the first 12 notes reserved for the
                root note and the remaining 12 for any other existing notes. Default False.
        @return a new nd-array of a similar shape as the input `frames` except for the last
                dimension where the length of the frame will be either 12 or 24 depending on whether
                `with_root` was False or True respectively
        """
        def extract(frame):
            NBR_NOTES = 12
            notes = np.nonzero(frame)[0]%NBR_NOTES
            unique_notes = np.unique(notes)
            new_frame = np.zeros((NBR_NOTES if not with_root else NBR_NOTES*2),dtype='int')
            if len(notes) == 0: return new_frame
            if with_root:
                new_frame[notes[0]] = 1
                for note in unique_notes:
                    if note == notes[0]: continue
                    new_frame[note+NBR_NOTES] = 1
            else:
                for note in unique_notes:
                    new_frame[note] = 1
            return new_frame
        new_frames = np.apply_along_axis(extract, -1, frames)
        return new_frames

    def get_new_melody_as_int(self):
        """
        TODO: missing documentation
        See NOTE(1)
        """
        return self.__reduced_new_melody_as_int__

    def load_new_melody(self, midi_path):
        """
        TODO: missing documentation
        See NOTE(1)
        """
        midi_path = self.__force_midi_extension__(midi_path)
        song = __Song__(midi_path, padding=False).parse()
        midi_file = midi_path.split('/')[-1]
        midi_file = self.__remove_midi_extension__(midi_file)
        if not midi_file in self.__tracks_info__.keys():
            print('No track info found for {} in tracks file. Skipping.'.format(midi_file))
            continue
        info = self.__tracks_info__[self.__remove_midi_extension__(midi_file)]
        self.__new_melody__ = song.instruments[info[MELODY_TRACK]].transpose_frames(0)
        count_frames += len(self.__new_melody__)
        if self.__smoothing__ > 1:
            self.__new_melody__ = np.array(self.__new_melody__[::self.__smoothing__])
        
        self.__reduced_new_melody__ = self.extract_notes_only(self.__new_melody__)
        self.__reduced_new_melody_as_int__ = np.apply_along_axis(
            lambda x: self.__reduced_mel_vocab__.index(str(tuple(x))), -1,\
            self.__reduced_new_melody__)


    # ******************END PUBLIC/IMPORTANT METHODS***************************

    def __generate_reduced_melody_vocab__(self):
        tuple_vocab = product([0,1], repeat=12)
        self.__np_reduced_mel_vocab__ = np.array(tuple_vocab)
        self.__reduced_mel_vocab__ = list(map(lambda x: str(x), tuple_vocab))

    def __probs_to_k_frame__(self, notes_probs):
        # notes_probs = tf.expand_dims(notes_probs, 0)
        # print(notes_probs)
        # print(notes_probs.shape)
        # print(self.__np_instr_vocab__.shape)
        # l = notes_probs*self.__np_instr_vocab__
        # print(l.shape)
        # print(self.__instr_vocab_nonzero_count__.shape)
        k = self.__probs_to_frames_k
        threshold = self.__probs_to_frames_threshold
        # print(self.__np_instr_vocab__.shape)
        activated = deepcopy(notes_probs)
        activated[activated < threshold] = 0
        logits =\
            np.sum(activated*self.__np_instr_vocab__, axis=-1)/self.__instr_vocab_nonzero_count__
        new_probs, idx = tf.math.top_k(logits, k=k)
        frames = tf.gather(self.__np_instr_vocab__, idx)
        id = randint(0, k-1)
        # print(idx)
        # print(self.__vocab__[1][idx[0]])

        frame = frames[id]
        # print(frame)
        # print(self.get_frame_index(frame))
        return frame

    def __split_dataset__(self, dataset, ratio):
        size = tf.data.experimental.cardinality(dataset)
        def get_split(size, ratio):
            size = size.numpy().astype(int)
            ratio = ratio.numpy()
            train = int(ratio[0]*size)
            dev = int(ratio[1]*size)
            test = size - (train+dev)
            assert (ratio[0] == 0 or train != 0) and (ratio[1] == 0 or dev != 0) and\
                (ratio[0]+ratio[1] == 1 or test != 0), 'Cannot split dataset of size ' + str(size)\
                + ' into train/dev/test ratio ' + str(list(ratio)) +'.\nTrain/Dev/Test sizes were '\
                + f'[{train},{dev},{test}]'
            print(f'Splitting {size} batches into {train}:{dev}:{test} Train:Dev:Test.')
            return train,dev,test
        train,dev,test = tf.py_function(get_split, [size, ratio],[tf.int64, tf.int64, tf.int64])
        train_ds = dataset.take(train)
        other_ds = dataset.skip(train)
        dev_ds = other_ds.take(dev)
        test_ds = other_ds.skip(dev)
        return train_ds, dev_ds, test_ds

    def __smooth_frames__(self):
        if self.__smoothing__ == 1: return
        new_songs = []
        print('Smoothing frames...')
        count = 0
        for frames in self.__songs__:
            new_frames = frames[::self.__smoothing__]
            new_songs.append(new_frames)
            count += len(new_frames)
        
        self.__songs__ = np.array(new_songs)
        print('Frames smoothed down to {} frames.'.format(count))

    def __read_tracks_info_file__(self, tracks_info_file):
        self.__tracks_info__ = {}
        with open(tracks_info_file, 'r') as lines:
            for line in lines.readlines():
                song, melody, instrumental = line.split(',')
                instrumental = ''.join([i for i in instrumental if i.isdigit()])
                song = self.__remove_midi_extension__(song)
                if not melody.strip().isdigit() or not instrumental.strip().isdigit(): continue
                self.__tracks_info__[song] = {
                    MELODY_TRACK: int(melody.strip()), 
                    INSTRUMENT_TRACK: int(instrumental.strip())}

    def __remove_midi_extension__(self, midi_file):
        if midi_file[-4:] == '.mid': return midi_file[:-4]
        return midi_file

    def __force_midi_extension__(self, midi_file):
        if midi_file[-4:] != '.mid': return midi_file + '.mid'
        return midi_file

    def __open_from_folder__(self, folder):
        if folder[-1] != '/': folder += '/'
        self.__open_files__([folder+f for f in listdir(folder) if f[-4:] == '.mid'])

    def __open_files__(self, midi_paths):
        self.__songs__ = None
        count_frames = 0
        print('Opening {} midi file(s)...'.format(len(midi_paths)))
        for index,midi_path in enumerate(midi_paths):
            midi_path = self.__force_midi_extension__(midi_path)
            song = __Song__(midi_path).parse()
            midi_file = midi_path.split('/')[-1]
            midi_file = self.__remove_midi_extension__(midi_file)
            if not midi_file in self.__tracks_info__.keys(): 
                print('No track info found for {} in tracks file. Skipping.'.format(midi_file))
                continue
            info = self.__tracks_info__[self.__remove_midi_extension__(midi_file)]
            progress_msg = 'Parsing "{}"'.format(midi_file)
            transposes = self.__transpose__
            nbr_transposes = len(transposes)
            for t_index, transpose in enumerate(transposes):
                melody_frames = song.instruments[info[MELODY_TRACK]].transpose_frames(transpose)
                instrumental_frames =\
                    song.instruments[info[INSTRUMENT_TRACK]].transpose_frames(transpose)
                count_frames += len(melody_frames) + len(instrumental_frames)
                new_song_frames = [melody_frames, instrumental_frames]
                if type(self.__songs__) == type(None):
                    self.__songs__ = np.asarray(new_song_frames)
                else: self.__songs__ = np.concatenate((self.__songs__, new_song_frames), axis=1)
                util.showprogress((t_index+1)/(nbr_transposes*len(midi_paths))\
                    + (index)/len(midi_paths), message=progress_msg, sub=True)
            util.showprogress((index+1)/len(midi_paths), message=progress_msg)
        self.__songs__ = np.asarray(self.__songs__)
        print('Loaded {} frames.'.format(count_frames))               


class __Track__():
    def __init__(self, track, binary_velocity=True):
        self.track = track
        self.name = track.name
        self.binary_velocity = binary_velocity
        self.tempo = None
        self.frames = []
        self.current_frame = []
        self.channel = None
    
    def parse(self):
        for message in self.track:
            msg_type = message.type
            if msg_type == 'note_on' or msg_type == 'note_off':
                delta_time,channel,note,velocity = self.__get_note_info__(message)
                self.__maybe_update_frames__(delta_time)
                if msg_type == 'note_off': velocity = 0
                self.__maybe_set_channel__(channel+1)
                self.__maybe_init_next_frames__()
                if velocity == 0 and False:
                    self.frames[-1][note] = 0
                else:
                    
                    self.__set_note__(note, velocity)
            if msg_type == 'set_tempo':
                if self.tempo != None: continue
                MPQN = message.tempo 
                self.tempo = round(60000000 / MPQN, 2)
        if len(self.current_frame) > 0: self.__maybe_update_frames__()
        self.frames = np.asarray(self.frames)
        return self

    def transpose_frames(self, amount):
        transposed = np.roll(self.frames, amount, axis=1)
        if amount > 0: transposed[:,:amount] = 0
        elif amount < 0: transposed[:, amount:] = 0
        return transposed


    @staticmethod
    def toFile(frames, file='', smoothing=1):
        midi_file = MidiFile()
        track = MidiTrack()
        midi_file.tracks.append(track)
        if file == '': file = 'generated.mid'

        current_frame = np.zeros(NOTES_RANGE, dtype=int)
        delta_time = 0
        print(f'Exporting frames to file {file}...')
        for i,frame in enumerate(frames):
            time_increase = 5
            idx = np.nonzero(current_frame - frame)
            for note in idx[0]:
                msg = 'note_on' if frame[note] > 0 else 'note_off'
                track.append(Message(msg, note=note, velocity=70, time=delta_time))
                delta_time = 0
            delta_time += time_increase*smoothing
            current_frame = frame
            util.showprogress((i+1)/len(frames))
        print('Export successful!')
        midi_file.save(file)

    def __maybe_set_channel__(self, channel:int):
        """
        Set the channel if it hasn't been set yet (if it's currently `None`).
        """
        if self.channel == None: self.channel = channel

    def __set_note__(self, note:int, velocity:int):
        vel = velocity
        if self.binary_velocity:
            vel = int(vel > 0)
        self.current_frame[note] = vel
        if vel > 0 and (len(self.frames) > 0 and self.frames[-1][note] > 0):
            self.frames[-1][note] = 0

    def __get_note_info__(self, message):
        return message.time, message.channel, message.note, message.velocity

    def __get_empty_frame__(self):
        return np.zeros(NOTES_RANGE,dtype=int)

    def __maybe_init_next_frames__(self):
        """
        This method is called is called to initialize a new frame that will then be filled according
        to what notes are played at that new frame. This will be stored in `current_frame`. However,
        before storing the new frame, it checks that we are not still working on the said
        `current_frame`. If `current_frame` has elements in it, then it's still being used. When
        it's no longer being used (when we have competely filled that frame), it will be set back to
        an empty array (through a call to `update_frames()`). That's how this method knows whether
        or not to actually re-initialize `current_frame` or not.
        """
        if len(self.current_frame) == 0:
            self.current_frame =\
                    deepcopy(self.frames[-1]) if len(self.frames) > 0\
                        else self.__get_empty_frame__()
    
    def __maybe_update_frames__(self, length:int=1):
        """
        Update the frames by adding the `current_frame` to the list of `frames`. It updates by
        filling the `length` rows with copies of `current_frame`. It will only update if
        either no `length` was given (defaulting to 1) or the provided `length` is greater than 0.
        This `length` is intended to be the delta time of note on/off events. For events that are
        triggered at the same time as the last seen event, their delta time will be 0, so nothing
        will be done. The frames will only be updated when we move onto an event with a delta time
        greater than 0.
        
        If the current frame is empty, then it uses a frame filled with 0's everywhere.
        """
        if length > 0:
            current_frame = self.current_frame if len(self.current_frame) > 0 else\
                            self.__get_empty_frame__()
            for i in range(length):
                self.frames.append(deepcopy(current_frame))
            self.current_frame = []


class __Song__():
    def __init__(self, midi_file, binary_velocity=True, padding=True):
        self.midi_file = midi_file
        self.__padding__ = padding
        self.mido = MidiFile(midi_file)
        self.binary_velocity = binary_velocity
        self.tempo = None
        self.instruments = []

    def parse(self):
        for midi_track in self.mido.tracks:
            track = __Track__(midi_track, self.binary_velocity)
            track.parse()
            self.__add_instrument__(track)
        if self.__padding__: self.__pad_frames__()
        if self.tempo == None: self.tempo = DEFAULT_TEMPO
        return self

    def __add_instrument__(self, track:__Track__):
        if self.tempo == None:
            self.tempo = track.tempo
        if len(track.frames) > 0:
            # Only add the track as an instrument if it has frames
            # print('length is ', len(track.frames))
            self.instruments.append(track)
    
    def __pad_frames__(self):
        length = max([len(instrument.frames) for instrument in self.instruments])
        for i,instrument in enumerate(self.instruments):
            frames = instrument.frames
            padded_frames = np.zeros((length, frames.shape[1]), dtype=int)
            padded_frames[:frames.shape[0]] = frames
            self.instruments[i].frames = padded_frames
    
    def toDimp(self, name:str='', relative_path=''):
        name = name if name != '' else self.midi_file[:-4] + FILE_EXTENSION
        if relative_path != '':
            relative_path += '/' if relative_path[-1] != '/' else ''
        
        path = relative_path+name
        with open(path, 'w') as f:
            f.write(f'Tempo:{self.tempo}\n')
            f.write(f'NbrInstruments:{len(self.instruments)}\n')
            for i,instrument in enumerate(self.instruments):
                f.write(f'Instrument#{i+1}{{\n')
                f.write(f'Name:"{instrument.name}"\n')
                f.write(f'Channel:{instrument.channel}\n')
                f.write(f'NrbFrames:{len(instrument.frames)}\n')
                f.write(f'Frames{{\n')
                for frame in instrument.frames:
                    f.write(str(frame.tolist())+'\n')
                f.write(f'}}EndFrames\n')
                f.write(f'}}EndInstrument#{i+1}\n')
            f.close()
        return self



if __name__ == '__main__':
    debug = True

    if debug: tf.enable_eager_execution()

    mididata = MidiData('midis/tracks.csv', open_files=['midis/modern tidings.mid'],
        smoothing=32, transpose=[-3,-2,-1,0,1,2,3])

    # frame = np.random.rand(3,128)
    # frame[frame >= .9] = 1
    # frame[frame < .9] = 0
    # print(frame)
    # print(mididata.extract_notes_only(frame, with_root=True))

    frame = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .9, 0.1, 0, 0, 0, 0, 0, 0, .89, 0, 0, 0, .9, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    frame = np.array(p)
    # f = mididata.compute_vocab_chopin_distribution([[[10, 52], [23, 69]], [[2,85], [35,94]]])
    f = mididata.compute_vocab_chopin_distribution([25])
    # print(list(f.tolist()))
    # print(f[0][6])
    print(np.argmax(f, axis=-1))


    # probs = np.random.rand(128)
    # frames = mididata.probs_to_frames(probs, k=1)
    # print(list(frames))
    # print(mididata.get_frame_index(frames))
    
    # ---Debugging---
    if debug or False:
        data, ddata, tdata  = mididata.get_instr_batched_dataset(sequence_length=64, batch_size=2,\
            smoothing=16, ratio=[.6,.2])
        # print(type(data))
        for i,d in enumerate(data):
            # print(d)
            # if i == 0: print(d)
            mididata.export_model_output_to_midi(d[0].numpy()[0],\
                midi_file=f'generated0{i}.mid')
            mididata.export_model_output_to_midi(d[1].numpy()[0],\
                midi_file=f'generated1{i}.mid')
            # __Track__.toFile(self.instr_idx_to_frames(d[0].numpy()[1]),
            #    file=f'generated{i}.mid', smoothing=self.__smoothing__)
            if i == 9: break
        # for i,d in enumerate(ddata):
        #     # print(d)
        #     # if i == 0: print(d)
        #     mididata.export_model_output_to_midi(d[0].numpy()[0],\
        #         midi_file=f'train_generated0{i}.mid')
        #     mididata.export_model_output_to_midi(d[1].numpy()[0],\
        #         midi_file=f'train_generated1{i}.mid')
        #     # __Track__.toFile(self.instr_idx_to_frames(d[0].numpy()[1]),
        #     #    file=f'generated{i}.mid', smoothing=self.__smoothing__)
        #     if i == 9: break
    # print(data.get_vocabs_sizes())