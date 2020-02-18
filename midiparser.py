from ast import literal_eval
from copy import deepcopy
from mido import Message, MidiFile, MidiTrack
from os import listdir
from util import showprogress
import csv
import numpy as np
import tensorflow as tf

# tf.enable_eager_execution()

FILE_EXTENSION = '.dimp' # Digital Music Representation
NOTES_RANGE = 128 # a.k.a the width of one frame
DEFAULT_TEMPO = 120
DEFAULT_TICKS_PER_SECOND = 96
MELODY_TRACK = 'melody'
INSTRUMENT_TRACK = 'instrument'

class MidiData():
    # *********PUBLIC/IMPORTANT METHODS************
    def __init__(self, tracks_info_file, open_files=None, open_from_folder=None, smoothing=1, transpose=[0]):
        """
        Creates an instance of MidiData. This is called automatically when you call 'MidiData()',
        see example usage below.
        Arguments:
            * tracks_info: the path to the location of the cvs file containing information about the
                midi files
            * open_files (recommanded): a list of the files to open in this MidiData object
            * open_from_folder: use this argument if you want to open all midi files inside of
                a specific folder without listing them all.
                NOTE: Only one of 'open_files' or 'open_from_folder' must be used.
            * smoothing (optional): an integer greater or equal to 1. It determines how much
                granularity we want the frames to have. A smoothing of 1 will have the frames as
                granular as they can be (96 frames per beat). A greater smoothing value means less
                granularity (useful for only retaining the most important note/chord changes and
                ignoring the potential small variations, especially when there are many such small
                variations).
            * transpose (optional): an array of at least one integer. This will decide if we want to
                transpose the midi data. If this argument is used, it should usually take a value
                like this [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]. With an argument like this one,
                the returned data would contain, in addition to the original data in the midi file,
                4 copies of the same data transposed respectively 4, 3, 2 and 1 times downward, then
                7 copies of the original data transposed respectively 1...7 times upward.
        
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
        if open_from_folder: self.__open_from_folder__(open_from_folder)
        elif open_files: self.__open_files__(open_files)
        self.__smooth_frames__()
        print('Extracting vocabulary...')
        frames2set = lambda x: str(tuple(x))
        mel_as_tuple = np.apply_along_axis(frames2set, 1, self.__songs__[0]).tolist()
        instr_as_tuple = np.apply_along_axis(frames2set, 1, self.__songs__[1]).tolist()
        mel_vocab = list(set(mel_as_tuple))
        instr_vocab = list(set(instr_as_tuple))
        self.__vocab__ = (mel_vocab, instr_vocab)
        print('Vocabularty size (melody: {}, instr: {})'.format(len(mel_vocab), len(instr_vocab)))
        print('Generating frame-to-int maps...')
        mel2idx = {frame:i for i,frame in enumerate(mel_vocab)}
        instr2idx = {frame:i for i,frame in enumerate(instr_vocab)}
        self.__mel_as_int__ = list(map(lambda x: mel2idx[x], mel_as_tuple))
        self.__instr_as_int__ = list(map(lambda x: instr2idx[x], instr_as_tuple))
        print('MidiData object ready!')

        # data = self.get_instr_batched_dataset(sequence_length=96, batch_size=2)
        # # print(type(data))
        # for i,d in enumerate(data):
        #     # print(d)
        #     if i == 0: print(d)
        #     __Track__.toFile(self.instr_idx_to_frames(d[0].numpy()[1]), file=f'generated{i}.mid', smoothing=self.__smoothing__)
        #     if i == 9: break

    def get_vocabs(self):
        """
        Get the unique frames from the loaded data, aka the vocabulary. Note that this returns 2
        sets of frames, the first one corresponding to the melody and the second one to the
        instrumental.

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
        return self.__vocab__[0], self.__vocab__[1]

    def get_vocabs_sizes(self):
        """
        Get the number of unique frames, aka the size of the vocabulary. Note that this returns 2
        sizes, the first one corresponding to the size of unique frames in the melody, and the
        second one corresponding to the size of frames in the instrumental.

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
        return len(self.__vocab__[0]), len(self.__vocab__[1])

    def get_frames_as_int(self):
        """
        Get the original melody and instrumental where each frame is replaced by its integer
        representation according to the vocabulary.

        Example Usage:
        # Assuming you declared a MidiData object
        data = MidiData(...)
        mel_as_int, instr_as_int = data.get_frames_as_int()
        """

        return self.__mel_as_int__, self.__instr_as_int__

    def get_instr_batched_dataset(self, sequence_length=(384*2), batch_size=128, buffer_size=10000):
        """
        Get the original frames expressed as their respective integer index according to the
        unique frames vocabulary. Note that this returns 2 arrays, the first one containing the
        indices corresponding to the frames of the original melody, and the second one containing
        the indices corresponding to the frames of the original instrumental.

        Example Usage:
        # Assuming you declared a MidiData object like so
        data = MidiData('path_to_tracks_info.csv', open_files=['file1.mid', 'file2.mid'])
        instr_dataset = data.get_instr_batched_dataset(sequence_length=500, batch_size=256)
        """
        instr_seq = tf.data.Dataset.from_tensor_slices(self.__instr_as_int__).batch(sequence_length+1, drop_remainder=True)

        split_input_target = lambda x: (x[:-1], x[1:])

        instr_dataset = instr_seq.map(split_input_target).shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        return instr_dataset


    def get_mel_to_instr_batched_dataset(self, sequence_length=(384*2), batch_size=128, buffer_size=10000):
        """
        Get the original frames expressed as their respective integer index according to the
        unique frames vocabulary. Note that this returns 2 arrays, the first one containing the
        indices corresponding to the frames of the original melody, and the second one containing
        the indices corresponding to the frames of the original instrumental.

        Example Usage:
        # Assuming you declared a MidiData object like so
        data = MidiData('path_to_tracks_info.csv', open_files=['file1.mid', 'file2.mid'])
        mel_to_instr_dataset = data.get_mel_to_instr_batched_dataset(sequence_length=500, batch_size=256)
        """
        mel_seq = tf.data.Dataset.from_tensor_slices(self.__mel_as_int__).batch(sequence_length+1, drop_remainder=True)
        instr_seq = tf.data.Dataset.from_tensor_slices(self.__instr_as_int__).batch(sequence_length+1, drop_remainder=True)

        mel_dataset = mel_seq.map(lambda x: x[:-1])
        instr_dataset = instr_seq.map(lambda x: x[1:])

        dataset = tf.data.Dataset.zip((mel_dataset, instr_dataset)).shuffle(buffer_size).batch(batch_size, drop_remainder=True)

        return dataset


    def export_model_output_to_midi(self, output, midi_file):
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
        frames = self.instr_idx_to_frames(output)
        __Track__.toFile(frames, file=midi_file, smoothing=self.__smoothing__)    

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

        map_idx_to_frames = lambda x: np.array(literal_eval(self.__vocab__[0][x]))
        return list(map(map_idx_to_frames, idx_array))

    
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

        # print(idx_array)
        map_idx_to_frames = lambda x: np.array(literal_eval(self.__vocab__[1][x]))
        return list(map(map_idx_to_frames, idx_array))



    # ******************END PUBLIC/IMPORTANT METHODS***************************

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
        print('Frames smoothed down to {}'.format(count))

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
        # print(folder)
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
            # print(midi_file)
            if not midi_file in self.__tracks_info__.keys(): 
                print('No track info found for {} in tracks file. Skipping.'.format(midi_file))
                continue
            info = self.__tracks_info__[self.__remove_midi_extension__(midi_file)]
            progress_msg = 'Parsing "{}"'.format(midi_file)
            nbr_transposes = len(self.__transpose__)
            for t_index, transpose in enumerate(self.__transpose__):
                melody_frames = song.instruments[info[MELODY_TRACK]].transpose_frames(transpose)
                instrumental_frames = song.instruments[info[INSTRUMENT_TRACK]].transpose_frames(transpose)
                count_frames += len(melody_frames) + len(instrumental_frames)
                new_song_frames = [melody_frames, instrumental_frames]
                if type(self.__songs__) == type(None): self.__songs__ = np.asarray(new_song_frames)
                else: self.__songs__ = np.concatenate((self.__songs__, new_song_frames), axis=1)
                showprogress((t_index+1)/(nbr_transposes*len(midi_paths)) + (index)/len(midi_paths), message=progress_msg, sub=True)
            showprogress((index+1)/len(midi_paths), message=progress_msg)
        self.__songs__ = np.asarray(self.__songs__)
        # print(self.__songs__)
        # print(self.__songs__.shape)
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
                # print(msg_type, velocity)
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
        # print(self.frames)
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
            # frame = frame.tolist()
            # print(current_frame, type(current_frame))
            # print(frame, type(frame))
            # for repeat in range(smoothing):
            idx = np.nonzero(current_frame - frame)
            delta_time += 5*smoothing
            for note in idx[0]:
                msg = 'note_on' if frame[note] > 0 else 'note_off'
                track.append(Message(msg, note=note, velocity=70, time=delta_time))
                delta_time = 0
            current_frame = frame
            showprogress((i+1)/len(frames))
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
    def __init__(self, midi_file, binary_velocity=True):
        self.midi_file = midi_file
        self.mido = MidiFile(midi_file)
        self.binary_velocity = binary_velocity
        self.tempo = None
        self.instruments = []

    def parse(self):
        for midi_track in self.mido.tracks:
            track = __Track__(midi_track, self.binary_velocity)
            track.parse()
            self.__add_instrument__(track)
        self.__pad_frames__()
        if self.tempo == None: self.tempo = DEFAULT_TEMPO
        return self

    def __add_instrument__(self, track:__Track__):
        if self.tempo == None:
            self.tempo = track.tempo
        if len(track.frames) > 0:
            # Only add the track as an instrument if it has frames
            self.instruments.append(track)
    
    def __pad_frames__(self):
        length = max([len(instrument.frames) for instrument in self.instruments])
        # print(length)
        for i,instrument in enumerate(self.instruments):
            # print(instrument.frames.shape)
            frames = instrument.frames
            # print(type(frames))
            padded_frames = np.zeros((length, frames.shape[1]), dtype=int)
            padded_frames[:frames.shape[0]] = frames
            self.instruments[i].frames = padded_frames
            # print(padded_frames.shape)
    
    def toDimp(self, name:str='', relative_path=''):
        name = name if name != '' else self.midi_file[:-4] + FILE_EXTENSION
        if relative_path != '':
            relative_path += '/' if relative_path[-1] != '/' else ''
        
        path = relative_path+name
        with open(path, 'w') as f:
            f.write(f'Tempo:{self.tempo}\n')
            # f.write(f'FramesPerTick:{self.samples_per_tick}\n')
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
    data = MidiData('midis/tracks.csv', open_files=['midis/white christmas.mid'], smoothing=32)
    print(data.get_vocabs_sizes())