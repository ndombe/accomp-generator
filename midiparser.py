from mido import Message, MidiFile, MidiTrack
import collections
import copy
import csv
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

FILE_EXTENSION = '.dimp' # Digital Music Representation
NOTES_RANGE = 128 # a.k.a the width of one frame
DEFAULT_TEMPO = 120
DEFAULT_TICKS_PER_SECOND = 96
MELODY_TRACK = 'melody'
INSTRUMENT_TRACK = 'instrument'
TRANSPOSE_DOWN_LIMIT = -4
TRANSPOSE_UP_LIMIT = 7
EMPTY_FRAME = np.zeros(NOTES_RANGE, dtype=int)


class ShowStepProgress():
    def __init__(self):
        self.__processes__ = collections.OrderedDict()
    
    def step(self, process_id, progress, total, start_msg:str=None, progress_msg:str='', finish_msg:str=None):
        process = (process_id, total)
        if len(self.__processes__.keys()) == 0:
            if type(start_msg) == type(None): start_msg = 'Starting process ' + process_id
            print(start_msg)


        self.__processes__[process] = progress

        percent = 0
        obsolete_processes = []
        for i,(process_id, total) in enumerate(self.__processes__):
            progress = self.__processes__[(process_id, total)]
            percent += progress / (total * np.prod([t for idx,(id,t) in enumerate(self.__processes__) if idx < i]))
            if process == (process_id, total):
                obsolete_processes = list(self.__processes__)[i+1:]
                break
        for p in obsolete_processes:
            del self.__processes__[p]


        self.__showprogress__(percent, progress_msg)

        if percent == 1 and len(self.__processes__.keys()) == 1:
            if type(finish_msg) == type(None): finish_msg = 'Finished process ' + process_id
            print('')
            print(finish_msg)
            del self.__processes__[process]

    def __showprogress__(self, percentage, message=''):
        length = 30
        done = int(length*percentage)
        arrow = 0
        if done < length:
            done -= 1 if done > 0 else 0
            arrow = 1
        left = length - done
        sys.stdout.write('\r[{}{}{}] {}% {}'.format(
            '='*done, '>'*arrow, '.'*left, int(round(percentage, 2)*100), message))
__progress__ = ShowStepProgress()

class MidiData():
    def __init__(self, tracks_info_file, open_from_folder=None, open_files=None):
        assert open_from_folder == None or open_files == None
        self.__read_tracks_info_file__(tracks_info_file)
        self.__tracks_vocab__ = {
            MELODY_TRACK: {tuple(EMPTY_FRAME): 0},
            INSTRUMENT_TRACK: {tuple(EMPTY_FRAME): 0}
        }
        if open_from_folder: self.__open_from_folder__(open_from_folder)
        elif open_files: self.__open_files__(open_files)
        # self.__unique_frames__ = (np.unique(self.__songs__[0], axis=0), np.unique(self.__songs__[1], axis=0))
        # frames2set = lambda x: str(x)
        # unique_mel = np.apply_along_axis(frames2set, 1, self.__songs__[0])
        # unique_instr = np.apply_along_axis(frames2set, 1, self.__songs__[1])
        # # print(unique_mel.shape, unique_instr.shape)
        # self.__unique_frames__ = np.asarray([np.asarray(list(set(unique_mel))),
        #                             np.asarray(list(set(unique_instr)))])
        # # self.__mel_frame2idx__ = {str(frame):i for i,frame in enumerate(self.__songs__[0])}
        # self.__instr_frame2idx__ = {str(frame):i for i,frame in enumerate(self.__songs__[1])}

    def get_unique_frames(self):
        """
        Get the unique frames from the loaded data, aka the vocabulary. Note that this returns 2
        sets of frames, the first one corresponding to the melody and the second one to the
        instrumental.

        Example Usage:
        data = MidiData('path_to_tracks_info.csv', open_files=['file1.mid', 'file2.mid'])
        unique_mel, unique_instr = data.get_unique_frames()

        This will return a tuple where the first element is a numpy array containing the unique
        frames for the melody, and the second element is a numpy array with the unique frames for
        the instrumental.

        For instance, if the list of frames for the instrumental look something like this:
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
        return self.__tracks_vocab__[MELODY_TRACK], self.__tracks_vocab__[INSTRUMENT_TRACK]

    def unique_frames_sizes(self):
        """
        Get the number of unique frames, aka the size of the vocabulary. Note that this returns 2
        sizes, the first one corresponding to the size of unique frames in the melody, and the
        second one corresponding to the size of frames in the instrumental.

        Example Usage:
        data = MidiData('path_to_tracks_info.csv', open_files=['file1.mid', 'file2.mid'])
        unique_mel_size, unique_instr_size = data.unique_frames_sizes()

        This is the same thing as calling "get_unique_frames()" then getting the length, like so:
        data = MidiData('path_to_tracks_info.csv', open_files=['file1.mid', 'file2.mid'])
        unique_mel, unique_instr = data.get_unique_frames()
        unique_mel_size = len(unique_mel)
        unique_instr_size = len(unique_instr)
        """
        return self.__tracks_vocab__[MELODY_TRACK].shape[0], self.__tracks_vocab__[INSTRUMENT_TRACK].shape[0]

    def get_instr_batch_dataset(self, sequence_length=(384*2), batch_size=128, buffer_size=10000):
        """
        Get the original frames expressed as their respective integer index according to the
        unique frames vocabulary. Note that this returns 2 arrays, the first one containing the
        indices corresponding to the frames of the original melody, and the second one containing
        the indices corresponding to the frames of the original instrumental.

        Example Usage:
        data = MidiData('path_to_tracks_info.csv', open_files=['file1.mid', 'file2.mid'])
        instr_dataset = data.get_instr_batch_dataset(sequence_length=500, batch_size=256)
        """

        map_frame_to_idx = lambda x: self.__instr_frame2idx__[str(x)]
        instr_as_int = np.apply_along_axis(map_frame_to_idx, 1, self.__songs__[1])

        instr_seq = tf.data.Dataset.from_tensor_slices(instr_as_int).batch(sequence_length+1, drop_remainder=True)

        split_input_target = lambda x: x[:-1], x[1:]

        instr_dataset = instr_seq.map(split_input_target).shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        return instr_dataset


    def get_mel_to_instr_batch_dataset(self, sequence_length=(384*2), batch_size=128, buffer_size=10000):
        """
        Get the original frames expressed as their respective integer index according to the
        unique frames vocabulary. Note that this returns 2 arrays, the first one containing the
        indices corresponding to the frames of the original melody, and the second one containing
        the indices corresponding to the frames of the original instrumental.

        Example Usage:
        data = MidiData('path_to_tracks_info.csv', open_files=['file1.mid', 'file2.mid'])
        mel_to_instr_dataset = data.get_mel_to_instr_batch_dataset(sequence_length=500, batch_size=256)
        """

        map_mel_frame_to_idx = lambda x: self.__mel_frame2idx__[str(x)]
        map_instr_frame_to_idx = lambda x: self.__instr_frame2idx__[str(x)]
        mel_as_int = np.apply_along_axis(map_mel_frame_to_idx, 1, self.__songs__[1])
        instr_as_int = np.apply_along_axis(map_instr_frame_to_idx, 1, self.__songs__[1])

        mel_seq = tf.data.Dataset.from_tensor_slices(mel_as_int).batch(sequence_length+1, drop_remainder=True)
        instr_seq = tf.data.Dataset.from_tensor_slices(instr_as_int).batch(sequence_length+1, drop_remainder=True)

        split_input_target = lambda x: x[:-1], x[1:]

        mel_dataset = instr_seq.map(lambda x: x[:-1])
        instr_dataset = instr_seq.map(lambda x: x[1:])

        dataset = tf.data.Dataset.zip((mel_dataset, instr_dataset)).shuffle(buffer_size).batch(batch_size, drop_remainder=True)

        return dataset

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
            predictions.append(model.predict(...))
            ...
        predicted_frames = data.mel_idx_to_frames(predictions)
        """

        map_idx_to_frames = lambda x: self.__tracks_vocab__[MELODY_TRACK][x]
        return np.apply_along_axis(map_idx_to_frames, 0, idx_array)

    
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
            predictions.append(model.predict(...))
            ...
        predicted_frames = data.instr_idx_to_frames(predictions)
        """

        map_idx_to_frames = lambda x: self.__tracks_vocab__[INSTRUMENT_TRACK][x]
        return np.apply_along_axis(map_idx_to_frames, 0, idx_array)


    def __read_tracks_info_file__(self, tracks_info_file):
        self.__tracks_info__ = {}
        with open(tracks_info_file, 'r') as lines:
            for line in lines.readlines():
                song, melody, instrumental = line.split(',')
                instrumental = ''.join([i for i in instrumental if i.isdigit()])
                song = __get_midi_file_name_only__(song)
                if not melody.strip().isdigit() or not instrumental.strip().isdigit(): continue
                self.__tracks_info__[song] = {
                    MELODY_TRACK: int(melody.strip()), 
                    INSTRUMENT_TRACK: int(instrumental.strip())}

    def __open_from_folder__(self, folder):
        if folder[-1] != '/': folder += '/'
        # print(folder)
        self.__open_files__([folder+f for f in os.listdir(folder) if f[-4:] == '.mid'])

    def __open_files__(self, midi_paths):
        self.__songs__ = None
        self.__songs_as_int__ = None
        count_frames = 0
        # print('Opening {} midi file(s).'.format(len(midi_paths)))
        for index,midi_path in enumerate(midi_paths):
            midi_file = __get_midi_file_name_only__(midi_path)
            progress_msg = 'Parsing "{}"'.format(midi_file)
            __progress__.step(
                'openfiles',
                index,
                len(midi_paths),
                start_msg='Opening {} midi file(s).'.format(len(midi_paths)),
                progress_msg=progress_msg)


            midi_path = __get_midi_file_name_with_extension__(midi_path)
            song = __Song__(midi_path, self.__tracks_info__, self.__tracks_vocab__).parse()
            # print(midi_file)
            if not midi_file in self.__tracks_info__.keys(): 
                print('No track info found for {} in tracks file. Skipping.'.format(midi_file))
                continue
            info = self.__tracks_info__[__get_midi_file_name_only__(midi_file)]
            transposed_melody_track = song.instruments[info[MELODY_TRACK]]
            transposed_instrumental_track = song.instruments[info[INSTRUMENT_TRACK]]
            # print(len(transposed_instrumental_track.transposed_frames[0]))
            # print(len(song.instruments))
            for t_index in range(len(transposed_instrumental_track.transposed_frames)):
                # __progress__.step(
                #     'readtransposed',
                #     t_index,
                #     12,
                #     progress_msg='Reading transposed tracks in ' + str(len(transposed_instrumental_track.transposed_frames)))
                melody_frames_as_int = transposed_melody_track.transposed_frames_as_int[t_index]
                instrumental_frames_as_int = transposed_instrumental_track.transposed_frames_as_int[t_index]
                count_frames += len(melody_frames_as_int) + len(instrumental_frames_as_int)
                new_song_frames_as_int = [melody_frames_as_int, instrumental_frames_as_int]
                # print(melody_frames.shape, instrumental_frames.shape)
                if type(self.__songs_as_int__) == type(None):
                    self.__songs_as_int__ = np.asarray(new_song_frames_as_int)
                else:
                    self.__songs_as_int__ = np.concatenate((self.__songs_as_int__, new_song_frames_as_int), axis=1)
                    print(len(new_song_frames_as_int[0]), len(new_song_frames_as_int[1]))
                # __progress__.step(
                #     'readtransposed',
                #     t_index+1,
                #     12,
                #     progress_msg='Reading transposed tracks in ' + str(t_index))
            __progress__.step(
                'openfiles',
                index+1,
                len(midi_paths),
                progress_msg=progress_msg,
                finish_msg='Loaded {} frames.'.format(count_frames))
        # self.__songs__ = np.asarray(self.__songs__)
        self.__songs_as_int____ = np.asarray(self.__songs_as_int__)
        # print(self.__songs__)
        # print(self.__songs__.shape)
        # print('Loaded {} frames.'.format(count_frames))               


class __Track__():
    def __init__(self, track, instrument_count, tracks_info, tracks_vocab, binary_velocity=True):
        self.track = track
        self.__read_track_info__(tracks_info, instrument_count, tracks_vocab)
        self.name = track.name
        self.binary_velocity = binary_velocity
        self.tempo = None
        self.frames = []
        self.transposed_frames = [[],]*(TRANSPOSE_UP_LIMIT+1-TRANSPOSE_DOWN_LIMIT)
        self.transposed_frames_as_int = [[],]*(TRANSPOSE_UP_LIMIT+1-TRANSPOSE_DOWN_LIMIT)
        self.current_frame = []
        self.channel = None
        self.__instrument_count__ = instrument_count
        self.__unique_frames__ = []
    
    def parse(self):
        for i,message in enumerate(self.track):
            __progress__.step(
                    'parsetrack',
                    i,
                    len(self.track),
                    progress_msg='Parsing track')
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
            
            __progress__.step(
                    'parsetrack',
                    i+1,
                    len(self.track),
                    progress_msg='Parsing track')
        if len(self.current_frame) > 0: self.__maybe_update_frames__()
        self.frames = np.asarray(self.frames)
        # self.transposed_frames = np.asarray(self.transposed_frames)
        for i,frames in enumerate(self.transposed_frames):
            self.transposed_frames[i] = np.asarray(frames)
            self.transposed_frames_as_int[i] = np.asarray(self.transposed_frames_as_int[i])
        # self.transposed_frames_as_int = np.asarray(self.transposed_frames_as_int)
        # print(self.transposed_frames.shape)
        return self

    def __transpose_frames__(self, amount, frames):
        transposed = np.roll(frames, amount, axis=0)
        if amount > 0: transposed[:amount] = 0
        elif amount < 0: transposed[amount:] = 0
        return transposed


    @staticmethod
    def toFile(frames):
        midi_file = MidiFile()
        track = MidiTrack()
        midi_file.tracks.append(track)

        current_frame = np.zeros(NOTES_RANGE)
        delta_time = 0
        for i,frame in enumerate(frames):
            idx = np.nonzero(current_frame - frame)
            delta_time += 5
            for note in idx[0]:
                msg = 'note_on' if frame[note] > 0 else 'note_off'
                track.append(Message(msg, note=note, velocity=70, time=delta_time))
                delta_time = 0
            current_frame = frame
        midi_file.save('generated.mid')

    def __read_track_info__(self, tracks_info, instrument_count, tracks_vocab):
        self.__internal_track_name__ = None
        self.__tracks_vocab__ = None
        for track in tracks_info:
            if tracks_info[track] == instrument_count:
                self.__internal_track_name__ = track
                self.__tracks_vocab__ = tracks_vocab[track]
                return
        

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
                    copy.deepcopy(self.frames[-1]) if len(self.frames) > 0\
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
                # When non transposable instruments are introduced:
                # TODO: Caution! This is ugly and messy. Some tracks will have frames, and others
                # will have transposed_frames. In the end it shouldn't hurt because we will only
                # pick what is mentioned by the vocabulary (so those with transposed_frames), but
                # it's still messy.
                if self.__tracks_vocab__ == None:
                    self.frames.append(copy.deepcopy(current_frame))
                    continue

                new_transposed_frames = []
                new_transposed_frames_as_int = []
                for i,amount in enumerate(range(TRANSPOSE_DOWN_LIMIT, TRANSPOSE_UP_LIMIT+1)):
                    transposed_frame = self.__transpose_frames__(amount, current_frame)
                    # key = np.array_str(transposed_frame)
                    key = tuple(transposed_frame.tolist())
                    # self.__unique_frames__.append('key')
                    if not key in self.__tracks_vocab__:
                        self.__tracks_vocab__[key] = len(self.__tracks_vocab__)
                    idx = self.__tracks_vocab__[key]
                    self.transposed_frames[i].append(transposed_frame)
                    self.transposed_frames_as_int[i].append(idx)

                
            self.current_frame = []


class __Song__():
    def __init__(self, midi_file, tracks_info, tracks_vocab, binary_velocity=True):
        self.__midi_file__ = midi_file
        self.__mido__ = MidiFile(midi_file, clip=True)
        self.binary_velocity = binary_velocity
        self.tempo = None
        self.instruments = []
        self.__parsed_instruments_count__ = 0
        self.__tracks_info__ = tracks_info
        self.__tracks_vocab__ = tracks_vocab

    def parse(self):
        for i,midi_track in enumerate(self.__mido__.tracks):
            __progress__.step(
                    'parseinstr',
                    i,
                    len(self.__mido__.tracks),
                    progress_msg='Parsing song')
            track = __Track__(midi_track,
                            self.__parsed_instruments_count__, 
                            self.__tracks_info__[__get_midi_file_name_only__(self.__midi_file__)],
                            self.__tracks_vocab__,
                            self.binary_velocity)
            track.parse()
            self.__add_instrument__(track)
            __progress__.step(
                    'parseinstr',
                    i+1,
                    len(self.__mido__.tracks),
                    progress_msg='Parsing song')
        self.__pad_frames__()
        if self.tempo == None: self.tempo = DEFAULT_TEMPO
        return self

    def __add_instrument__(self, track:__Track__):
        if self.tempo == None:
            self.tempo = track.tempo
        if len(track.frames) > 0 or len(track.transposed_frames[0]) > 0:
            # Only add the track as an instrument if it has frames
            self.instruments.append(track)
    
    def __pad_frames__(self):
        frame_length = max([len(instrument.frames) for instrument in self.instruments])
        transposed_length = max([len(instrument.transposed_frames[0]) for instrument in self.instruments])
        length = max(frame_length, transposed_length)
        # print(length)
        for i,instrument in enumerate(self.instruments):
            # print(instrument.frames.shape)
            if len(instrument.frames) > 0:
                frames = instrument.frames
                # print(type(frames))
                padded_frames = np.zeros((length, NOTES_RANGE), dtype=int)
                padded_frames[:frames.shape[0]] = frames
                self.instruments[i].frames = padded_frames
            if instrument.transposed_frames[0].shape[0] > 0:
                # print(self.instruments[i].transposed_frames.shape)
                # self.instruments[i].transposed_frames = self.instruments[i].transposed_frames.tolist()
                # print('padding for instrument '+ str(i))
                
                for j, frames in enumerate(instrument.transposed_frames):
                    frames_as_int = instrument.transposed_frames_as_int[j]
                    # print(frames.shape)
                    padded_frames = np.zeros((length, NOTES_RANGE), dtype=int)
                    padded_frames[:frames.shape[0]] = frames
                    
                    padded_frames_as_int = np.zeros((length), dtype=int)
                    padded_frames_as_int[:frames_as_int.shape[0]] = frames_as_int

                    self.instruments[i].transposed_frames[j] = padded_frames
                    self.instruments[i].transposed_frames_as_int[j] = padded_frames_as_int
                # print(len(self.instruments[i].transposed_frames[0]))
            # print(padded_frames.shape)
    
    def toDimp(self, name:str='', relative_path=''):
        name = name if name != '' else self.__midi_file__[:-4] + FILE_EXTENSION
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


def __get_midi_file_name_only__(midi_file):
    midi_file = midi_file.split('/')[-1]
    if midi_file[-4:] == '.mid': return midi_file[:-4]
    return midi_file

def __get_midi_file_name_with_extension__(midi_file):
    if midi_file[-4:] != '.mid': return midi_file + '.mid'
    return midi_file


def __showprogress__(percentage, message='', sub=False):
    length = 30
    percentage = percentage if percentage < 1 else 1
    done = int(length*percentage)
    left = length - done
    arrow = 0
    if done < length:
        done -= 1 if done > 0 else 0
        arrow = 1
    sys.stdout.write('\r[{}{}{}] {}% {}'.format(
        '='*done, '>'*arrow, '.'*left, int(round(percentage, 2)*100), message))
    if percentage == 1 and not sub:
        print('')
        

if __name__ == '__main__':
    # __Song__('midis/smile.mid').parse().toDimp()
    # song = __Song__('midis/smile.mid').parse()#.toDimp()
    # __Track__.toFile(song.instruments[1].frames)
    data = MidiData('midis/tracks.csv', open_files=['midis/smile.mid'])
    print(data.unique_frames_sizes())
    # print(data.tracks_info)