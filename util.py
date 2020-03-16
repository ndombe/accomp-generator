from sys import stdout
import numpy as np

def showprogress(percentage, message='', sub=False):
    """
    Show a progress bar in your command line to track the progress of your process. This is meant to
    be used in cases such as for-loops that are susceptible to take some time to exectute
    (eg. predicting in a for loop for a non-trivial number of iterations).
    @param percentage: a float number less or equal to 1 that represents the current progress. This
            this can simply be thought of as the quotient "i/total" where "i" is the index of the
            current element in the iteration, and "total" is the total number of elements to iterate
            through. NOTE: You're advised to you "i+1" instead of "i" and to put the statement at
            the very bottom of the loop.
    @param message (optional): a string message. If this variable is used and assigned a string,
            that string will be displayed next to the progress bar. This is mostly useful in case 
            going through one iteration may require some heavy computations and therefore the for
            loop remains on the same element for some time. In such cases, this argument can be used
            to better indicate the status of the progress, eg. if there's a sort of label associated
            with each such iteration (like the name of something corresponding to the current
            index) we can display a message with that label (eg. "Now parsing file file_name").
    @param sub (optional): a boolean determining weather the current process is a sub process. This
            is useful because this progress method will display a new line once it reaches 100%. If
            we want to avoid that, we set this argument to "True". This argument is False by
            default.
    
    NOTE: It's advised to put this method as last statement of the iteration, with the index being
        the next index instead of the current one, i.e. "i" in "i/total" should actually be "i+1".
    
    Example Usage:
    for i,element in enumerate(array):
        ...
        # some computation
        ...
        util.showprogress((i+1)/len(array))

    # OR

    for i in range(some_integer_length):
        ...
        # some computation
        ...
        util.progress((i+1) / some_integer_length)
    """
    length = 30
    percentage = percentage if percentage < 1 else 1
    done = int(length*percentage)
    left = length - done
    arrow = 0
    if done < length:
        done -= 1 if done > 0 else 0
        arrow = 1
    stdout.write('\r[{}{}{}] {}% {}'.format(
        '='*done, '>'*arrow, '.'*left, int(round(percentage, 2)*100), message))
    if percentage == 1 and not sub:
        print('')


def chopin(references, candidates, mean=False):
    """
    Chord Proxy Substitution ([Cho]rd [P]roxy Subst[i]tutio[n])
    Given a reference chord and a candidate chord, the Chopin evaluation score gives an estimate (in
    the form of a probability ranging from 0 to 100) of how confident we can be that the candidate
    chord can be used as a valid proxy to substitute in place of the reference chord.
    @param references: nd-array/list of frames. This is a group of integers where a value greater
            than 0 indicates a note being played. The shape of 'references' must be at least 1 and
            the same as the shape of `candidates`.
    @param candidates: nd-array/list of frames. This is a group of integers where a value greater
            than 0 indicates a note being played. The shape of 'candidates' must be at least 1 and
            the same as the shape of `references`.
    @param mean(optional): a boolean determining if we want the method to precompute the mean of all
            the scores before returning. If True, the method will return a float value corresponding
            to the overall score, if False, the method will return an nd-array of similar dimensions
            as the 'candidates' parameters where each frame is replaced by its score. This argument
            is False by default.
    
    @return score: if `mean` is False, an nd-array of the similar shape as the 'candidates'
            parameter except for the last dimension where each frame array is replaced by its
            respective score against the corresponding references frame; if `mean` is True, the mean
            of all the scores for all the frames.
    """
    references, candidates = np.array(references, dtype=int), np.array(candidates, dtype=int)
    assert references.ndim >= candidates.ndim and references.ndim > 0, "The references and "\
            +"candidates arrays must have the same dimensions"
    NOTES_RANGE = candidates.shape[-1]
    stacked = np.stack([references, candidates], axis=-2)
    nonzeros = np.apply_along_axis(lambda x: str(np.nonzero(x)[0]), -1, stacked)#.astype(np.dtype(('U', 16)))
    # print()
    # print(list(nonzeros))

    def seq2score(nonzeros):
        ref_idx, can_idx = nonzeros
        ref_idx = np.fromstring(ref_idx[1:-1], sep=' ', dtype=int)
        can_idx = np.fromstring(can_idx[1:-1], sep=' ', dtype=int)
        if (ref_idx.shape[0] == 0 and can_idx.shape[0] != 0) or\
            (ref_idx.shape[0] != 0 and can_idx.shape[0] == 0):
            # Either `candidate` played where `reference` didn't play, or `candidate` didn't play
            # where `reference` played. In both cases, `candidate` is definitely not a proxy of
            # `reference`.
            return 0
        elif ref_idx.shape[0] == 0 and can_idx.shape[0] == 0:
            # `candidate` didn't play where `reference` didn't play. This is 100% match.
            return 1

        root_score = 1.
        ref2can_activation = 1.
        can2ref_activation = 1.
        NBR_NOTES = 12

        # As far as the Chopin score is concerned, we're only interested in the musical notes, not
        # the octave in which they're played. Therefore, we simplify all of them to the basic 12
        # notes.
        reduced_ref_idx = ref_idx%NBR_NOTES
        reduced_can_idx = can_idx%NBR_NOTES

        # Root Score
        if reduced_can_idx[0] == reduced_ref_idx[0]:
            # `candidate` has the same root note as `reference`. 100% match on the root.
            root_score = 1
        elif reduced_can_idx[0] in reduced_ref_idx[1:]:
            # If the root note in the candidate is a valid relative 3rd or 5th of the root note in
            # the reference and exists amongst the notes of the reference, then it's not a total
            # match but a very probably one (75%). Else if the new root note is not a relative of
            # the old root note, but is still amongst the notes in reference, then we give it 50%.
            true_root = reduced_ref_idx[0]
            relatives = [(true_root+i)%NBR_NOTES for i in [3,4,6,7,8]]
            root_score = .75 if reduced_can_idx[0] in relatives else .5
        else:
            root_score = .1
        
        # Reference to Candidate Score
        for note in reduced_ref_idx[1:]:
            if note not in reduced_can_idx:
                ref2can_activation -= 1/len(reduced_ref_idx[1:])

        # Reference to Candidate Average Note Proximity
        ref2can_proximity = np.mean(list(map(
            lambda x: np.max((NOTES_RANGE - np.abs(can_idx-x))/NOTES_RANGE), ref_idx)))
        

        # Candidate to Reference Score
        for note in reduced_can_idx[1:]:
            if note not in reduced_ref_idx:
                can2ref_activation -= 1/len(reduced_can_idx[1:])

        # Candidate to Reference Average Note Proximity
        can2ref_proximity = np.mean(list(map(
            lambda x: np.max((NOTES_RANGE - np.abs(ref_idx-x))/NOTES_RANGE), can_idx)))

        # ref2can_proximity = 1.
        # can2ref_proximity = 1.
        score =\
            root_score*ref2can_activation*can2ref_activation*ref2can_proximity*can2ref_proximity*100
        # print(score)
        return score

    # NOTE: An existing issue with numpy (https://github.com/numpy/numpy/issues/8352) is casting
    # of the regular `np.apply_along_axis` into integers while the score function was verified to be
    # correctly returning floats. This is why we're using the equivalent of this function under
    # `np.ma`. Benchmarks tests show that it runs a bit slower, but at least we keep the decimal
    # precision that we need.
    scores = np.ma.apply_along_axis(seq2score, -1, nonzeros)
    # print(list(scores))
    if mean:
        return np.mean(scores)
    else:
        
        return scores

if __name__ == '__main__':
    ref1 = [[[1,0,1,1], [1,0,1,1], [0,1,1,0]]]
    ref2 = [[[1,1,1,1], [0,0,1,1], [0,0,1,0]]]
    can1 = [[[0,1,1,1], [0,1,1,0], [0,0,0,0]]]
    can2 = [[[1,0,1,1], [1,0,1,1], [0,0,1,0]]]

    ref = [[[1,0,1,1], [1,0,1,1], [0,1,1,0]], [[1,1,1,1], [0,0,1,1], [0,0,1,0]]]
    can = [[[0,1,1,1], [0,1,1,0], [0,1,1,0]], [[1,0,1,1], [1,0,1,1], [0,0,1,0]]]
    # ref = [1,1,1,0]
    # can = [0,0,1,0]
    # ref = [[1,0,1,1], [1,0,1,1], [0,1,1,0]]
    # can = [[0,1,1,1], [0,1,1,0], [0,1,1,0]]
    
    ref = np.array([np.random.rand(128)]*1698)
    # ref = np.random.rand(1698,128)
    ref[ref >= .9] = 1
    ref[ref < .9] = 0
    can = np.random.rand(1698,128)
    can[can >= .9] = 1
    can[can < .9] = 0

    s = chopin(ref, can)
    # s /= 6
    print(s)