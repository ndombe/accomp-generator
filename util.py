from sys import stdout

def showprogress(percentage, message='', sub=False):
    """
    Show a progress bar in your command line to track the progress of your process. This is meant to
    be used in cases such as for-loops that are susceptible to take some time to exectute
    (eg. predicting in a for loop for a non-trivial number of iterations).
    Arguments:
        * percentage: a float number less or equal to 1 that represents the current progress. This
            this can simply be thought of as the quotient "i/total" where "i" is the index of the
            current element in the iteration, and "total" is the total number of elements to iterate
            through. NOTE: You're advised to you "i+1" instead of "i" and to put the statement at
            the very bottom of the loop.
        * message (optional): a string message. If this variable is used and assigned a string, that
            string will be displayed next to the progress bar. This is mostly useful in case going
            through one iteration may require some heavy computations and therefore the for loop
            remains on the same element for some time. In such cases, this argument can be used to
            better indicate the status of the progress, eg. if there's a sort of label associated
            with each such iteration (like the name of something corresponding to the current
            index) we can display a message with that label (eg. "Now parsing file file_name").
        * sub (optional): a boolean determining weather the current process is a sub process. This
            is useful because this progress method will display a new line once it reaches 100%. If
            we want to avoid that, we set this argument to "True".
    
    NOTE: It's advised to put this method as last statement of the iteration, with the index being
        the next index instead of the current one, i.e. "i" in "i/total" should actually be "i+1".
    
    Example Usage:
    for i,element in enumerate(array):
        ...
        ...
        ...
        util.showprogress((i+1)/len(array))

    # OR

    for i in range(some_integer_length):
        ...
        ...
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
