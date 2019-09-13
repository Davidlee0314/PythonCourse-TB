import pickle as pkl

def read_all(files):
    '''
    files: list(length of 3) of file location, [combine, train, test]
    '''
    if(len(files) != 3):
        raise ValueError('The files list need to be length of 3 (combine, train, test)')
    with open(files[0], 'rb') as f:
        combine = pkl.load(f)
    with open(files[1], 'rb') as f:
        train = pkl.load(f)
    with open(files[2], 'rb') as f:
        test = pkl.load(f)
    return combine, train, test

def read(file):
    '''
    files: string of file location, [combine, train, test]
    '''
    with open(file, 'rb') as f:
        temp = pkl.load(f)
    return temp