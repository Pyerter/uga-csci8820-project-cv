import os

BASE_DIR = 'C:/Coding Projects/uga-csci8820-project-cv'
BASE_SEP = '/'

def norm_path(path: str, separator:str=BASE_SEP, prepend:bool=False, append:bool=False) -> str:
    splits = path.split(separator)
    if len(splits) <= 1:
        if len(splits) == 1:
            return (os.sep if prepend else '') + splits[0] + (os.sep if append else '')
        else:
            return path
    
    norm_path = splits[0]
    for i in range(1, len(splits)):
        norm_path = norm_path + os.sep + splits[i]
    return (os.sep if prepend else '') + norm_path + (os.sep if append else '')

def norm_path_from_base(path: str):
    path = BASE_DIR + BASE_SEP + path
    return norm_path(path)