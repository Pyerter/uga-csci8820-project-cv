import os

class DirectoryHelper():
    def __init__(self, base_dir: str = 'C:/Coding Projects/uga-csci8820-project-cv'):
        self.base_dir: str = base_dir

dir_helper = DirectoryHelper()
BASE_SEP = '/'

def fix_base_dir(basedir: str):
    dir_helper.base_dir = path_to_url(basedir)

def get_base_dir():
    return dir_helper.base_dir

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

def path_to_url(path: str, prepend:bool=False, append:bool=False) -> str:
    splits = path.split(os.sep)
    if len(splits) <= 1:
        if len(splits) == 1:
            return (BASE_SEP if prepend else '') + splits[0] + (BASE_SEP if append else '')
        else: return path

    url_path = splits[0]
    for i in range(1, len(splits)):
        url_path = url_path + BASE_SEP + splits[i]
    return (BASE_SEP if prepend else '') + url_path + (BASE_SEP if append else '')

def norm_path_from_base(path: str):
    path = get_base_dir() + BASE_SEP + path
    return norm_path(path)