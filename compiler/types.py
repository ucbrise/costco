from enum import Enum

class Role(Enum):
    SERVER = "SERVER"
    CLIENT = "CLIENT"

from typing import TypeVar


T = TypeVar('T')

class cint(int):
    pass


class sint(int):
    pass

class sintarray(list):
    pass


class cbool(int):
    pass


class Input:
    def __init__(self, role: Role, obj: object):
        self.role = role
        self.obj = obj

class Output:
    def __init__(self, obj: object):
        self.obj = obj


def is_secret(t: T) -> bool:
    return t == sint or t == sintarray
