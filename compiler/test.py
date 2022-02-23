from .types import cint, sint
LEN = 32

def remainder(val, mod):
    mod = mod
    val = val
    rem = cint(0)
    j = cint(LEN-1)
    for _unused in range(LEN):
        rem = rem << 1
        rem = rem + ((val >> j) & 1)
        if rem >= mod:
            rem = rem - mod
        j = j - cint(1)
    return rem


def gcd(a, b):
    x = a
    y = b
    for i in range(2*LEN):
        if y != 0:
            r = remainder(x, y)
            x = y
            y = r
    return x


a = gcd(sint(1), sint(2))
