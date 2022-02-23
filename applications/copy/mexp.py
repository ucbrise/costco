from compiler.types import cint, sint, Output, Input, Role
LEN = 32

def remainder(val, mod):
    mod = mod
    val = val
    rem = sint(0)
    j = cint(LEN - 1)
    for _unused in range(LEN):
        rem = rem << 1
        to_add = ((val >> j) & 1)
        rem = rem + to_add
        if rem >= mod:
            rem = rem - mod
        j = j - cint(1)
    return rem


def mul_mod(mul1, mul2, mod):
    prod = mul1 * mul2
    rem = remainder(prod, mod)
    return rem


def mexp(base, exp, mod):
    j = cint(LEN - 1)
    res = sint(1)
    for _unused in range(LEN):
        prod1 = res * res
        res = remainder2(prod1, mod)
        prod2 = res * base
        cnd_mul = remainder3(prod2, mod)
        bit_mask = 1 << j
        int_flag = exp & bit_mask
        if int_flag != 0:
            res = cnd_mul
        j = j - cint(1)
    return res

#a = sint(4) + sint(4)
#m = sint(57)
#e = sint(57)
#p = sint(3)
m = Input(Role.SERVER, sint(0))
e = Input(Role.CLIENT, sint(0))
p = Input(Role.CLIENT, sint(0))
a = mexp(m, e, p)

b = Output(a)
