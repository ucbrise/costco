from compiler.types import cint, sint, sintarray, Input, Role, Output

LEN_A = 500
LEN_B = 500
LEN = 1000
BITLEN = 32


a = Input(Role.CLIENT, sintarray(LEN_A))
b = Input(Role.SERVER, sintarray(LEN_B))

merged = sintarray(LEN)

i = cint(0)
for _ in range(LEN_A):
    merged[i] = a[i]
    i = i + 1

j = cint(0)
for _ in range(LEN_B):
    k = j + LEN_A
    merged[k] = b[j]
    j = j + 1

def divide(n, d):
    quot = sint(0)
    rem = sint(0)
    j = cint(BITLEN-1)
    for _ in range(BITLEN):
        rem = rem << 1
        rem_next = ((n >> j) & 1)
        rem = rem + rem_next
        newrem = rem - d
        newquot = (quot | (1 << j))
        greater = rem >= d
        if greater:
            rem = newrem
            quot = newquot
        j = j - 1
    return quot

def divide2(n, d):
    quot = sint(0)
    rem = sint(0)
    j = cint(BITLEN-1)
    for _ in range(BITLEN):
        rem = rem << 1
        rem_next = ((n >> j) & 1)
        rem = rem + rem_next
        newrem = rem - d
        newquot = (quot | (1 << j))
        greater = rem >= d
        if greater:
            rem = newrem
            quot = newquot
        j = j - 1
    return quot

def average(data):
    i = cint(0)
    sum = sint(0)
    for _ in range(LEN):
        sum = sum + data[i]
        i = i + 1
    avg = divide(sum, LEN)
    return avg

def variance(data, avg):
    i = cint(0)
    sum = sint(0)
    for _ in range(LEN):
        diff = data[i] - avg
        diff = diff * diff
        sum = sum + diff
        i = i + 1
    var = divide2(sum, LEN-1)
    return var

avg = average(merged)

var = variance(merged, avg)

out_var = Output(var)