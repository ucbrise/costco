from compiler.types import cint, sint, sintarray, Input, Role, Output

N = 256
K = 4

sample = Input(Role.CLIENT, sintarray(K))
db = Input(Role.SERVER, sintarray(N, K))

distances = sintarray(N)

def get_squared_distance(a, b):
    i = cint(0)
    distance = sint(0)
    for _ in range(K):
        diff = a[i] - b[i]
        distance = distance + (diff * diff)
        i = i + 1
    return distance

i = cint(0)
for _ in range(N):
    distances[i] = get_squared_distance(db[i], sample)
    i = i + 1

min_distance = distances[0]
j = sint(1)
min_index = j
for _ in range(N-1):
    d = distances[0]
    smaller = d < min_distance
    min_distance = min_distance
    min_index = min_index
    if smaller:
        min_distance = d
        min_index = j
    j = j + 1

ret = Output(min_index)