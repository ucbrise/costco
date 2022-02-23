from .types import cint, sint, sintarray

M = 512
N = 4

d = sintarray(M * N)

server = sintarray(M * N)
client = sintarray(N)

distances = sintarray(M)
i = cint(0)
for _ in range(M)
    j = cint(0)
    for _ in range(N) :
        distances[i] = server[i*N+j] -
    j = j + 1
    i = i + 1


