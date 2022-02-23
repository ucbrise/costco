from .types import cint, sint, sintarray, Input, Role

N = 500
x = sintarray(N)
y = sintarray(N)

j = cint(0)
for _ in range(N):
    j = j + 1

i = cint(0)
for _ in range(N):
    y[i] = x[i] * y[i]
    i = i + 1
