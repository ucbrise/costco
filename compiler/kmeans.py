from .types import cint, sint, sintarray, Input, Role

DIMENSIONS = 2
OBSERVATIONS = 500
ITERATIONS = 8
CENTROIDS = 4

x = sintarray(DIMENSIONS)
y = sintarray(DIMENSIONS)
observations = Input(Role.SERVER, sintarray(OBSERVATIONS, DIMENSIONS))

centroids = sintarray(CENTROIDS, DIMENSIONS)

i = cint(0)
for _ in range(CENTROIDS):
    j = cint(0)
    for _ in range(DIMENSIONS):
        centroids[i][j] = observations[i][j]
        j = j + 1
    i = i + 1

assignments = sintarray(OBSERVATIONS, CENTROIDS)


def get_squared_distance(a, b):
    i = cint(0)
    distance = sint(0)
    for _ in range(DIMENSIONS):
        diff = a[i] - b[i]
        distance = distance + (diff * diff)
        i = i + 1
    return distance


def get_squared_distance2(a, b):
    i = cint(0)
    distance = sint(0)
    for _ in range(DIMENSIONS):
        diff = a[i] - b[i]
        distance = distance + (diff * diff)
        i = i + 1
    return distance


def find_clusters(observations, centroids, assignments):
    i = cint(0)
    for _ in range(OBSERVATIONS):
        min_distance = get_squared_distance(observations[i], centroids[0])
        assignments[i][0] = sint(1)
        j = cint(1)
        for _ in range(CENTROIDS-1):
            distance = get_squared_distance2(observations[i], centroids[j])
            smaller = distance < min_distance
            assignments[i][j] = smaller
            if smaller:
                min_distance = distance
            j = j + 1
        i = i + 1
    return True

def update_centroids(observations, centroids, assignments):
    cluster_sizes = sintarray(CENTROIDS)
    i = cint(0)
    for _ in range(CENTROIDS):
        cluster_sizes[i] = 0
        d = cint(0)
        for _ in range(DIMENSIONS):
            centroids[i][d] = 0
            d = d + 1
        i = i + 1

    o = cint(0)
    for _ in range(OBSERVATIONS):
        c_inv = cint(1)
        found_cluster = sint(1)
        for _ in range(CENTROIDS):
            c = CENTROIDS - c_inv
            in_cluster = found_cluster * assignments[o][c]
            found_cluster = found_cluster * (1-assignments[o][c])
            cluster_sizes[c] = cluster_sizes[c] + in_cluster
            d = cint(0)
            for _ in range(DIMENSIONS):
                centroids[c][d] = centroids[c][d] + (in_cluster * observations[o][d])
                d = d + 1
            c_inv = c_inv + 1
        o = o + 1

    c = cint(0)
    for _ in range(CENTROIDS):
        d = cint(0)
        for _ in range(DIMENSIONS):
            centroids[c][d] = centroids[c][d] / cluster_sizes[c]
            d = d + 1
        c = c + 1
    return True


for _ in range(ITERATIONS):
    z = find_clusters(observations, centroids, assignments)
    x = update_centroids(observations, centroids, assignments)
