#!/usr/bin/python

import sys
from scipy.sparse import random

n = int(sys.argv[1])
density = float(sys.argv[2])
seed = int(sys.argv[3])

m = random(n, n, density, format="csr", random_state=seed)
m.data *= 2
m.data -= 1

print n,  n, len(m.data), 0
for x in m.data:
	print x,
print
for x in m.indptr:
	print x,
print
for x in m.indices:
	print x,
