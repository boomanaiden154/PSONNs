import os
import timeit
import sys

from timeit import default_timer as timer

times = [];
for x in range(0,1000):
    start = timer()
    os.system(sys.argv[1]);
    end = timer()
    times.append(end - start)

print "average time is printed below"
print sum(times) / len(times);
