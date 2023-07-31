import sys
import time
from collections import deque

# Demo to update some lines
# Prints N=3 lines, then keeps removing the oldest when inserting a new one.

sys.stdout.write("Hello, world!\n")

queue = deque([], 3)
for t in range(20):
    time.sleep(0.5)
    s = "update %d" % t
    for _ in range(len(queue)):
        sys.stdout.write("\x1b[1A\x1b[2K") # move up cursor and delete whole line
    queue.append(s)
    for i in range(len(queue)):
        sys.stdout.write(queue[i] + "\n") # reprint the lines

sys.stdout.write("Bye, world!\n")
