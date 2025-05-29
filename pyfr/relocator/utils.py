# Location: pyfr/relocator/crprint_utils.py

# export FORCE_COLOR=1

import sys
from termcolor import colored
from pprint import pformat
from pyfr.mpiutil import get_comm_rank_root

def crprint(rank, *args, sep=' ', end='\n', flush=True):
    """
    Print a colorized message for the given rank, and also
    write the same text to a file named rank-<rank>.txt.

    If you pass rank=-1, we'll auto-detect this rank via get_comm_rank_root().
    """
    comm, this_rank, root = get_comm_rank_root()
    if rank == -1:
        rank = this_rank
    
    # Choose color by rank
    color_map = {0: 'blue', 1: 'green', 2: 'red', 3: 'magenta'}
    color = color_map.get(rank, 'white')
    
    # Build the full message string from *args, just like print() does
    msg_body = sep.join(str(a) for a in args)
    # We'll add "[Rank X]" automatically
    msg_with_prefix = f"[Rank {rank}] {msg_body}"
    
    # Colorize for console
    colored_msg = colored(msg_with_prefix, color)
    
    # 1) Print to console (colorized)
    print(colored_msg, end=end)
    if flush:
        sys.stdout.flush()
    
    # 2) Also append to "rank-<rank>.txt" (plain)
    with open(f"rank-{rank}.txt", 'a', encoding='utf-8') as f:
        f.write(msg_with_prefix + end)
        if flush:
            f.flush()

def crpprint(rank, obj, flush=True):
    """
    Pretty-print (using Python's pprint) an object or data structure,
    colorized by rank, also tee'd to rank-<rank>.txt.

    If you pass rank=-1, we'll auto-detect this rank.
    """
    # Convert the object into a nicely formatted string
    s = pformat(obj)

    # Now we may have multiple lines.  We can either:
    # 1) Print them line-by-line (each line gets a [Rank X] prefix).
    # 2) Print them all in one chunk.

    # Option (A): line-by-line approach:
    # for line in s.splitlines():
    #     crprint(rank, line, flush=flush)

    # Option (B): all in one chunk:
    crprint(rank, s, flush=flush)
