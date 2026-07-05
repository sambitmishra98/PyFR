import copy

import numpy as np

from pyfr.mpiutil import DestExchanger


class EtypeExchanger:
    def __init__(self, comm, dests, geidxs):
        self._exch = DestExchanger(comm, dests)
        self._perm = None
        self.dests = dests
        self.oldgeidxs = geidxs
        self.newgeidxs = self._exch.Exchange(geidxs)
        self._oldsrt = np.argsort(geidxs)
        self._newsrt = np.argsort(self.newgeidxs)

    def with_perm(self, order):
        ex = copy.copy(self)
        ex._perm = order
        ex.newgeidxs = self.newgeidxs[order]
        ex._newsrt = np.argsort(ex.newgeidxs)
        return ex

    def Exchange(self, arr, axis=0):
        recv = self._exch.Exchange(arr, axis=axis)

        if self._perm is not None:
            return np.take(recv, self._perm, axis=axis)
        else:
            return recv

    def dest_of_global(self, geidxs):
        srt = self._oldsrt
        lidx = srt[np.searchsorted(self.oldgeidxs, geidxs, sorter=srt)]
        return self.dests[lidx]

    def new_local_of(self, geidxs):
        srt = self._newsrt
        return srt[np.searchsorted(self.newgeidxs, geidxs, sorter=srt)]

    def region_exchanger(self, oldgeidxs, newgeidxs):
        dests = self.dest_of_global(oldgeidxs)
        rgnexch = DestExchanger(self._exch.comm, dests)

        recvg = rgnexch.Exchange(oldgeidxs)
        newsrt = np.argsort(newgeidxs)
        recvl = newsrt[np.searchsorted(newgeidxs, recvg, sorter=newsrt)]

        return RegionExchanger(rgnexch, np.argsort(recvl))


class RegionExchanger:
    def __init__(self, exch, order):
        self._exch = exch
        self._order = order

    def Exchange(self, arr, axis=-1):
        recv = self._exch.Exchange(arr, axis=axis)
        return np.take(recv, self._order, axis=axis)


def make_exchangers(comm, eidxs, ownermap, etypes):
    exchangers = {}

    for et in etypes:
        dests = np.asarray(ownermap.get(et, []), dtype=int)
        geidxs = np.asarray(eidxs.get(et, []), dtype=int)
        exchangers[et] = EtypeExchanger(comm, dests, geidxs)

    return exchangers
