import itertools as it

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BaseSolnPlugin, PostactionMixin, RegionMixin
from pyfr.writers.native import NativeWriter
from pyfr.util import first


class PseudoResidualWriterPlugin(PostactionMixin, RegionMixin, BaseSolnPlugin):
    name = 'pseudoresidualwriter'
    systems = ['*']
    formulations = ['dual']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Base output directory and file name
        basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(cfgsect, 'basename')

        if 'solver-dual-time-integrator-multip' in intg.cfg.sections():
            self.level = self.cfg.getint(self.cfgsect, 
                'level', intg.cfg.getint('solver','order'))

        # Get the element map and region data
        emap, erdata = intg.system.ele_map, self._ele_region_data

        # Figure out the shape of each element type in our region
        nvars = self.nvars
        ershapes = {etype: (nvars, emap[etype].nupts) for etype in erdata}

        # Construct the solution writer
        self._writer = NativeWriter.from_integrator(intg, basedir, basename,
                                                    'soln')
        self._writer.set_shapes_eidxs(ershapes, erdata)

        # Asynchronous output options
        self._async_timeout = self.cfg.getfloat(cfgsect, 'async-timeout', 60)

        # Output time step and last output time
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_last = intg.tcurr

        # Output field names
        self.fields = list(first(intg.system.ele_map.values()).convars)

        # Output data type
        self.fpdtype = intg.backend.fpdtype

        # Register our output times with the integrator
        intg.call_plugin_dt(intg.tcurr, self.dt_out)

        # If we're not restarting then make sure we write out the initial
        # solution when we are called for the first time
        if not intg.isrestart:
            self.tout_last -= self.dt_out

    def _prepare_metadata(self, intg):
        comm, rank, root = get_comm_rank_root()

        stats = Inifile()
        stats.set('data', 'fields', ','.join(self.fields))
        stats.set('data', 'prefix', 'soln')
        intg.collect_stats(stats)

        # If we are the root rank then prepare the metadata
        if rank == root:
            metadata = {**intg.cfgmeta, 'stats': stats.tostr(),
                        'mesh-uuid': intg.mesh_uuid}
        else:
            metadata = None

        # Fetch data from other plugins and add it to metadata with ad-hoc keys
        for csh in intg.plugins:
            try:
                prefix = intg.get_plugin_data_prefix(csh.name, csh.suffix)
                pdata = csh.serialise(intg)
            except AttributeError:
                pdata = {}

            if rank == root:
                metadata |= {f'{prefix}/{k}': v for k, v in pdata.items()}

        return metadata

    def get_pseudo_residual_field(self, intg):
        """
        Get difference of registers at the highest polynomial level
        """

        if 'solver-dual-time-integrator-multip' in intg.cfg.sections():
            pintg = intg.pseudointegrator.pintgs[self.level]
        else:
            pintg = intg.pseudointegrator
            
        res = []

        for em, dtaum in it.zip_longest(pintg.system.ele_banks, pintg.dtau_upts):
            r_prev = em[pintg._idxprev].get()
            r_curr = em[pintg._idxcurr].get()
            res.append((r_curr - r_prev)/dtaum.get())

        return res

    def _prepare_data(self, intg):
        data = {}

        pseudo_res = self.get_pseudo_residual_field(intg)

        for idx, etype, rgn in self._ele_regions:
            d = pseudo_res[idx]
            data[etype] = d[..., rgn].T.astype(self.fpdtype)

        return data

    def __call__(self, intg):
        self._writer.probe()

        if intg.tcurr - self.tout_last < self.dt_out - self.tol:
            return

        # Prepare the data and metadata
        data = self._prepare_data(intg)
        metadata = self._prepare_metadata(intg)

        # Prepare a callback to kick off any postactions
        callback = lambda fname, t=intg.tcurr: self._invoke_postaction(
            intg=intg, mesh=intg.system.mesh.fname, soln=fname, t=t
        )

        # Write out the file
        self._writer.write(data, intg.tcurr, metadata, self._async_timeout,
                           callback)

        # Update the last output time
        self.tout_last = intg.tcurr

    def finalise(self, intg):
        super().finalise(intg)

        self._writer.flush()
