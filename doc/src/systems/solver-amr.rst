************
[solver-amr]
************

The ``[solver-amr]`` section configures the initial online octree
h-adaptation path for affine hexahedral meshes.  Adaptation is evaluated only
at explicit physical-time targets, after the time integrator has completed an
accepted step and returned from ``advance_to``.  It is not evaluated on the
normal right-hand-side path.

Current Scope
=============

Native scheduled AMR currently requires all of the following:

* an explicit ``rk4`` integrator with ``controller = none``;
* the OpenMP backend and at least two MPI ranks;
* a pure affine Hex mesh;
* Euler, or Navier--Stokes with constant viscosity;
* no shock capturing or mixed spatial order;
* no ordinary solver/solution plugins or triggers;
* a shared filesystem path for staged native meshes.

Generalized nonconforming faces created by adaptation use the
``one-to-many-v1`` / ``quad-2x2`` mortar format.  Curved, periodic,
mixed-volume,
cross-rank h-mortar, CUDA, variable-viscosity, and adapted-MPI-restart
scheduling are outside the current native scheduler scope and fail closed where
applicable.

Indicator Options
=================

#. ``indicator`` --- element indicator

    ``density-variation`` (default)

    For each active Hex, the current indicator is

    .. math::

        \eta_e = \frac{\rho_{\max,e} - \rho_{\min,e}}
        {\max(|\bar{\rho}_e|, \rho_\mathrm{floor})}.

#. ``refine-threshold`` --- score at or above which a leaf may refine

    *float* (required)

#. ``coarsen-threshold`` --- maximum sibling-family score for coarsening

    *float* (required)

    Must satisfy ``0 <= coarsen-threshold < refine-threshold``.

#. ``density-floor`` --- positive denominator floor

    *float* (default: ``1e-14``)

#. ``tie-tolerance`` --- relative tolerance for deterministic score ties

    *float* (default: ``1e-12``)

#. ``min-level`` --- minimum octree level after coarsening

    *int* (default: ``0``)

#. ``max-level`` --- maximum octree refinement level

    *int* (default: ``2``)

At one adaptation target, refinement has priority.  Otherwise at most one
complete sibling family is selected for coarsening.  Existing AMR closure,
transfer, ownership, migration, validation, commit, and rollback logic remains
responsible for the topology-changing transaction.

Native Scheduling Options
=========================

Native scheduling is enabled only when ``schedule-dt`` is present.
Configurations containing ``[solver-amr]`` without this option may still use
the indicator API externally without changing the integrator lifecycle.

#. ``schedule-dt`` --- interval between regular AMR decision times

    *float*, positive

#. ``schedule-start`` --- origin of the regular AMR time grid

    *float* (default: ``tstart``)

    Regular targets are the future members of
    ``schedule-start + k*schedule-dt``.  A grid member equal to the current
    physical time is skipped.  The absolute grid avoids phase-shifting future
    decision times when starting from a later physical time.

#. ``initial-time`` --- optional one-off early AMR decision time

    *float*

    This may be used to adapt shortly after startup before entering the regular
    schedule.

#. ``stage-dir`` --- shared directory for staged adapted native meshes

    *path* (required)

    Every MPI rank must be able to access the same path.  A node-local
    temporary directory is not sufficient for multi-node execution.

Only future targets strictly before ``tend`` are scheduled.  No AMR decision is
made at ``tend`` by the native scheduler because a terminal topology rebuild
cannot affect subsequent PDE evolution.

Example
-------

.. code-block:: ini

    [solver-amr]
    indicator = density-variation
    refine-threshold = 0.05
    coarsen-threshold = 0.01
    min-level = 0
    max-level = 1

    initial-time = 0.0025
    schedule-start = 0.0
    schedule-dt = 0.2
    stage-dir = /shared/scratch/my-case/amr-stage

This example performs one early AMR decision at ``t = 0.0025`` and then
regular decisions at ``0.2, 0.4, 0.6, ...`` while omitting any decision at the
final integration time.
