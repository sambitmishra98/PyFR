*******************
[solver-interfaces]
*******************

Parameterises the interfaces with

1. ``riemann-solver`` --- type of Riemann solver:

    ``rusanov`` | ``hll`` | ``hllc`` | ``roe`` | ``roem`` | ``exact``

2. ``ldg-beta`` --- beta parameter used for LDG:

    *float*

3. ``ldg-tau`` --- tau parameter used for LDG:

    *float*

4. ``mortar-implementation`` --- implementation used for nonconforming
   mortar interfaces:

    ``fused`` | ``staged``

   ``fused`` retains the original pointwise mortar kernel. ``staged``
   delegates interpolation and projection to the backend matrix providers
   and evaluates only the common flux pointwise. The default is ``fused``.

Example:

.. code-block:: ini

    [solver-interfaces]
    riemann-solver = rusanov
    ldg-beta = 0.5
    ldg-tau = 0.1
    mortar-implementation = staged
