*************************
[soln-plugin-dtaustats]
*************************

Write pseudo-step field statistics out to a CSV file. Parameterised
with

#. ``flushsteps`` --- flush to disk every ``flushsteps``:

    *int*

#. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

#. ``file-header`` --- if to output a header row or not:

    *boolean*

#. ``abstraction`` --- level of detail for the stats (1 or 2):

    *int*

Example::

    [soln-plugin-pseudostats]
    flushsteps = 100
    file = pseudostats.csv
    file-header = true
    abstraction = 2
