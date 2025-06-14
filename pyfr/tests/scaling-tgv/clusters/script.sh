# 0. put the CSV and cfg.ini in the same folder
CSV=benchmark.csv
CFG=cfg.ini

# 1. create / update meshes (silent)
pyfr meshmaker generate-mesh --options "$CSV" --silent

pyfr benchmark preprocess-configs --config cfg.ini --options benchmark.csv

dofs=(100000 200000 500000 1000000)
etypes=(tet hex)
orders=(2 4 6)

# Loop over all configurations

for order in "${orders[@]}"; do
    for etype in "${etypes[@]}"; do
        for dofs in "${dofs[@]}"; do
            echo "Running benchmark for etype=${etype}, order=${order}, dofs=${dofs}"
            CMD="pyfr run -b hip \
            etype-${etype}_order-${order}_dof-${dofs}.pyfrm \
            cfgsolver_order-${order}_soln-plugin-writer_basename-${etype}p${order}d${dofs}.ini"
            echo "Command: $CMD"
            eval $CMD
        done
    done
done

pyfr benchmark postprocess --files *.pyfrs --options 'config:solver-time-integrator_scheme' 'stats:mesh_nelems-.*' 'stats:observer-onerankcomputetime_mean' 'stats:observer-onerankcomputetime_sem' 'stats:mesh_gndofs' --output bench_results.csv
