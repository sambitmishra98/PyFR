<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>

<%pyfr:kernel name='mortarcflux' ndim='1'
              uc='inout view fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              uf0='inout view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              uf1='inout view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              ic0='in fpdtype_t[${str(nmpts)}][${str(ncfpts)}]'
              ic1='in fpdtype_t[${str(nmpts)}][${str(ncfpts)}]'
              if0='in fpdtype_t[${str(nmpts)}][${str(ntfpts)}]'
              if1='in fpdtype_t[${str(nmpts)}][${str(ntfpts)}]'
              pc0='in fpdtype_t[${str(ncfpts)}][${str(nmpts)}]'
              pc1='in fpdtype_t[${str(ncfpts)}][${str(nmpts)}]'
              pf0='in fpdtype_t[${str(ntfpts)}][${str(nmpts)}]'
              pf1='in fpdtype_t[${str(ntfpts)}][${str(nmpts)}]'
              nl0='in fpdtype_t[${str(nmpts)}][${str(ndims)}]'
              nl1='in fpdtype_t[${str(nmpts)}][${str(ndims)}]'>
    fpdtype_t fc[${ncfpts}][${nvars}] = {};
    fpdtype_t ff0[${ntfpts}][${nvars}] = {};
    fpdtype_t ff1[${ntfpts}][${nvars}] = {};

% for child in range(2):
    for (int q = 0; q < ${nmpts}; q++)
    {
        fpdtype_t ul[${nvars}] = {};
        fpdtype_t ur[${nvars}] = {};

        for (int j = 0; j < ${ncfpts}; j++)
        {
        % for v in range(nvars):
            ul[${v}] += ic${child}[q][j]*uc[j][${v}];
        % endfor
        }
        for (int j = 0; j < ${ntfpts}; j++)
        {
        % for v in range(nvars):
            ur[${v}] += if${child}[q][j]*uf${child}[j][${v}];
        % endfor
        }

        fpdtype_t mag_n = sqrt(${pyfr.dot(f'nl{child}[q][{{i}}]', i=ndims)});
        <% norm = f'(1 / mag_n)*nl{child}[q][{{i}}]' %>
        fpdtype_t norm_n[] = ${pyfr.array(norm, i=ndims)};
        fpdtype_t fn[${nvars}];
        ${pyfr.expand('rsolve', 'ul', 'ur', 'norm_n', 'fn')};

        for (int j = 0; j < ${ncfpts}; j++)
        {
        % for v in range(nvars):
            fc[j][${v}] += pc${child}[j][q]*mag_n*fn[${v}];
        % endfor
        }
        for (int j = 0; j < ${ntfpts}; j++)
        {
        % for v in range(nvars):
            ff${child}[j][${v}] -= pf${child}[j][q]*mag_n*fn[${v}];
        % endfor
        }
    }
% endfor

% for j in range(ncfpts):
% for v in range(nvars):
    uc[${j}][${v}] = fc[${j}][${v}];
% endfor
% endfor
% for j in range(ntfpts):
% for v in range(nvars):
    uf0[${j}][${v}] = ff0[${j}][${v}];
    uf1[${j}][${v}] = ff1[${j}][${v}];
% endfor
% endfor
</%pyfr:kernel>
