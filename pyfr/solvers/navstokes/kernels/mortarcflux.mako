<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<% beta, tau = c['ldg-beta'], c['ldg-tau'] %>

<%pyfr:kernel name='mortarcflux' ndim='1'
              ucoarse='inout view fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              ufine0='inout view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              ufine1='inout view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              ic0='in fpdtype_t[${str(nmpts)}][${str(ncfpts)}]'
              ic1='in fpdtype_t[${str(nmpts)}][${str(ncfpts)}]'
              if0='in fpdtype_t[${str(nmpts)}][${str(ntfpts)}]'
              if1='in fpdtype_t[${str(nmpts)}][${str(ntfpts)}]'
              pc0='in fpdtype_t[${str(ncfpts)}][${str(nmpts)}]'
              pc1='in fpdtype_t[${str(ncfpts)}][${str(nmpts)}]'
              pf0='in fpdtype_t[${str(ntfpts)}][${str(nmpts)}]'
              pf1='in fpdtype_t[${str(ntfpts)}][${str(nmpts)}]'
              nl0='in fpdtype_t[${str(nmpts)}][${str(ndims)}]'
              nl1='in fpdtype_t[${str(nmpts)}][${str(ndims)}]'
              gradcoarse0='in view fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              gradcoarse1='in view fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              gradcoarse2='in view fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              gradfine00='in view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              gradfine01='in view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              gradfine02='in view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              gradfine10='in view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              gradfine11='in view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              gradfine12='in view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'>
    fpdtype_t fc[${ncfpts}][${nvars}] = {};
    fpdtype_t ff0[${ntfpts}][${nvars}] = {};
    fpdtype_t ff1[${ntfpts}][${nvars}] = {};

% for child in range(2):
    for (int q = 0; q < ${nmpts}; q++)
    {
        fpdtype_t ul[${nvars}] = {};
        fpdtype_t ur[${nvars}] = {};
        fpdtype_t gradul[${ndims}][${nvars}] = {{0}};
        fpdtype_t gradur[${ndims}][${nvars}] = {{0}};

        for (int j = 0; j < ${ncfpts}; j++)
        {
        % for v in range(nvars):
            ul[${v}] += ic${child}[q][j]*ucoarse[j][${v}];
        % for d in range(ndims):
            gradul[${d}][${v}] += ic${child}[q][j]
                                  *gradcoarse${d}[j][${v}];
        % endfor
        % endfor
        }
        for (int j = 0; j < ${ntfpts}; j++)
        {
        % for v in range(nvars):
            ur[${v}] += if${child}[q][j]*ufine${child}[j][${v}];
        % for d in range(ndims):
            gradur[${d}][${v}] += if${child}[q][j]
                                  *gradfine${child}${d}[j][${v}];
        % endfor
        % endfor
        }

        fpdtype_t mag_n = sqrt(${pyfr.dot(f'nl{child}[q][{{i}}]', i=ndims)});
        <% norm = f'(1 / mag_n)*nl{child}[q][{{i}}]' %>
        fpdtype_t norm_n[] = ${pyfr.array(norm, i=ndims)};

        fpdtype_t ficomm[${nvars}], fvcomm, fn;
        ${pyfr.expand('rsolve', 'ul', 'ur', 'norm_n', 'ficomm')};

    % if beta != -0.5:
        fpdtype_t fvl[${ndims}][${nvars}] = {{0}};
        ${pyfr.expand('viscous_flux_add', 'ul', 'gradul', 'fvl')};
    % endif
    % if beta != 0.5:
        fpdtype_t fvr[${ndims}][${nvars}] = {{0}};
        ${pyfr.expand('viscous_flux_add', 'ur', 'gradur', 'fvr')};
    % endif

    % for v in range(nvars):
<% fvl_n = ' + '.join(f'norm_n[{d}]*fvl[{d}][{v}]' for d in range(ndims)) %>
<% fvr_n = ' + '.join(f'norm_n[{d}]*fvr[{d}][{v}]' for d in range(ndims)) %>
    % if beta == -0.5:
        fvcomm = ${fvr_n};
    % elif beta == 0.5:
        fvcomm = ${fvl_n};
    % else:
        fvcomm = ${0.5 + beta}*(${fvl_n})
               + ${0.5 - beta}*(${fvr_n});
    % endif
    % if tau != 0.0:
        fvcomm += ${tau}*(ul[${v}] - ur[${v}]);
    % endif

        fn = mag_n*(ficomm[${v}] + fvcomm);
        for (int j = 0; j < ${ncfpts}; j++)
            fc[j][${v}] += pc${child}[j][q]*fn;
        for (int j = 0; j < ${ntfpts}; j++)
            ff${child}[j][${v}] -= pf${child}[j][q]*fn;
    % endfor
    }
% endfor

% for j in range(ncfpts):
% for v in range(nvars):
    ucoarse[${j}][${v}] = fc[${j}][${v}];
% endfor
% endfor
% for j in range(ntfpts):
% for v in range(nvars):
    ufine0[${j}][${v}] = ff0[${j}][${v}];
    ufine1[${j}][${v}] = ff1[${j}][${v}];
% endfor
% endfor
</%pyfr:kernel>
