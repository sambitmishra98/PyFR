<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<% beta, tau = c['ldg-beta'], c['ldg-tau'] %>

<%pyfr:kernel name='mortarfluxpatch' ndim='2'
              ul='in fpdtype_t[${str(nvars)}]'
              ur='in fpdtype_t[${str(nvars)}]'
              gl0='in fpdtype_t[${str(nvars)}]'
              gl1='in fpdtype_t[${str(nvars)}]'
              gl2='in fpdtype_t[${str(nvars)}]'
              gr0='in fpdtype_t[${str(nvars)}]'
              gr1='in fpdtype_t[${str(nvars)}]'
              gr2='in fpdtype_t[${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'
              fc='out fpdtype_t[${str(nvars)}]'>
    fpdtype_t gradul[${ndims}][${nvars}];
    fpdtype_t gradur[${ndims}][${nvars}];
% for d in range(ndims):
% for v in range(nvars):
    gradul[${d}][${v}] = gl${d}[${v}];
    gradur[${d}][${v}] = gr${d}[${v}];
% endfor
% endfor

    fpdtype_t mag_n = sqrt(${pyfr.dot('nl[{i}]', i=ndims)});
    fpdtype_t norm_n[] = ${pyfr.array(
        '(1 / mag_n)*nl[{i}]', i=ndims
    )};

    fpdtype_t ficomm[${nvars}];
    ${pyfr.expand('rsolve', 'ul', 'ur', 'norm_n', 'ficomm')};

% if beta != -0.5:
    fpdtype_t fvl[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand('viscous_flux_add', 'ul', 'gradul', 'fvl')};
% endif
% if beta != 0.5:
    fpdtype_t fvr[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand('viscous_flux_add', 'ur', 'gradur', 'fvr')};
% endif

    fpdtype_t fvcomm;
% for v in range(nvars):
<% fvl_n = ' + '.join(f'norm_n[{d}]*fvl[{d}][{v}]' for d in range(ndims)) %>
<% fvr_n = ' + '.join(f'norm_n[{d}]*fvr[{d}][{v}]' for d in range(ndims)) %>
% if beta == -0.5:
    fvcomm = ${fvr_n};
% elif beta == 0.5:
    fvcomm = ${fvl_n};
% else:
    fvcomm = ${0.5 + beta}*(${fvl_n}) + ${0.5 - beta}*(${fvr_n});
% endif
% if tau != 0.0:
    fvcomm += ${tau}*(ul[${v}] - ur[${v}]);
% endif
    fc[${v}] = mag_n*(ficomm[${v}] + fvcomm);
% endfor
</%pyfr:kernel>
