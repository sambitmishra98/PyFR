<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<% beta, tau = c['ldg-beta'], c['ldg-tau'] %>

<%pyfr:kernel name='mortarfluxstaged' ndim='2'
              ul0='in fpdtype_t[${str(nvars)}]'
              ur0='in fpdtype_t[${str(nvars)}]'
              ul1='in fpdtype_t[${str(nvars)}]'
              ur1='in fpdtype_t[${str(nvars)}]'
              gl00='in fpdtype_t[${str(nvars)}]'
              gl01='in fpdtype_t[${str(nvars)}]'
              gl02='in fpdtype_t[${str(nvars)}]'
              gr00='in fpdtype_t[${str(nvars)}]'
              gr01='in fpdtype_t[${str(nvars)}]'
              gr02='in fpdtype_t[${str(nvars)}]'
              gl10='in fpdtype_t[${str(nvars)}]'
              gl11='in fpdtype_t[${str(nvars)}]'
              gl12='in fpdtype_t[${str(nvars)}]'
              gr10='in fpdtype_t[${str(nvars)}]'
              gr11='in fpdtype_t[${str(nvars)}]'
              gr12='in fpdtype_t[${str(nvars)}]'
              nl0='in fpdtype_t[${str(ndims)}]'
              nl1='in fpdtype_t[${str(ndims)}]'
              fc0='out fpdtype_t[${str(nvars)}]'
              fc1='out fpdtype_t[${str(nvars)}]'>
% for child in range(2):
    fpdtype_t gradul${child}[${ndims}][${nvars}];
    fpdtype_t gradur${child}[${ndims}][${nvars}];
% for d in range(ndims):
% for v in range(nvars):
    gradul${child}[${d}][${v}] = gl${child}${d}[${v}];
    gradur${child}[${d}][${v}] = gr${child}${d}[${v}];
% endfor
% endfor

    fpdtype_t mag_n${child} = sqrt(
        ${pyfr.dot(f'nl{child}[{{i}}]', i=ndims)}
    );
    fpdtype_t norm_n${child}[] = ${pyfr.array(
        f'(1 / mag_n{child})*nl{child}[{{i}}]', i=ndims
    )};

    fpdtype_t ficomm${child}[${nvars}];
    ${pyfr.expand(
        'rsolve', f'ul{child}', f'ur{child}',
        f'norm_n{child}', f'ficomm{child}'
    )};

% if beta != -0.5:
    fpdtype_t fvl${child}[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand(
        'viscous_flux_add', f'ul{child}', f'gradul{child}',
        f'fvl{child}'
    )};
% endif
% if beta != 0.5:
    fpdtype_t fvr${child}[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand(
        'viscous_flux_add', f'ur{child}', f'gradur{child}',
        f'fvr{child}'
    )};
% endif

    fpdtype_t fvcomm${child};
% for v in range(nvars):
<% fvl_n = ' + '.join(
    f'norm_n{child}[{d}]*fvl{child}[{d}][{v}]' for d in range(ndims)
) %>
<% fvr_n = ' + '.join(
    f'norm_n{child}[{d}]*fvr{child}[{d}][{v}]' for d in range(ndims)
) %>
% if beta == -0.5:
    fvcomm${child} = ${fvr_n};
% elif beta == 0.5:
    fvcomm${child} = ${fvl_n};
% else:
    fvcomm${child} = ${0.5 + beta}*(${fvl_n})
                               + ${0.5 - beta}*(${fvr_n});
% endif
% if tau != 0.0:
    fvcomm${child} += ${tau}*(ul${child}[${v}] - ur${child}[${v}]);
% endif
    fc${child}[${v}] = mag_n${child}*(
        ficomm${child}[${v}] + fvcomm${child}
    );
% endfor
% endfor
</%pyfr:kernel>
