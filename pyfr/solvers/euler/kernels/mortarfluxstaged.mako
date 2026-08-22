<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>

<%pyfr:kernel name='mortarfluxstaged' ndim='2'
              ul0='inout fpdtype_t[${str(nvars)}]'
              ur0='in fpdtype_t[${str(nvars)}]'
              ul1='inout fpdtype_t[${str(nvars)}]'
              ur1='in fpdtype_t[${str(nvars)}]'
              nl0='in fpdtype_t[${str(ndims)}]'
              nl1='in fpdtype_t[${str(ndims)}]'>
% for child in range(2):
    fpdtype_t mag_n${child} = sqrt(
        ${pyfr.dot(f'nl{child}[{{i}}]', i=ndims)}
    );
    fpdtype_t norm_n${child}[] = ${pyfr.array(
        f'(1 / mag_n{child})*nl{child}[{{i}}]', i=ndims
    )};
    fpdtype_t fn${child}[${nvars}];
    ${pyfr.expand(
        'rsolve', f'ul{child}', f'ur{child}',
        f'norm_n{child}', f'fn{child}'
    )};
% for v in range(nvars):
    ul${child}[${v}] = mag_n${child}*fn${child}[${v}];
% endfor
% endfor
</%pyfr:kernel>
