<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>

<%pyfr:kernel name='mortarfluxpatch' ndim='2'
              ul='inout fpdtype_t[${str(nvars)}]'
              ur='in fpdtype_t[${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'>
    fpdtype_t mag_n = sqrt(${pyfr.dot('nl[{i}]', i=ndims)});
    fpdtype_t norm_n[] = ${pyfr.array(
        '(1 / mag_n)*nl[{i}]', i=ndims
    )};
    fpdtype_t fn[${nvars}];
    ${pyfr.expand('rsolve', 'ul', 'ur', 'norm_n', 'fn')};
% for v in range(nvars):
    ul[${v}] = mag_n*fn[${v}];
% endfor
</%pyfr:kernel>
