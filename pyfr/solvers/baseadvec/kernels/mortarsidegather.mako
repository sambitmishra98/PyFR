<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='mortarsidegather' ndim='1'
              u='in view fpdtype_t[${str(nfpts)}][${str(nvars)}]'
              b='out fpdtype_t[${str(nfpts)}][${str(nvars)}]'>
% for j in range(nfpts):
% for v in range(nvars):
    b[${j}][${v}] = u[${j}][${v}];
% endfor
% endfor
</%pyfr:kernel>
