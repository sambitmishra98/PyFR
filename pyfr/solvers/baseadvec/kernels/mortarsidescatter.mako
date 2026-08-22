<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='mortarsidescatter' ndim='1'
              b='in fpdtype_t[${str(nfpts)}][${str(nvars)}]'
              u='out view fpdtype_t[${str(nfpts)}][${str(nvars)}]'>
% for j in range(nfpts):
% for v in range(nvars):
    u[${j}][${v}] = b[${j}][${v}];
% endfor
% endfor
</%pyfr:kernel>
