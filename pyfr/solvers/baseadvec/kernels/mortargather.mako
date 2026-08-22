<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='mortargather' ndim='1'
              uc='in view fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              uf0='in view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              uf1='in view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              bc='out fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              bf0='out fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              bf1='out fpdtype_t[${str(ntfpts)}][${str(nvars)}]'>
% for j in range(ncfpts):
% for v in range(nvars):
    bc[${j}][${v}] = uc[${j}][${v}];
% endfor
% endfor
% for j in range(ntfpts):
% for v in range(nvars):
    bf0[${j}][${v}] = uf0[${j}][${v}];
    bf1[${j}][${v}] = uf1[${j}][${v}];
% endfor
% endfor
</%pyfr:kernel>
