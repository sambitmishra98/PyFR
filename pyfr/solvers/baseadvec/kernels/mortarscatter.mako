<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='mortarscatter' ndim='1'
              bc='in fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              bf0='in fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              bf1='in fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              uc='out view fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              uf0='out view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              uf1='out view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'>
% for j in range(ncfpts):
% for v in range(nvars):
    uc[${j}][${v}] = bc[${j}][${v}];
% endfor
% endfor
% for j in range(ntfpts):
% for v in range(nvars):
    uf0[${j}][${v}] = bf0[${j}][${v}];
    uf1[${j}][${v}] = bf1[${j}][${v}];
% endfor
% endfor
</%pyfr:kernel>
