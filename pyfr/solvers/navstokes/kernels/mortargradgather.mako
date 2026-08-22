<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='mortargradgather' ndim='1'
              gc0='in view fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              gc1='in view fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              gc2='in view fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              gf00='in view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              gf01='in view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              gf02='in view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              gf10='in view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              gf11='in view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              gf12='in view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              bc0='out fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              bc1='out fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              bc2='out fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              bf00='out fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              bf01='out fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              bf02='out fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              bf10='out fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              bf11='out fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              bf12='out fpdtype_t[${str(ntfpts)}][${str(nvars)}]'>
% for d in range(ndims):
% for j in range(ncfpts):
% for v in range(nvars):
    bc${d}[${j}][${v}] = gc${d}[${j}][${v}];
% endfor
% endfor
% for j in range(ntfpts):
% for v in range(nvars):
    bf0${d}[${j}][${v}] = gf0${d}[${j}][${v}];
    bf1${d}[${j}][${v}] = gf1${d}[${j}][${v}];
% endfor
% endfor
% endfor
</%pyfr:kernel>
