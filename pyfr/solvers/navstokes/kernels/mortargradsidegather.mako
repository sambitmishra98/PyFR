<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='mortargradsidegather' ndim='1'
              g='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              b0='out view fpdtype_t[${str(nvars)}]'
              b1='out view fpdtype_t[${str(nvars)}]'
              b2='out view fpdtype_t[${str(nvars)}]'>
% for d in range(ndims):
% for v in range(nvars):
    b${d}[${v}] = g[${d}][${v}];
% endfor
% endfor
</%pyfr:kernel>
