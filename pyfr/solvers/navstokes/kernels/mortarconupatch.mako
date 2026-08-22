<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<% beta = c['ldg-beta'] %>

<%pyfr:kernel name='mortarconupatch' ndim='2'
              ul='in fpdtype_t[${str(nvars)}]'
              ur='in fpdtype_t[${str(nvars)}]'
              cu='out fpdtype_t[${str(nvars)}]'>
% for v in range(nvars):
    cu[${v}] = ${0.5 - beta}*ul[${v}] + ${0.5 + beta}*ur[${v}];
% endfor
</%pyfr:kernel>
