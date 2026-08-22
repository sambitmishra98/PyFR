<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<% beta = c['ldg-beta'] %>

<%pyfr:kernel name='mortarconustaged' ndim='2'
              ul0='in fpdtype_t[${str(nvars)}]'
              ur0='in fpdtype_t[${str(nvars)}]'
              ul1='in fpdtype_t[${str(nvars)}]'
              ur1='in fpdtype_t[${str(nvars)}]'
              cu0='out fpdtype_t[${str(nvars)}]'
              cu1='out fpdtype_t[${str(nvars)}]'>
% for child in range(2):
% for v in range(nvars):
    cu${child}[${v}] = ${0.5 - beta}*ul${child}[${v}]
                       + ${0.5 + beta}*ur${child}[${v}];
% endfor
% endfor
</%pyfr:kernel>
