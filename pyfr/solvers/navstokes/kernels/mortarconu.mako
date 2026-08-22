<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<% beta = c['ldg-beta'] %>

<%pyfr:kernel name='mortarconu' ndim='1'
              ucoarse='in view fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              ufine0='in view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              ufine1='in view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              commcoarse='out view fpdtype_t[${str(ncfpts)}][${str(nvars)}]'
              commfine0='out view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              commfine1='out view fpdtype_t[${str(ntfpts)}][${str(nvars)}]'
              ic0='in fpdtype_t[${str(nmpts)}][${str(ncfpts)}]'
              ic1='in fpdtype_t[${str(nmpts)}][${str(ncfpts)}]'
              if0='in fpdtype_t[${str(nmpts)}][${str(ntfpts)}]'
              if1='in fpdtype_t[${str(nmpts)}][${str(ntfpts)}]'
              pc0='in fpdtype_t[${str(ncfpts)}][${str(nmpts)}]'
              pc1='in fpdtype_t[${str(ncfpts)}][${str(nmpts)}]'
              pf0='in fpdtype_t[${str(ntfpts)}][${str(nmpts)}]'
              pf1='in fpdtype_t[${str(ntfpts)}][${str(nmpts)}]'>
    fpdtype_t uc[${ncfpts}][${nvars}] = {};
    fpdtype_t uf0[${ntfpts}][${nvars}] = {};
    fpdtype_t uf1[${ntfpts}][${nvars}] = {};

% for child in range(2):
    for (int q = 0; q < ${nmpts}; q++)
    {
        fpdtype_t ul[${nvars}] = {};
        fpdtype_t ur[${nvars}] = {};
        fpdtype_t ucomm[${nvars}];

        for (int j = 0; j < ${ncfpts}; j++)
        {
        % for v in range(nvars):
            ul[${v}] += ic${child}[q][j]*ucoarse[j][${v}];
        % endfor
        }
        for (int j = 0; j < ${ntfpts}; j++)
        {
        % for v in range(nvars):
            ur[${v}] += if${child}[q][j]*ufine${child}[j][${v}];
        % endfor
        }

    % for v in range(nvars):
        ucomm[${v}] = ${0.5 - beta}*ul[${v}]
                    + ${0.5 + beta}*ur[${v}];
    % endfor

        for (int j = 0; j < ${ncfpts}; j++)
        {
        % for v in range(nvars):
            uc[j][${v}] += pc${child}[j][q]*ucomm[${v}];
        % endfor
        }
        for (int j = 0; j < ${ntfpts}; j++)
        {
        % for v in range(nvars):
            uf${child}[j][${v}] += pf${child}[j][q]*ucomm[${v}];
        % endfor
        }
    }
% endfor

% for j in range(ncfpts):
% for v in range(nvars):
    commcoarse[${j}][${v}] = uc[${j}][${v}];
% endfor
% endfor
% for j in range(ntfpts):
% for v in range(nvars):
    commfine0[${j}][${v}] = uf0[${j}][${v}];
    commfine1[${j}][${v}] = uf1[${j}][${v}];
% endfor
% endfor
</%pyfr:kernel>
