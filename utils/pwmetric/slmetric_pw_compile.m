function slmetric_pw_compile()
%SLMETRIC_PW_COMPILE Compiles the CPP mex files for slmetric_pw
%
% [ History ]
%   - Created by Dahua Lin, on Aug 16, 2007
%

mdir = fileparts(mfilename('fullpath'));
if ~strcmp(pwd(), mdir)
    cd(mdir);
end

disp('compile pwhamming_cimp.cpp');
mex -O -largeArrayDims pwhamming_cimp.cpp

disp('compile pwmetrics_cimp.cpp');
mex -O -largeArrayDims pwmetrics_cimp.cpp

disp('compile omp_pwmetrics_cimp.cpp');
mex -O -largeArrayDims omp_pwmetrics_cimp.cpp  COMPFLAGS="$COMPFLAGS /openmp" LINKFLAGS="$LINKFLAGS /openmp"

disp('compilation completed.');
