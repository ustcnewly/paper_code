% This make.m is used under Windows

mex -O -largeArrayDims -D_DENSE_REP -c ..\blas\*.c -outdir ..\blas
mex -O -largeArrayDims -D_DENSE_REP -c ..\linear.cpp
mex -O -largeArrayDims -D_DENSE_REP -c ..\tron.cpp
mex -O -largeArrayDims -D_DENSE_REP -c linear_model_matlab.cpp -I..\
mex -O -largeArrayDims -D_DENSE_REP strain_weight.cpp -I..\ tron.obj linear.obj linear_model_matlab.obj ..\blas\*.obj
mex -O -largeArrayDims -D_DENSE_REP spredict.cpp -I..\ tron.obj linear.obj linear_model_matlab.obj ..\blas\*.obj
mex -O -largeArrayDims -D_DENSE_REP libsvmread.c
mex -O -largeArrayDims -D_DENSE_REP libsvmwrite.c
