function extr_sift(img_dir, data_dir)
% for example
% img_dir = 'image/Caltech101';
% data_dir = 'data/Caltech101';

addpath('E:\Projects\MATLAB\SketchReg-Hai-Huy\SOURCE CODE\Experiments\Using Locality-constrained Linear Coding\sift');

gridSpacing = 6;
patchSize = 16;
maxImSize = 300;
nrml_threshold = 1;

[database, lenStat] = CalculateSiftDescriptor(img_dir, data_dir, gridSpacing, patchSize, maxImSize, nrml_threshold);