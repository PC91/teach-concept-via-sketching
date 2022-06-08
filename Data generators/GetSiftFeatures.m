clear all; close all; clc;

img_dir = 'E:\Cloud\Copy\Sketch Recognition\Image\images_32x32_classified';                                    % directory for the image database                             
data_dir = 'E:\Projects\MATLAB\SketchReg-Hai-Huy\SOURCE CODE\Experiments\DataGenerators\SIFT features';        % directory for saving SIFT descriptors

%extr_sift(img_dir, data_dir);
calculate_codebook(data_dir, 'dfsd', 100);