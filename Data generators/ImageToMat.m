clear all; close all; clc;

img_dir = 'E:\Projects\MATLAB\SketchReg-Hai-Huy\SOURCE CODE\Experiments\Data\Image pixels';
folders = dir(img_dir);
folders(1:2,:) = [];
count =0;
data = [];
InputSize = 64*64;
for i=1:size(folders,1)
    path = strcat(img_dir,'\',folders(i).name);
    files = dir(strcat(path,'\*.png'));
    for ii=1:size(files,1)
        count = count+1;
        image_path = strcat(path,'/',files(ii).name);
        img = imread(image_path);
        img = imresize(img, [32 32]);
        bimg = rgb2gray(img);
        bimg = im2double(bimg);
        f = reshape(bimg,InputSize,1);
        f = [f ; i];
        data = [data f];
        image_path = [];
    end
    path = [];
    fprintf('type %d \n',i);
end

sketch_pixel_data = data(1:InputSize, :);
sketch_label = data(InputSize+1, :);
sketch_label = sketch_label';

save('E:\Projects\MATLAB\SketchReg-Hai-Huy\SOURCE CODE\Experiments\Data\Image pixels\Image_Pixels_64x64.mat', 'sketch_pixel_data', 'sketch_label');

