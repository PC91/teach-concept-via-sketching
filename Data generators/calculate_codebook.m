function [centroid] = ...
    calculate_codebook(train_data_dir, fea_dir, n_mean)

addpath kmeans;

train_len = [];
train_mat = [];
train_label = [];
sift_feature = [];

num_per_img = 40; %round(num_smp/num_img);
dimFea = 128;

subfolders = dir(train_data_dir);

for ii = 1:length(subfolders),
    subname = subfolders(ii).name;
    
    if ~strcmp(subname, '.') & ~strcmp(subname, '..'),       
        frames = dir(fullfile(train_data_dir, subname, '*.mat'));
        c_num = length(frames);
        
        for jj = 1:c_num,
            c_path = fullfile(train_data_dir, subname, frames(jj).name);
            load(c_path);
            
            num_fea = size(feaSet.feaArr, 2);
            rndidx = randperm(num_fea);
            min_num = min(num_fea, num_per_img);
            %X(:, cnt+1:cnt+num_per_img) = feaSet.feaArr(:, rndidx(1:num_per_img));
    
            train_label = [train_label; ii - 2];
            train_len = [train_len; min_num];
            sift_feature = [sift_feature, feaSet.feaArr(:, rndidx(1:min_num))];
        end;    
    end;
end;

sift_feature = sift_feature';

options = zeros(1,14);
options(1) = 1; % display
options(2) = 1;
options(3) = 0.1; % precision
options(5) = 1; % initialization
options(14) = 100; % maximum iterations

centers = zeros(n_mean, size(sift_feature,2));

%% run kmeans
fprintf('\nRunning k-means\n');
centroid = sp_kmeans(centers, sift_feature, options);

fprintf('Finish k-means\n');

% save('dictionary/flickr_logos_27_SIFT_train.mat', 'sift_feature');
% 
% [idx, centroid] = kmeans(sift_feature, n_mean);

B = centroid';

if ~isdir('dictionary'),
    mkdir('dictionary');
end;

save('dictionary/flickr_logos_27_SIFT_Kmeans_800.mat', 'B');

disp('Kmeans is done!');