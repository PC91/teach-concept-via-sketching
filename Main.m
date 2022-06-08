clear all; close all; clc;

im_w = 64;
im_h = 64;
gridSpacing = 4;
patchSize = 16;
% remX = mod(im_w-patchSize,gridSpacing);
% offsetX = floor(remX/2)+1;
% remY = mod(im_h-patchSize,gridSpacing);
% offsetY = floor(remY/2)+1;    
% [gridX,gridY] = meshgrid(offsetX:gridSpacing:im_w-patchSize+1, offsetY:gridSpacing:im_h-patchSize+1);
            

% -------------------------------------------------------------------------
% parameter setting
pyramid = [1, 2, 4];            % spatial block structure for the SPM
knn = 10;                       % number of neighbors for local coding
nRounds = 1;                    % number of random test on the dataset
num_of_train_per_class = 50;              % training examples per category
num_of_test_per_class  = 30;              % testing examples per category
n_mean = 100;                   % num of clusters in k-means

mem_block = 3000;               % maxmum number of testing features loaded each time  
c = 10;                         % regularization parameter for linear SVM in Liblinear package
SIFT_threshold = 5;            % min threshold for a SIFT descriptor to be accepted
% -------------------------------------------------------------------------
% set path
addpath 'CVPR10-LLC';
addpath 'Liblinear\windows';
addpath 'kmeans';
addpath 'sift';

total_dir = 'E:\Cloud\Copy\Sketch Recognition\Image\HCI data';       % directory for the entire data
img_dir = strcat(total_dir, '\Small');                              % directory for the image database                             
data_dir = strcat(total_dir, '\Sift');                               % directory for saving SIFT descriptors

% -------------------------------------------------------------------------
% retrieve the directory of the database and load the codebook
siftDatabase = retr_database_dir(data_dir);

if isempty(siftDatabase),
    error('Data directory error!');
end

fprintf('\n Testing...\n');
clabel = unique(siftDatabase.label);
nclass = length(clabel);
accuracy = zeros(nRounds, 1);
subfolders = dir(data_dir);
subfolders(1:2) = [];

for ii = 1:nRounds,
    fprintf('Round: %d...\n', ii);
    cur_data_dir = strcat(total_dir, '\', int2str(ii));
    mkdir(cur_data_dir);
    
    % -------------------------------------------------------------------------
    % extract sift and find codebook
    train_sift_feature = rand(128, 350000);
    num_of_train_sift_feature = 0;

    num_sift_per_img = 40;

    tr_idx = [];
    ts_idx = [];

    for jj = 1:length(subfolders),
        
        subname = subfolders(jj).name;
        %fprintf('%s\n', subname);
        frames = dir(fullfile(data_dir, subname, '*.mat'));

        idx_label = find(siftDatabase.label == clabel(jj));
        num = length(idx_label);

        idx_rand = randperm(num);

        tr_idx = [tr_idx, idx_label(idx_rand(1:num_of_train_per_class))];
        ts_idx = [ts_idx, idx_label(idx_rand(num_of_train_per_class+1:end))];

        %Save train sift
        %fprintf('%d\n', size(tr_idx, 1));
        for kk = 1:size(tr_idx, 1),
            
            c_path = strcat(data_dir, '\', subname, '\', frames(tr_idx(kk)).name);
            load(c_path);

            num_fea = size(feaSet.feaArr, 2);
            cur_feaSet = feaSet.feaArr;
            for ll = 1:size(cur_feaSet, 2),
                cur_SIFT = cur_feaSet(:, ll);
                sum_Feature = sum(cur_SIFT, 1);
                if(sum_Feature >= SIFT_threshold)
                    train_sift_feature(:, num_of_train_sift_feature + 1) = cur_SIFT;
                    num_of_train_sift_feature = num_of_train_sift_feature + 1;
                end
            end
        end
    end
    train_sift_feature = train_sift_feature';

    train_sift_feature = train_sift_feature(1:num_of_train_sift_feature, :);
    options = zeros(1,14);
    options(1) = 1; % display
    options(2) = 1;
    options(3) = 0.1; % precision
    options(5) = 1; % initialization
    options(14) = 10; % maximum iterations

    centers = zeros(n_mean, size(train_sift_feature,2));

    %run kmeans
    fprintf('\nRunning k-means\n');
    centroid = sp_kmeans(centers, train_sift_feature, options);    

    centroid = centroid';
    save(strcat(cur_data_dir, '\', 'Codebook.mat'), 'centroid');
    
    nCodebook = size(centroid, 2);              % size of the codebook

    
    % -------------------------------------------------------------------------
    % calculate image features

    dFea = sum(nCodebook*pyramid.^2);
    nFea = length(siftDatabase.path);

    fileDatabase = struct;
    fileDatabase.path = cell(nFea, 1);         % path for each image feature
    fileDatabase.label = zeros(nFea, 1);       % class label for each image feature

    for iter1 = 1:nFea,  
        if ~mod(iter1, 5),
           fprintf('.');
        end
        if ~mod(iter1, 100),
            fprintf(' %d images processed\n', iter1);
        end
        fpath = siftDatabase.path{iter1};
        flabel = siftDatabase.label(iter1);

        folderpath = fullfile(cur_data_dir, siftDatabase.cname{flabel});
        %fprintf('%d\n', exist(folderpath, 'dir'));
        if(exist(folderpath, 'dir') == 0)
            mkdir(folderpath);    
        end

        load(fpath);
        [rtpath, fname] = fileparts(fpath);
        feaPath = fullfile(cur_data_dir, siftDatabase.cname{flabel}, [fname '.mat']);


        fea = LLC_pooling(feaSet, centroid, pyramid, knn);
        label = siftDatabase.label(iter1);

        save(feaPath, 'fea', 'label');

        fileDatabase.label(iter1) = flabel;
        fileDatabase.path{iter1} = feaPath;
    end;
    
    indexFile = strcat(cur_data_dir, '\', 'Indices');
    save(indexFile, 'tr_idx', 'ts_idx');
    
    num_train_images = size(tr_idx, 1) * size(tr_idx, 2);
    num_test_images = size(ts_idx, 1) * size(ts_idx, 2);
    fprintf('Training number: %d\n', num_train_images);
    fprintf('Testing number:%d\n', num_test_images);
    
    % load the training features
    tr_fea = zeros(num_train_images, dFea);
    tr_label = zeros(num_train_images, 1);
    
    for jj = 1:nclass,
        for kk = 1:num_of_train_per_class,
            fpath = fileDatabase.path{tr_idx(kk, jj)};
            load(fpath, 'fea', 'label');
            tr_fea((jj - 1) * num_of_train_per_class + kk, :) = fea';
            tr_label((jj - 1) * num_of_train_per_class + kk) = label;
        end
    end

    
    % -------------------------------------------------------------------------
    % evaluate the performance of the image feature using linear SVM
    % we used Liblinear package in this example code

    % train the loaded features
    options = ['-c ' num2str(c)];
    model = train(double(tr_label), sparse(double(tr_fea)), options);
    clear tr_fea;

    % load the testing features
    ts_fea = zeros(num_test_images, dFea);
    ts_label = zeros(num_test_images, 1);

    
    for jj = 1:nclass,
        for kk = 1:num_of_test_per_class,
            fpath = fileDatabase.path{ts_idx(kk, jj)};
            load(fpath, 'fea', 'label');
            ts_fea((jj - 1) * num_of_test_per_class + kk, :) = fea';
            ts_label((jj - 1) * num_of_test_per_class + kk) = label;
        end
    end

    [C] = predict(ts_label, sparse(ts_fea), model);
    
    acc = zeros(nclass, 1);
    %mkdir(strcat('ClassResult/round_',int2str(ii)));
    for jj = 1 : nclass,
        c = clabel(jj);
        idx = find(ts_label == c);
        curr_pred_label = C(idx);
        curr_gnd_label = ts_label(idx);    
        if length(idx) ~=0
            acc(jj) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
        else
            acc(jj) = 0;
        end
        %path = strcat('ClassResult/round_',int2str(ii),'/',database.cname{jj});
        %mkdir(path);
        %indexFile = strcat(cur_data_dir, '\', 'IndexResult');
        right_idx = find(curr_pred_label == curr_gnd_label);
        wrong_idx = find(curr_pred_label ~= curr_gnd_label);
        
        fprintf('%d %d %f\n', right_idx, wrong_idx, length(right_idx) / 30);
        %save(indexFile,'right_idx','wrong_idx');
    end
    
    indexFile = strcat(cur_data_dir, '\', 'Model');
    
    accuracy(ii) = mean(acc); 
    acc_value = accuracy(ii);
    save(indexFile, 'model', 'acc_value');
    fprintf('Classification accuracy for round %d: %f\n', ii, accuracy(ii));
end

fileResult = strcat(total_dir, '\', 'Result');
save(fileResult, 'accuracy');
Ravg = mean(accuracy);                  % average recognition rate
Rstd = std(accuracy);                   % standard deviation of the recognition rate

fprintf('===============================================');
fprintf('Average classification accuracy: %f\n', Ravg);
fprintf('Standard deviation: %f\n', Rstd);    
fprintf('===============================================');
    
