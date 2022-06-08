%ghi cac sift vo 1 file txt roi kmeans
data_dir = 'E:\Projects\MATLAB\SketchReg-Hai-Huy\SOURCE CODE\Experiments\DataGenerators\SIFT features';
folders = ls(data_dir);
folders(1:2,:)=[];
sift_data = [];
fileID = fopen('dictionary/flickr_logos_27_SIFT_train.txt','w');
ndivied = 3; 

for i=1:size(folders,1)
    sift_dir = strcat(data_dir,'\',folders(i,:));
    files = dir(strcat(sift_dir,'\*.mat'));
    for ii=1:size(files,1)
        file_path = strcat(sift_dir,'\',files(ii).name);
        load(file_path);
        %sift_data = [sift_data feaSet.feaArr];
        feaSet.feaArr = feaSet.feaArr';
        indices_col = randperm(ceil(size(feaSet.feaArr,1)/ndivied))';
        for r = 1:size(indices_col,1)
            for c=1:size(feaSet.feaArr,2)
                fprintf(fileID,'%12.8f ',feaSet.feaArr(indices_col(r),c));
            end
            fprintf(fileID,'\n');
        end
        file_path = [];
    end
    sift_dir = [];
    fprintf ('process %d/%d\n',i,size(folders,1));
end
fclose(fileID);