%Lay cac codebook ra
img_dir = 'features/flickr_logos_27/LLC_test';
folders = dir(img_dir);
folders(1:2,:) = [];
fileID = fopen('features/flickr_logos_27/LLC_test_Codebook_512.txt','w');
count =0;
for i=1:size(folders,1)
    
    path = strcat(img_dir,'/',folders(i).name);
    files = dir(strcat(path,'/*.mat'));
    for ii=1:size(files,1)
        count = count+1;
        fea_path = strcat(path,'/',files(ii).name);
        load(fea_path);
        for k=1:size(fea,1)
            fprintf(fileID,'%12.8f ',fea(k,1));
        end
        fprintf(fileID,'\t %d \t %d\n',label,count);
        fea_path = [];
    end
    path = [];
    fprintf('process %d/27 \n',i);
end
fclose(fileID);

