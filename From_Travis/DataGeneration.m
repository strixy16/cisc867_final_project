%% code to pre process data

% Move to directory with mhd files

% cd('\\penssur2\SimpsonLab\Projects\Abhishek_Midya\Liver_Tumor_Classification_Expanded\Benign\');
% cd('\\penssur2\SimpsonLab\Projects\Abhishek_Midya\Liver_Tumor_Classification_Expanded\HCC\');
% cd('\\penssur2\SimpsonLab\Projects\Abhishek_Midya\Liver_Tumor_Classification_Expanded\CRLM\');
% cd('G:\Abhishek\Research\Deep learning\Data\Abhishek\miccai_data\training\');
% **THIS ONE** cd('\\penssur2\SimpsonLab\ProjectData\CRLM_HGP\CRLM_HGP\tumor');
% cd('K:\Surgery\Data\ResearchPrograms\Hepatobiliary\Projects\Fellows Projects\Abhishek Midya\Combined_HCC_ICC\AllData\');
% cd('\\penssur2\SimpsonLab\Projects\Jayasree_Chakraborty\CRLM_HGP');
% cd('K:\Surgery\Data\ResearchPrograms\Hepatobiliary\Projects\Fellows Projects\Jayasree Chakraborty\imaging_data\HCC\WashU\tumors\');
%  cd('\\penssur2\SimpsonLab\Projects\Abhishek_Midya\Liver_Tumor_Classification\ICC\ICCNew\tumors\');
%  cd('\\penssur2\SimpsonLab\Projects\Abhishek_Midya\Liver_Tumor_Classification\ICC\ICCRecurrence\tumor\');
%  cd('\\penssur2\SimpsonLab\Projects\Abhishek_Midya\Cholango-13-066\Tumor\');
% cd('\\penssur2\SimpsonLab\Projects\Abhishek_Midya\HassanAllData\');
cd('/Users/katyscott/Documents/ICC/Data/cholangio/MSK/tumor');

% basedirs=dir('*segmented*.mhd');
% basedirs=dir('*Tumor*.mhd');
% basedirs=dir('*hcc*.mhd');
% basedirs_hcc1=dir('*hcctumor1*.mhd');
% basedirs_hcc2=dir('*hcc_Tumor.mhd');
% basedirs=[basedirs_hcc1; basedirs_hcc2];
% basedirs_hcc =dir('*hcc*.mhd');
% basedirs =dir('*icc*.mhd');
% basedirs =dir('*_recurrence*');
basedirs =dir('*Tumor*.mhd');
% basedirs=dir('*hcctumor1*.mhd');
% basedirs=dir('*_rec*');

cd('/Users/katyscott/Documents/ICC/Code');
% cd('G:\My Documents\Simpson Lab\Code');
%% Reading MHD file
%  ImageLoc='\\penssur\SimpsonLab\FellowProjects\Abhishek_Midya\Liver_Tumor_Classification_Expanded\Benign\';
% ImageLoc='\\penssur2\SimpsonLab\Projects\Abhishek_Midya\Liver_Tumor_Classification_Expanded\HCC\';
% ImageLoc='K:\Surgery\Data\ResearchPrograms\Hepatobiliary\Projects\Fellows Projects\Abhishek Midya\Combined_HCC_ICC\AllData\';
% **THIS ONE** ImageLoc='\\penssur2\SimpsonLab\ProjectData\CRLM_HGP\CRLM_HGP\tumor\';
% ImageLoc='K:\Surgery\Data\ResearchPrograms\Hepatobiliary\Projects\Fellows Projects\Jayasree Chakraborty\imaging_data\HCC\WashU\tumors\';
% ImageLoc='G:\Abhishek\Research\Deep learning\Data\Abhishek\miccai_data\training\';
% ImageLoc1='G:\Abhishek\Research\Deep learning\Data\Abhishek\miccai_data\testing\';
% ImageLoc='\\penssur2\SimpsonLab\Projects\Abhishek_Midya\Liver_Tumor_Classification_Expanded\CRLM\';
% ImageLoc='\\penssur2\SimpsonLab\Projects\Abhishek_Midya\Liver_Tumor_Classification\ICC\ICCNew\tumors\';
% ImageLoc='\\penssur2\SimpsonLab\Projects\Abhishek_Midya\Liver_Tumor_Classification\ICC\ICCRecurrence\tumor\';
% ImageLoc='\\penssur2\SimpsonLab\Projects\Abhishek_Midya\Liver_Tumor_Classification\Benign\tumors\';
% ImageLoc='\\penssur2\SimpsonLab\Projects\Abhishek_Midya\HassanAllData\';
ImageLoc = "/Users/katyscott/Documents/ICC/Data/cholangio/MSK/tumor/";

nData=size(basedirs ,1);
MaxHeight=0;
MaxWidth=0;
for i=1:nData
%      filename = strcat(ImageLoc,basedirs(i).name,'\',basedirs(i).name,'_Tumor.mhd');
     'Computing Size for '
     i
    filename=strcat(ImageLoc,basedirs(i).name);
    info = mha_read_header(filename);
    I = double(mha_read_volume(info)); 
    I = double(ProcessImage(I));
    [nRows, nCols, nSlice]=size(I);
    for currSlice=1:nSlice
        Image= I(:,:,currSlice);
        if  ~isnan(nanstd(Image(:))) 
            [rows,cols]=find(~isnan(Image));
            tempHeight=max(rows)-min(rows);
            store_ht(i,currSlice)=tempHeight;
            tempWidth=max(cols)-min(cols);
            store_wd(i,currSlice)=tempWidth;
            if tempHeight>MaxHeight
                MaxHeight=tempHeight;
            end
             if tempWidth>MaxWidth
                MaxWidth=tempWidth;
             end  
            clear rows cols;
            
        end
    end
end

% MaxHeight=217;
% MaxWidth=263;


ImageIdxTr=0; % K: what is this for?
for i=1:nData
    i
%     filename = strcat(ImageLoc,basedirs(i).name,'\',basedirs(i).name,'_Tumor.mhd');
    filename=strcat(ImageLoc,basedirs(i).name);
    info = mha_read_header(filename);
    I = double(mha_read_volume(info));
    I = double(ProcessImage(I));
%    <you can do further pre processing here>
    [nRows, nCols, nSlice]=size(I);
    
    sliceId=0;
    for currSlice=1:nSlice
        Image= I(:,:,currSlice);
        if  ~isnan(nanstd(Image(:))) 

            [rows,cols]=find(~isnan(Image));
            RowCtr=round(sum(rows)/length(rows));
            ColCtr=round(sum(cols)/length(cols));
            StartRow=round(RowCtr-MaxHeight/2);
            StartCol=round(ColCtr-MaxWidth/2);
            if StartRow<=0
                StartRow=1;
            end
            if StartCol<=0
                StartCol=1;
            end 
            if StartRow+MaxHeight>nRows
                StartRow=StartRow-(nRows-MaxHeight);
            end
            if  StartCol+MaxWidth>nCols
                StartCol=StartCol-(nCols-MaxWidth);
            end

            ImageCr=imcrop(Image,[StartCol StartRow, MaxWidth MaxHeight]);
            % Resized for inception 
            ImageCrR= imresize(ImageCr,[299 299]); 
            figure(1);
            imshow(ImageCrR,[])
            ImageCrRZ=ImageCrR;
            ImageCrRZ(isnan(ImageCrRZ))=0;
            sliceId=sliceId+1;
            ImageIdxTr=ImageIdxTr+1;
%             Data(1,1,1:299,1:299)=ImageCrR;
%             Data(1,2,1:299,1:299)=ImageCrR;
%             Data(1,3,1:299,1:299)=ImageCrR;
            if ~isempty(findstr(basedirs(i).name,'hcc'))
                Label=1; %% HCC class
            elseif ~isempty(findstr(basedirs(i).name,'ICC'))
                    Label=0; %% ICC class
            elseif ~isempty(findstr(basedirs(i).name,'Benign'))
                Label=3;
            else
                Label=2;
            end
            BinfileName=strcat(basedirs(i).name, 'Slice_',num2str(sliceId), '.bin');
            fileID2 = fopen(strcat(ImageLoc,'NaN\',BinfileName),'w');
            fwrite(fileID2,ImageCrR,'double');
            fclose(fileID2);           
            fileID1 = fopen(strcat(ImageLoc,'Zero\',BinfileName),'w');
            fwrite(fileID1,ImageCrRZ,'double');
            fclose(fileID1);            
            
            clear ImageCr ImageCrR rows cols;
        end

    end   
    
end



