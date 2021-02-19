function preprocessMHA(conf_f)
%Name: preprocessMHA.m
%Description: Code to preprocess Insight Meta-Image imaging data for ICC project. Generated
%             based on DataGeneration.m, recieved from Travis Williams
%             2020.
%
%INPUT: conf_f: configuration file for certain variables 
%OUTPUT: bin files of images
%Environment: MATLAB R2020b
%Notes: 
%Author: Katy Scott
%Created: 14 Jan 2021
%Updates: 
%  19 Feb 2021 - changed image resizing to be input in conf_f
    
    
    % Getting variables from configuration file
    if ischar(conf_f)
        conf_f = str2func(conf_f);
        options = conf_f();
    else
        options = conf_f;
    end
    % TEMP LINE FOR FUNCTION TESTING
%     options = msk_tumor();
    
    % Getting list of MHD files
    baseDirs = dir(strcat(options.ImageLoc, "*Tumor*.mhd"));
    
    % Counter for loop
    nData = size(baseDirs, 1);
    % Recording max height and width for crop/rescaling
    maxHeight = 0;
    maxWidth = 0;
    
    procImages = cell(nData, 1);
    
    % Reading each individual MHD file to find the max height and width
    % from the images
    % Iterating through each MHD file in the directory
    for currFile=1:nData
        fprintf('Computing Size for %i \n', currFile)
        filename = strcat(options.ImageLoc,baseDirs(currFile).name);
        info = mha_read_header(filename);
        vol = double(mha_read_volume(info));
        vol = double(ProcessImage(vol));
        procImages{currFile} = vol;
        [~, ~, nSlice] = size(vol);
%         imshow(vol(:,:,29))
        
        % Iterating through each slice of the volume
        for currSlice=1:nSlice
            slice = vol(:,:,currSlice);
            % Checking if standard deviation is NaN (image loaded in
            % properly?)
            if ~isnan(std(slice(:),'omitnan'))
                % From first for loop in DataGeneration
                [rows,cols] = find(~isnan(slice));
                tempHeight = max(rows)-min(rows);
%                 store_ht(currFile,currSlice) = tempHeight;
                tempWidth = max(cols)-min(cols);
%                 store_wd(currFile,currSlice) = tempWidth;
                % Update maxHeight
                if tempHeight > maxHeight
                    maxHeight = tempHeight;
                end
                % Update maxWidth
                if tempWidth > maxWidth
                    maxWidth = tempWidth;
                end
                clear rows cols;
            end % end if
        end % end slice loop
    end % end file loop
    
    % Second for loop to crop images based on max and minHeight
    
    for currFile=1:nData
        fprintf('Cropping images for %i \n', currFile)
        vol = procImages{currFile};
        [nRows, nCols, nSlice] = size(vol);
        sliceID = 0; % part of naming of binfile
        % Iterating through each slice of the volume
        for currSlice=1:nSlice
            slice = vol(:,:,currSlice);
            if ~isnan(std(slice(:),'omitnan'))
                [rows,cols] = find(~isnan(slice));
                rowCtr = round(sum(rows)/length(rows));
                colCtr = round(sum(cols)/length(cols));
                startRow = round(rowCtr-maxHeight/2);
                startCol = round(colCtr-maxWidth/2);
                if startRow <= 0
                    startRow = 1;
                end
                if startCol <= 0
                    startCol = 1;
                end
                if startRow + maxHeight > nRows
                    startRow = startRow - (nRows - maxHeight);
                end
                if startCol + maxWidth > nCols
                    startCol = startCol - (nCols - maxWidth);
                end
                
                % cropping image based on maximum height and width of all
                % images
                imageCr = imcrop(slice, [startCol startRow, maxWidth maxHeight]);
                
                % resizing for Inception currently
                % TODO: make this an input argument
                imageCrR = imresize(imageCr, options.ImageSize);
%                 figure();
%                 imshow(imageCrR, []);
                
                % replace nans with 0s in image
                imageCrRZ = imageCrR;
                imageCrRZ(isnan(imageCrRZ)) = 0;
                sliceID = sliceID + 1;
                
                % Removed class labelling, using only ICC class
                binFileName = strcat(baseDirs(currFile).name(1:end-4), '_Slice_', num2str(sliceID), '.bin');
                
                % Save image version with NaNs
                nanFileID = fopen(strcat(options.BinLoc, 'NaN/', binFileName), 'w');
                fwrite(nanFileID, imageCrR, 'double');
                fclose(nanFileID);
                
                % Save image version with zeros instead of NaNs
                zeroFileID = fopen(strcat(options.BinLoc, 'Zero/', binFileName), 'w');
                fwrite(zeroFileID, imageCrRZ, 'double');
                fclose(zeroFileID);
                
                clear imageCr imageCrR imageCrRZ rows cols;
                
            end % end if
        end % end slice loop
    end % end file loop
end


