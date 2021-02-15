function createCSV(excelfile,output_fname, conf_f)
% Name: createCSV
% Description: Function to generate CSV file to correspond patients with images and
% slices from preprocessMHA
%
% INPUT: 
%   excelfile    -- spreadsheet with labels
%   output_fname -- name to give CSV file output
%   conf_f       -- configuration file for different datasets
%
% OUTPUT:
%   A CSV file named output_fname with FILL IN WHAT IT IS HERE
%
% Environment: MATLAB R2020b
% Author: Travis Williams
% Updated by: Katy Scott
% Notes: 
%   Feb 6, 2021 - added function description and more commenting
%               - added conf_f input to use configuration files

    % Getting variables from configuration file
    if ischar(conf_f)
        conf_f = str2func(conf_f);
        options = conf_f();
    else
        options = conf_f;
    end
    
    % Get location of bin files with zeros in background
    bin_dir = strcat(options.BinLoc, 'Zero/');
    % Bring up folder selection box to find bin files
%     d = uigetdir(pwd, 'Select a folder');
    % Get list of all bin files
    bin_files = dir(fullfile(bin_dir, '*.bin'));
    % Have folder as structure, change to table for stuff later on
    tfiles_allinfo = struct2table(bin_files);
    % Extract file names and sort names alphanumerically, now a cell array
    tfiles = natsort(tfiles_allinfo{:,'name'});
    % Make two copies of file names for use later ? 
    otfiles = tfiles;
    % For storing unique file names 
    utfiles = tfiles;

    % Parsing through filename to get slice index
    % Every unique name becomes a number 
    for i=1:length(tfiles)
        % Index where .bin starts in filename i
        bidx = strfind(tfiles{i},'.bin');
        % Underscore index
        uidx = strfind(tfiles{i},'_');
        % Getting last index of file name without _Slice_#.bin
        nidx = uidx(end-1);
        % Extract patient label out of file name
        utfiles{i}(nidx:end) = '';
        % Get just slice number
        tfiles{i}(bidx:end) = '';
        tfiles{i}(1:uidx(end)) = '';
    end
    % Get unique file names without resorting (need order to correspond
    % back to slices?)
    % didx connects slices back to patients
    [ut,~,didx] = unique(utfiles, 'stable');
    % Get number of slices for each patient
    count = hist(didx,unique(didx));

    % Reading in HGP excel file -- replace with RFS_Scout
    % if replace ~ get raw excel file as a cell array
    % num will capture both columns of numbers of RFS 
    % TODO: 
    [num,txt,~] = xlsread(excelfile) ;

    %TODO: figure out what happens from here onward
    gtxt = txt(2:end,1);
    [~, ia, ib] = intersect(ut, gtxt);
    asc_num = (1:length(ut))';
    loc = ismember(asc_num, ia);
    zidx = find(loc==0);

    tmp = [];
    for i=1:length(zidx)
        temp = find(didx==zidx(i));
        tmp = [tmp; temp];
    end

    otfiles(tmp) = [];
    didx(tmp) = [];
    tfiles(tmp) = [];

    num_sort = num(ib);
    tmp2 = [];
    for i=1:length(num_sort)
        temp = num_sort(i) * ones(1,count(ia(i)));
        tmp2 = [tmp2; temp'];
    end

    % HGP = histologic growth pattern (0 or 1)
    header = {'File', 'ID', 'Slice', 'HGP'};
    data = [cellstr(otfiles), num2cell(didx), cellstr(tfiles), num2cell(tmp2)];
    writetable(cell2table([header;data]),output_fname,'writevariablenames',0)
end