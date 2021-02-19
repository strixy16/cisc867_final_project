function createCSV(conf_f, background)
% Name: createCSV
% Description: Function to generate CSV file to correspond patients with images and
% slices from preprocessMHA
%
% INPUT: 
%   conf_f       -- configuration file for different datasets
%   background   -- either "zeros" or "nans"
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
%   Feb 15/16 2021 - added more commenting
%                  - changed some variable names to be more readable
%                  - replaced excelfile and output_fname with configure

    % Getting variables from configuration file
    if ischar(conf_f)
        conf_f = str2func(conf_f);
        options = conf_f();
    else
        options = conf_f;
    end
    
    if background == "nans"
        % use location of bin files with nans in background
        bin_dir = options.NaNLoc;
        output_fname = options.NaNCSV;
    else
        % use location of bin files with zeros in background
        if background ~= "zeros"
            disp("Incorrect input for background, using zeros.")
        end
        bin_dir = options.ZeroLoc;
        output_fname = options.ZeroCSV;
    end
        
    % Get list of all bin files
    bin_files = dir(fullfile(bin_dir, '*.bin'));
    % Have folder as structure, change to table for stuff later on
    tfiles_allinfo = struct2table(bin_files);
    % Extract file names and sort names alphanumerically, now a cell array
    tfiles = natsort(tfiles_allinfo{:,'name'});
    % List of original slice file names 
    otfiles = tfiles;
    % For storing unique slice file names - list of patients, repeated # of times
    % there are slices for that patient
    utfiles = tfiles;

    % Parsing through filename to get slice index
    % Every unique name becomes a number 
    for i=1:length(tfiles)
        % Index where .bin starts in filename i
        bidx = strfind(tfiles{i},'.bin');
        % Underscore index
        uidx = strfind(tfiles{i},'_');
        % Getting last index of file name without _Tumor_Slice_#.bin
        nidx = uidx(end-2);
        % Extract patient label out of file name
        utfiles{i}(nidx:end) = '';
        % Get just slice number
        tfiles{i}(bidx:end) = '';
        tfiles{i}(1:uidx(end)) = '';
    end
    % Get unique file names without resorting (need order to correspond
    % back to slices)
    % didx connects slices back to patients
    [ut,~,slice2pat] = unique(utfiles, 'stable');
    % Get number of slices for each patient
    count = hist(slice2pat,unique(slice2pat));

    % Read in label spreadsheet 
    % if replace ~ get raw excel file as a cell array
    % num will capture both columns of numbers of RFS 
    % num = [RFS codes (1 or 0), RFS time (months?)]
    % txt = [header; patient_ids empty empty]
    [num,txt,~] = xlsread(options.Labels) ;

    % get just the patient ids from label spreadsheet
    pats_with_labels = txt(2:end,1);
    % match up these names with the unique patient ids that we have slices
    % for
    [~, ut_idx, pwl_idx] = intersect(ut, pats_with_labels);
    asc_num = (1:length(ut))';
    % finding which indices do/don't have label
    loc = ismember(asc_num, ut_idx);
    % patient index with no label in the slice list
    % nolabel_patient
    nl_patient = find(loc==0);

    % Finding slice indices with no label to drop from output
    % nolabel_slices
    nl_slices = [];
    for i=1:length(nl_patient)
        % nl_patient contains what patient index the slices belong to
        % zidx is patient indices with no labels
        nl_pat_slice_idx = find(slice2pat==nl_patient(i));
        nl_slices = [nl_slices; nl_pat_slice_idx];
    end

    % Removing files with no label from 
    % Slice file name list, slice2pat index, and slices matrix
    otfiles(nl_slices) = [];
    slice2pat(nl_slices) = [];
    tfiles(nl_slices) = [];

    % get rows from label spreadsheet for the corresponding patients in ut
    % withlabel_patient
    wl_patient = num(pwl_idx);
    % withlabel_slices
    wl_slices = [];
    for i=1:length(wl_patient)
        pat_wl_slice_idx = wl_patient(i) * ones(1,count(ut_idx(i)));
        wl_slices = [wl_slices; pat_wl_slice_idx'];
    end

    % Setting up and outputing CSV file
    header = {'File', 'ID', 'Slice', options.LabelType};
    data = [cellstr(otfiles), num2cell(slice2pat), cellstr(tfiles), num2cell(wl_slices)];
    writetable(cell2table([header;data]),output_fname,'writevariablenames',0)
end