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
%   A CSV file containing slice file names, associated patient number,
%   slice number, and labels from conf_f Labels file
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
%   Mar 1, 2021 - updated to include all label columns in options.Labels in
%                 output
%               - changed CSV header to be a variable in conf_f

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
    % slice2pat connects slices back to patients
    [ut,~,slice2pat] = unique(utfiles, 'stable');
    % Get number of slices for each patient
    num_slices_per_pat = hist(slice2pat,unique(slice2pat));

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
    % rfs code for patients with a confirmed label
    rl_patient = num(pwl_idx,:);
    % withlabel_slices
    lbl_4_slices = [];

    for i=1:length(rl_patient)
        recurrence_lbl_for_slices = rl_patient(i,:)' .* ones(size(num,2),num_slices_per_pat(ut_idx(i)));
%         time_lbl_for_slices = num(ut_idx(i),2) * ones(1,num_slices_per_pat(ut_idx(i)));
        lbl_4_slices = [lbl_4_slices; recurrence_lbl_for_slices'];
%         time_labels = [time_labels; time_lbl_for_slices'];
    end

    % Setting up and outputing CSV file
    header = options.CSV_header;
%     header = {'File', 'ID', 'Slice', options.LabelType};
    data = [cellstr(otfiles), num2cell(slice2pat), cellstr(tfiles), num2cell(lbl_4_slices(:,1)), num2cell(lbl_4_slices(:,2))];
    writetable(cell2table([header;data]),output_fname,'writevariablenames',0)
end