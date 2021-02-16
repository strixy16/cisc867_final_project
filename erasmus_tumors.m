function opt = erasmus_tumors
%Description: Configuration file for preprocessMHA for Erasmus tumor
%             image set
%
%OUTPUT: opt - struct containing variables defined and described below
%
%Environment: MATLAB R2020b
%Notes: 
%Author: Katy Scott
%Created: 14 Jan 2021
%Updates:
% Feb 16, 2021 - updated comments, added OutputCSV for createCSV

    % Location of image files for tumour image set, for use in preprocessMHA
    opt.ImageLoc = "/Users/katyscott/Documents/ICC/Data/cholangio/Erasmus/tumors/";
    % Location of bin folder containing tumour image slice set, for use in
    % preprocess MHA and createCSV
    % NOTE: this has subfolders NaN and zero in it 
    opt.BinLoc = "/Users/katyscott/Documents/ICC/Data/cholangio/Erasmus/bin_tumors/";
    
    % Spreadsheet of labels, excel file, for use in createCSV.m
    opt.Labels = "/Users/katyscott/Documents/ICC/Data/RFS_Scout.xlsx";
    % File name + location to output in createCSV.m
    opt.OutputCSV = "/Users/katyscott/Documents/ICC/Data/Labels/Erasmus_RSF.csv";
end