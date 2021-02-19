function opt = all_tumors
%Description: Configuration file for createCSV for both MSK and Erasmus
%tumor images
%
%OUTPUT: opt - struct containing variables defined here
%
%Environment: MATLAB R2020b
%Notes: 
%Author: Katy Scott
%Created: 19 Jan 2021
%Updates:

    % Location of bin folder containing tumour image slice set with zeros
    % in background for use in createCSV
    opt.ZeroLoc = "/Users/katyscott/Documents/ICC/Data/Images/Tumors/Zero/";
    % Location of bin folder containing tumour image slice set with NaNs
    % in background for use in createCSV
    opt.NaNLoc = "/Users/katyscott/Documents/ICC/Data/Images/Tumors/NaN/";
    
    % Spreadsheet of labels, excel file, for use in createCSV.m
    opt.Labels = "/Users/katyscott/Documents/ICC/Data/RFS_Scout.xlsx";
    % Heading
    opt.LabelType = 'RFS';
    % File name + location to output in createCSV.m
    opt.ZeroCSV = "/Users/katyscott/Documents/ICC/Data/Labels/RFS_all_tumors_zero.csv";
    opt.NaNCSV = "/Users/katyscott/Documents/ICC/Data/Labels/RFS_all_tumors_NaN.csv";
end