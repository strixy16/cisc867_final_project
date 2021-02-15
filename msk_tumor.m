function opt = msk_tumor
%Description: Configuration file for preprocessMHA for MSK tumor
%             image set
%
%OUTPUT: opt - struct containing variables defined here
%
%Environment: MATLAB R2020b
%Notes: 
%Author: Katy Scott
%Created: 14 Jan 2020

    % Location of image files for Erasmus tumour set
    opt.ImageLoc = "/Users/katyscott/Documents/ICC/Data/cholangio/MSK/tumor/";
    opt.BinLoc = "/Users/katyscott/Documents/ICC/Data/cholangio/MSK/bin_tumor/";
    
    % Spreadsheet of labels, excel file
    opt.Labels = "/Users/katyscott/Documents/ICC/Data/RFS_Scout.xlsx";
end