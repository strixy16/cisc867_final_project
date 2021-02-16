function [out] = ProcessImage(I)
    % Convert image to double
    I = double(I);
    % Generate mask excluding values outside of this range (Hounsfield
    % units)
    % This range includes only soft tissue
    Imask = I<-100|I>300;
    % Flips black and white in the mask
    ImaskInv = imcomplement(Imask);
    Inew = I .* ImaskInv;
    Inew(Inew==0) = NaN;
    out = Inew;
end