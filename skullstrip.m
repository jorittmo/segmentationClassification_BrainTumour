function f = skullstrip(ims, ero, threshold)
    grayImage = ims;
    
    if nargin < 2
    ero = 15;
    threshold = 0.1;
    end
    
    if nargin < 3
    threshold = 0.1;
    end

    binaryImage = grayImage > threshold;
    % Get rid of small specks of noise
    binaryImage = bwareaopen(binaryImage, 10);
 
%% Seal off the bottom of the head - make the last row white.
     binaryImage(240,:) = 1;
    % Fill the image
    binaryImage = imfill(binaryImage, 'holes');
    %% Erode away 15 layers of pixels.
    se = strel('disk', ero, 0);
    binaryImage = imerode(binaryImage, se);

    %% Mask original image
    finalImage = grayImage; 
    finalImage(~binaryImage) = 0;
    f = finalImage;


end