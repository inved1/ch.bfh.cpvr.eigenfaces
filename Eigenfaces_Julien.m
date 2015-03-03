clear; clc;
close all;

scaleFactor = 1.3;

img = imread('Images/cpvr_classes/2014HS/02.JPG');
%imshow(img); figure;

faceDetector = vision.CascadeObjectDetector('ClassificationModel', 'FrontalFaceCART');

bbox = step(faceDetector, img);

%imgBB = insertObjectAnnotation(img,'rectangle',bbox,'Face');
%imshow(imgBB);

for i = 1:length(bbox)
    
    row = bbox(i,:);
    
    oriValue = row(3);
    resizeValue = row(3) * scaleFactor;
    
    row(1) = row(1) - (resizeValue - oriValue) / 2;
    row(2) = row(2) - (resizeValue - oriValue) / 2;
    row(3) = row(3) * scaleFactor;
    row(4) = row(4) * scaleFactor;
    
    bbox(i,:) = row;
    
    % Crop Face
    croppedImg = imcrop(img,row);
    
    % Reshape Face
    croppedImg = imresize(croppedImg, [160 120]);
    
    % Save Face
    facesImgs{i} = croppedImg;
    
    % Save Vector for PCA
    faces(:,i) = facesImgs{i}(:);
    
    %imshow(croppedImg); figure;
end

imgBB = insertObjectAnnotation(img,'rectangle',bbox,'Face');
imshow(imgBB);

% Do PCA

