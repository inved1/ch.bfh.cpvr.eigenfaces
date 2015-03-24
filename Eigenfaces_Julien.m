clear; clc;
close all;
clear all;

% Do the PCA
%% Step 1: Load face images & convert each image into a vector of a matrix
k = 0;
for i=0:1:11
    if (i ~= 10)
        for j=1:1:10
            filename  = sprintf('images/cpvr_faces_160/%04d/%02d.JPG',i,j);
            %disp(filename)
            image_data = imread(filename);
            k = k + 1;
            facesDB(:,k) = image_data(:);
         end;
    end
end;
nImages = k;                     %total number of images
imsize = size(image_data);       %size of image (they all should have the same size) 
nPixels = imsize(1)*imsize(2);   %number of pixels in image
facesDB = double(facesDB)/255;

disp(imsize);

% Step 2: Calculate & show the mean image and shift all faces by it
mn = mean(facesDB, 2);
for i=1:nImages
    facesDB(:,i) = facesDB(:,i)-mn;          % substruct the mean
end;
%figure('Color',[1 1 1]); 
%imshow(reshape(mn, imsize)); title('mean face');


%% Step 3: Calculate Eigenvectors & Eigenvalues 
% Method 2: Create covariance matrix faster by using 
% Turk and Pentland's trick to get the eigenvectors of faces*faces' from
% the eigenvectors of faces'*faces
tic;
C = facesDB'*facesDB;
[eigvec,eigval] = eig(C);
eigvec = facesDB * eigvec;                        % Convert eigenvectors back as if they came from A'*A
eigvec = eigvec / (sqrt(abs(eigval)));          % Normalize eigenvectors
% eigvec & eigval are in fact sorted but in the wrong order
eigval = diag(eigval);                          % Get the eigenvalue from the diagonal
eigval = eigval / nImages;                      % Normalize eigenvalues
[eigval, indices] = sort(eigval, 'descend');    % Sort the eigenvalues
eigvec = eigvec(:, indices);                    % Sort the eigenvectors accordingly
toc;

% Display the summed up eigenvalues as it is the summed up variance
normEigval = eigval / sum(eigval);              % Normalize the eigenvalues

facesDB2 = eigvec' * facesDB;
 

%% Load Image with faces to search

img = imread('Images/cpvr_classes/2014HS/05.JPG');
%imshow(img); figure;

faceDetector = vision.CascadeObjectDetector('ClassificationModel', 'FrontalFaceCART');

bbox = step(faceDetector, img);

%imgBB = insertObjectAnnotation(img,'rectangle',bbox,'Face');
%imshow(imgBB); figure;

disp(length(bbox));

imgBB = insertObjectAnnotation(img,'rectangle',bbox,'Face');
imshow(imgBB);

%pause;

scaleFactor = 1.2;

k = 0;
for i = 1:length(bbox)
    
    row = bbox(i,:);
    
    % Take bigger area around face
    oriValue = row(3);
    resizeValueWidth = row(3) * scaleFactor;
    resizeValueHeight = row(4) * scaleFactor * 160 / 120; % *1.3333: to get correct proportions
    
    row(1) = row(1) - (resizeValueWidth - oriValue) / 2;
    row(2) = row(2) - (resizeValueHeight - oriValue) / 2 - (row(4) * 0.1); %- (row(4) * 0.1): move 10% of height up to get better face-center
    row(3) = row(3) * scaleFactor;
    row(4) = row(4) * scaleFactor * 160 / 120;
    
    
    
    % set bounding box to 120 x 160 pixel
%     deltaY = round(160 * 1.2 - row(3));
%     deltaX = round(120 * 1.2 - row(4));
%     
%     row(2) = row(2) - 10;
%     
%     newX =      max(round(round(row(1)) - deltaX / 2), 0)
%     newY =      max(round(round(row(2)) - deltaY / 2), 0)
%     newWidth =  max(round(round(row(3)) + deltaX), 0)
%     newHeight = max(round(round(row(4)) + deltaY), 0)
%     
%     row(1) = newX;
%     row(2) = newY;
%     row(3) = newWidth;
%     row(4) = newHeight;


    
    
    bbox(i,:) = row;
    
    %disp(bbox);
    
    % Crop Face
    croppedImg = imcrop(img,row);
    
    % Reshape Face
    croppedImg = imresize(croppedImg, [160 120]);
    
    % Save Face
    facesImgs{i} = croppedImg; 
     
    % Save Vector for PCA
    faces(:,i) = facesImgs{i}(:);
    
    %imshow(croppedImg); figure;
    %searchFace = reshape(mn+facesDB(:,i), imsize);
    
    faceImgToSearchHudritsch = facesDB(:,i);
    faceImgToSearch = croppedImg(:);
    faceImgToSearch = double(faceImgToSearch)/255;
    
    faceImgToSearch = faceImgToSearch - mn;
    %pause;
    
    searchFace = reshape(mn+faceImgToSearch, imsize);    %reshape from vector image
    search = eigvec' * (searchFace(:) - mn);  
    
    
    
    % Calculate the squared euclidean distances to all faces in the PC space
    % We use the dot product to square the vector difference.
    for i=1:nImages
        distPC(i) = dot(facesDB2(:,i)-search, facesDB2(:,i)-search);
    end;


    % Sort the distances and show the nearest 14 faces
    [sortedDistPC, sortIndex] = sort(distPC); % sort distances
    figure('Position',[100 500 1000 600]);
    
    for i=1:14
        subplot(3, 7, 4);
        imshow(croppedImg);
        
        subplot(3,7,i+7); 
        imshow((reshape(mn+facesDB(:,sortIndex(i)), imsize)));
        
        title(sprintf('Dist=%2.2f',sortedDistPC(i)));
    end;
    
    %pause;
end













