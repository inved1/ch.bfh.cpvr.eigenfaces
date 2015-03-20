clear; clc;
close all;
clear all;

% Do the PCA
%% Step 1: Load face images & convert each image into a vector of a matrix
k = 0;
for i=0:1:11
    for j=1:1:10
        filename  = sprintf('images/cpvr_faces_160/%04d/%02d.JPG',i,j);
        %disp(filename)
        image_data = imread(filename);
        k = k + 1;
        facesDB(:,k) = image_data(:);
     end;
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

% Display the 30 first eigenvectors as eigenfaces
%figure('Color',[1 1 1]);
%for n = 1:30
    %subplot(3, 10, n);
    %eigvecImg = reshape(eigvec(:,n), imsize);   % Reshape vector to image
    %imshow(eigvecImg, []);                      % Show eigenface image with max. contrast
%end

% Display the summed up eigenvalues as it is the summed up variance
normEigval = eigval / sum(eigval);              % Normalize the eigenvalues
%figure('Color',[1 1 1]);
%plot(cumsum(normEigval));
%xlabel('Index of Eigenvalue'), ylabel('Normalized Summed up Variance');
%xlim([1 400]), ylim([0 1]), grid on;

%% Step 4: Transform the mean shifted faces into the faces2 space

%########################
facesDB2 = eigvec' * facesDB;
%######################## 

%% Step 5: Reconstruction of a face out of the PC's
% i = 78;  %index of face to be reconstructed
% eigvec001 = eigvec; eigvec001(:,  2:end) = 0; % keep the     biggest PC 
% eigvec010 = eigvec; eigvec010(:, 11:end) = 0; % keep the  10 biggest PC
% eigvec050 = eigvec; eigvec050(:, 51:end) = 0; % keep the  50 biggest PC
% eigvec100 = eigvec; eigvec100(:,101:end) = 0; % keep the 100 biggest PC
% eigvec200 = eigvec; eigvec200(:,201:end) = 0; % keep the 200 biggest PC
% 
% facesDB001 = eigvec001 * facesDB2(:,i);
% facesDB010 = eigvec010 * facesDB2(:,i);
% facesDB050 = eigvec050 * facesDB2(:,i);
% facesDB100 = eigvec100 * facesDB2(:,i);
% facesDB200 = eigvec200 * facesDB2(:,i);
% facesDB400 = eigvec    * facesDB2(:,i);
% 
% diff001 = abs(facesDB001 - facesDB(:,i));
% diff010 = abs(facesDB010 - facesDB(:,i));
% diff050 = abs(facesDB050 - facesDB(:,i));
% diff100 = abs(facesDB100 - facesDB(:,i));
% diff200 = abs(facesDB200 - facesDB(:,i));
% diff400 = abs(facesDB400 - facesDB(:,i));
% 
% diffSum001 = sprintf('delta per px: %3.2e',sum(sum(diff001))/nPixels);
% diffSum010 = sprintf('delta per px: %3.2e',sum(sum(diff010))/nPixels);
% diffSum050 = sprintf('delta per px: %3.2e',sum(sum(diff050))/nPixels);
% diffSum100 = sprintf('delta per px: %3.2e',sum(sum(diff100))/nPixels);
% diffSum200 = sprintf('delta per px: %3.2e',sum(sum(diff200))/nPixels);
% diffSum400 = sprintf('delta per px: %3.2e',sum(sum(diff400))/nPixels);

%figure('Color',[1 1 1]);
%set(0,'DefaultLineLineSmoothing','on');  % do antialiased plotting
%subplot(2,6, 1); imshow(reshape(mn+facesDB001, imsize)); title('Reconstr. w. 1 PC');
%subplot(2,6, 7); imshow(reshape(diff001,     imsize)); title(diffSum001);
%subplot(2,6, 2); imshow(reshape(mn+facesDB010, imsize)); title('Reconstr. w. 10 PC');
%subplot(2,6, 8); imshow(reshape(diff010,     imsize)); title(diffSum010);
%subplot(2,6, 3); imshow(reshape(mn+facesDB050, imsize)); title('Reconstr. w. 50 PC');
%subplot(2,6, 9); imshow(reshape(diff050,     imsize)); title(diffSum050);
%subplot(2,6, 4); imshow(reshape(mn+facesDB100, imsize)); title('Reconstr. w. 100 PC');
%subplot(2,6,10); imshow(reshape(diff100,     imsize)); title(diffSum100);
%subplot(2,6, 5); imshow(reshape(mn+facesDB200, imsize)); title('Reconstr. w. 200 PC');
%subplot(2,6,11); imshow(reshape(diff200,     imsize)); title(diffSum200);
%subplot(2,6, 6); imshow(reshape(mn+facesDB400, imsize)); title('Reconstr. w. 400 PC');
%subplot(2,6,12); imshow(reshape(diff400,     imsize)); title(diffSum400);

%% Step 6: Select a face to search in the PC space
%i = 13;                                         %index of face to be searched
%searchFace = reshape(mn+facesDB(:,i), imsize);    %reshape from vector image
%search = eigvec' * (searchFace(:) - mn);        %transform into PC space



%% Load Image with faces to search
scaleFactor = 2;

img = imread('Images/cpvr_classes/2014HS/14.JPG');
%imshow(img); figure;

faceDetector = vision.CascadeObjectDetector('ClassificationModel', 'FrontalFaceCART');

bbox = step(faceDetector, img);

%imgBB = insertObjectAnnotation(img,'rectangle',bbox,'Face');
%imshow(imgBB); figure;

disp(length(bbox));


%imgBB = insertObjectAnnotation(img,'rectangle',bbox,'Face');
%imshow(imgBB);

k = 0;
for i = 1:length(bbox)
    
    row = bbox(i,:);
    
    % Take bigger area around face
%     oriValue = row(3)
%     resizeValue = row(3) * scaleFactor
%     
%     row(1) = row(1) - (resizeValue - oriValue) / 2;
%     row(2) = row(2) - (resizeValue - oriValue) / 2;
%     row(3) = row(3) * scaleFactor;
%     row(4) = row(4) * scaleFactor;
    
    %% set b
    
    % set bounding box to 120 x 160 pixel
    deltaY = round(160 - row(3));
    deltaX = round(120 - row(4));
    
    newX =      max(round(round(row(1)) - deltaX / 2), 0)
    newY =      max(round(round(row(2)) - deltaY / 2), 0)
    newWidth =  max(round(round(row(3)) + deltaX), 0)
    newHeight = max(round(round(row(4)) + deltaY), 0)
    
    row(1) = newX;
    row(2) = newY;
    row(3) = newWidth;
    row(4) = newHeight;
    
    %disp(row)
    
    bbox(i,:) = row;
    
    
    
    disp(bbox);
    
    
    
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
    
    pause;
end













