%##########################################################################
% File:       PCA4-Eigenfaces.m
% Purpose:    Principal Component Analysis Demo for Eigenface analysis
%             The example is loosly based several tutorials found on
%             Mathworks.com
% Author:     Marcus Hudritsch
% Date:       Nov-2013
% Copyright:  Marcus Hudritsch, Kirchrain 18, 2572 Sutz
%             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
%             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
%##########################################################################
clear all; close all; clc; %clear matrices, close figures & clear cmd wnd.

%% Download the face database
% You can find the database at the follwoing link, 
% http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html The
% database contains 400 pictures of 40 subjects. Download the zipped
% database and unzip it in the same directory as this file. 

%% Step 1: Load face images & convert each image into a vector of a matrix
k = 0;
for i=0:1:11
    for j=1:1:10
        filename  = sprintf('images/cpvr_faces_160/%04d/%02d.JPG',i,j);
        disp(filename)
        image_data = imread(filename);
        k = k + 1;
        faces(:,k) = image_data(:);
     end;
end;
nImages = k;                     %total number of images
imsize = size(image_data);       %size of image (they all should have the same size) 
nPixels = imsize(1)*imsize(2);   %number of pixels in image
faces = double(faces)/255;       %convert to double and normalize

%% Step 2: Calculate & show the mean image and shift all faces by it
mn = mean(faces, 2);
for i=1:nImages
    faces(:,i) = faces(:,i)-mn;          % substruct the mean
end;
figure('Color',[1 1 1]); 
imshow(reshape(mn, imsize)); title('mean face');

%% Step 3: Calculate Eigenvectors & Eigenvalues
% Method 1 using princomp from Matlab
% tic;
% [eigvec, score, eigval] = princomp(faces');     %eigvec: <10304x10304>, eigval: <10304x1> sorted descendent
% eigvec = eigvec(:,1:nImages);                   %eigvec: Cut down to <10304x400>
% toc;

%% Step 3: Calculate Eigenvectors & Eigenvalues 
% Method 2: Create covariance matrix faster by using 
% Turk and Pentland's trick to get the eigenvectors of faces*faces' from
% the eigenvectors of faces'*faces
tic;
C = faces'*faces;
[eigvec,eigval] = eig(C);
eigvec = faces * eigvec;                        % Convert eigenvectors back as if they came from A'*A
eigvec = eigvec / (sqrt(abs(eigval)));          % Normalize eigenvectors
% eigvec & eigval are in fact sorted but in the wrong order
eigval = diag(eigval);                          % Get the eigenvalue from the diagonal
eigval = eigval / nImages;                      % Normalize eigenvalues
[eigval, indices] = sort(eigval, 'descend');    % Sort the eigenvalues
eigvec = eigvec(:, indices);                    % Sort the eigenvectors accordingly
toc;

% tic;
% C2 = faces'*faces;
% C2 = C2./(size(faces,1)-1);
% [eigvec2,eigval2] = eig(C2);
% % eigvec & eigval are in fact sorted but in the wrong order
% eigval2 = diag(eigval2);                          % Get the eigenvalue from the diagonal
% [eigval2, indices2] = sort(eigval2, 'descend');    % Sort the eigenvalues
% eigvec2 = eigvec2(:, indices2);                    % Sort the eigenvectors accordingly
% toc;
% 
% tic;
% C3 = cov(faces);
% [eigvec3,eigval3] = eig(C3);
% % eigvec & eigval are in fact sorted but in the wrong order
% eigval3 = diag(eigval3);                          % Get the eigenvalue from the diagonal
% [eigval3, indices3] = sort(eigval3, 'descend');    % Sort the eigenvalues
% eigvec3 = eigvec3(:, indices3);                    % Sort the eigenvectors accordingly
% toc;



% Display the 30 first eigenvectors as eigenfaces
figure('Color',[1 1 1]);
for n = 1:30
    subplot(3, 10, n);
    eigvecImg = reshape(eigvec(:,n), imsize);   % Reshape vector to image
    imshow(eigvecImg, []);                      % Show eigenface image with max. contrast
end

% Display the summed up eigenvalues as it is the summed up variance
normEigval = eigval / sum(eigval);              % Normalize the eigenvalues
figure('Color',[1 1 1]);
plot(cumsum(normEigval));
xlabel('Index of Eigenvalue'), ylabel('Normalized Summed up Variance');
xlim([1 400]), ylim([0 1]), grid on;
 
%% Step 4: Transform the mean shifted faces into the faces2 space

%########################
faces2 = eigvec' * faces;
%######################## 

%% Step 5: Reconstruction of a face out of the PC's
i = 78;  %index of face to be reconstructed
eigvec001 = eigvec; eigvec001(:,  2:end) = 0; % keep the     biggest PC 
eigvec010 = eigvec; eigvec010(:, 11:end) = 0; % keep the  10 biggest PC
eigvec050 = eigvec; eigvec050(:, 51:end) = 0; % keep the  50 biggest PC
eigvec100 = eigvec; eigvec100(:,101:end) = 0; % keep the 100 biggest PC
eigvec200 = eigvec; eigvec200(:,201:end) = 0; % keep the 200 biggest PC

faces001 = eigvec001 * faces2(:,i);
faces010 = eigvec010 * faces2(:,i);
faces050 = eigvec050 * faces2(:,i);
faces100 = eigvec100 * faces2(:,i);
faces200 = eigvec200 * faces2(:,i);
faces400 = eigvec    * faces2(:,i);

diff001 = abs(faces001 - faces(:,i));
diff010 = abs(faces010 - faces(:,i));
diff050 = abs(faces050 - faces(:,i));
diff100 = abs(faces100 - faces(:,i));
diff200 = abs(faces200 - faces(:,i));
diff400 = abs(faces400 - faces(:,i));

diffSum001 = sprintf('delta per px: %3.2e',sum(sum(diff001))/nPixels);
diffSum010 = sprintf('delta per px: %3.2e',sum(sum(diff010))/nPixels);
diffSum050 = sprintf('delta per px: %3.2e',sum(sum(diff050))/nPixels);
diffSum100 = sprintf('delta per px: %3.2e',sum(sum(diff100))/nPixels);
diffSum200 = sprintf('delta per px: %3.2e',sum(sum(diff200))/nPixels);
diffSum400 = sprintf('delta per px: %3.2e',sum(sum(diff400))/nPixels);

figure('Color',[1 1 1]);
set(0,'DefaultLineLineSmoothing','on');  % do antialiased plotting
subplot(2,6, 1); imshow(reshape(mn+faces001, imsize)); title('Reconstr. w. 1 PC');
subplot(2,6, 7); imshow(reshape(diff001,     imsize)); title(diffSum001);
subplot(2,6, 2); imshow(reshape(mn+faces010, imsize)); title('Reconstr. w. 10 PC');
subplot(2,6, 8); imshow(reshape(diff010,     imsize)); title(diffSum010);
subplot(2,6, 3); imshow(reshape(mn+faces050, imsize)); title('Reconstr. w. 50 PC');
subplot(2,6, 9); imshow(reshape(diff050,     imsize)); title(diffSum050);
subplot(2,6, 4); imshow(reshape(mn+faces100, imsize)); title('Reconstr. w. 100 PC');
subplot(2,6,10); imshow(reshape(diff100,     imsize)); title(diffSum100);
subplot(2,6, 5); imshow(reshape(mn+faces200, imsize)); title('Reconstr. w. 200 PC');
subplot(2,6,11); imshow(reshape(diff200,     imsize)); title(diffSum200);
subplot(2,6, 6); imshow(reshape(mn+faces400, imsize)); title('Reconstr. w. 400 PC');
subplot(2,6,12); imshow(reshape(diff400,     imsize)); title(diffSum400);

%% Step 6: Select an face to search in the PC space
i = 78;                                         %index of face to be searched
searchFace = reshape(mn+faces(:,i), imsize);    %reshape from vector image
search = eigvec' * (searchFace(:) - mn);        %transform into PC space

% Calculate the squared euclidean distances to all faces in the PC space
% We use the dot product to square the vector difference.
for i=1:nImages
    distPC(i) = dot(faces2(:,i)-search, faces2(:,i)-search);
end;

% Sort the distances and show the nearest 14 faces
[sortedDistPC, sortIndex] = sort(distPC); % sort distances
figure('Color',[1 1 1]);
for i=1:14
    subplot(2,7,i); 
    imshow((reshape(mn+faces(:,sortIndex(i)), imsize))); 
    title(sprintf('Dist=%2.2f',sortedDistPC(i)));
end;