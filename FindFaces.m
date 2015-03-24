clear all;
close all;
clc;

% first get Image

img = imread('Images/cpvr_classes/2014HS/02.JPG');
imshow(img);title('one'); figure;

% make a color transformation from RGB to CIELAB color space
cform = makecform('srgb2lab');
img2 = applycform(img,cform);

%make hsv for testing
% 
% img3 = rgb2hsv(img);

%imshow(img3);title('test');figure;
%imshow(img2);title('two');figure;

% extract gray channel three times
img_gray1 = img2(:,:,1);
img_gray2 = img2(:,:,2);
img_gray3 = img2(:,:,3);

% img_hsv1 = img3(:,:,1);
% img_hsv2 = img3(:,:,2);
% img_hsv3 = img3(:,:,3);


%imshow(img_gray1);title('three'); figure;
%imshow(img_gray2);title('four'); figure;
%imshow(img_gray3);title('five'); figure;

% imshow(img_hsv1);title('three'); figure;
% imshow(img_hsv2);title('four'); figure;
% imshow(img_hsv3);title('five'); figure;

%make threshhold gray
g_TS1 = graythresh(img_gray1);
g_TS2 = graythresh(img_gray2);
g_TS3 = graythresh(img_gray3);


%now make BW image with this threshhold
imgBW1 = im2bw(img_gray1,g_TS1);
imgBW2 = im2bw(img_gray2,g_TS2);

imgBW1 = medfilt2(imgBW1, [5 5]);
%imgBW2 = medfilt2(imgBW2, [8 8]);

%magic
O = imgBW1.*imgBW2;


%remove small stuff less than 4000px
%this is maybe not so clever...
O = bwareaopen(O,4000);

%O = edge(O,'canny');

%filling and closing
%se = strel('disk',2);
se = strel('disk',2);
O = imcomplement(imdilate(imcomplement(O), se));

O = imfill(O,'holes');

O = bwareaopen(O,4000);



imshow(O); title('eight'); figure;

[B,L] = bwboundaries(O,'noholes');
imshow(label2rgb(L, @jet, [.2 .2 .2]))
hold on
for i = 1: length(B)
    boundary = B{i};
    plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth',2)
end 
hold off
figure;

imgstats = regionprops(O,'all');

% imshow(img); figure;
% hold on
% for i=1:length(imgstats)
%     x = imgstats(i).Centroid(1);
%     y = imgstats(i).Centroid(2);
%     line(x, y, 'Marker', '*', 'MarkerEdgeColor', 'r')  
% end
% hold off


for i=1:length(imgstats)
    
    boundary = B{i};
  
    %some tries to check if anything is round (face) or not (arm)
    delta_sq = diff(boundary).^2;
    perimeter = sum(sqrt(sum(delta_sq,2)));
    area = imgstats(i).Area;
    metric = 4*pi*area/perimeter^2;
    %disp(imgstats(i).EquivDiameter);

    %expand the BB a bit, so that I get everything
    bboxArea = imgstats(i).BoundingBox;  

    %get hair
    %bboxArea(1) = bboxArea(1) -100;
    bboxArea(2) = bboxArea(2) -10;
    %bboxArea(3) = bboxArea(3) +200;
    bboxArea(4) = bboxArea(4) +20;
    
    if((bboxArea(3) < 120) || (bboxArea(4) < 160))
        
        crop = imcrop(img,bboxArea);
        %get center of old image
        s = size(crop);
        h = s(1);
        w = s(2);
        c = [h/2 w/2];
        hautfarbe = mean(reshape(crop,[],3),1);
        %hautfarbe = mean(impixel(crop,c(1)-5:c(1)+5,c(2)-5:c(2)+5));
        crop = addborder(crop, 50, hautfarbe, 'outer');
        %get center
        s = size(crop);
        h = s(1);
        w = s(2);
        c = [h/2 w/2];
        bbox = bboxArea;
        bbox(1) = c(2) - 60;
        bbox(2) = c(1) - 80;
        bbox(3) = 120; 
        bbox(4) = 160;
        crop = imcrop(crop,bbox);
             
    else
        %here image is bigger
        crop = imcrop(img,bboxArea);
        s = size(crop);
        h = s(1);
        w = s(2);
        if(160/h > 120/w)
            crop = imresize(crop, [(160/h)*160 NaN]);    
        else
            crop = imresize(crop, [NaN (120/w)*120]);
        end
           %get center before crop
        s = size(crop);
        h = s(1);
        w = s(2);
        c = [h/2 w/2];
        
        hautfarbe =mean(reshape(crop,[],3),1);
        crop = addborder(crop, 50, hautfarbe, 'outer');
        %get center
        s = size(crop);
        h = s(1);
        w = s(2);
        c = [h/2 w/2];
        bbox = bboxArea;
        bbox(1) = c(2) - 60;
        bbox(2) = c(1) - 80;
        bbox(3) = 120; 
        bbox(4) = 160;
        crop = imcrop(crop,bbox);
     
    end
    %round
    crop = imcrop(crop, [0, 0, 120, 160]);
    axis on   
    subplot(2,11,i);
    imshow(crop); 
    
    foundFaces{i} = crop;
    
    %pause;
        
end
figure;
% PCA

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


% Step 2: Calculate & show the mean image and shift all faces by it
mn = mean(facesDB, 2);
for i=1:nImages
    facesDB(:,i) = facesDB(:,i)-mn;          % substruct the mean
end;


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
%% Step 4: Transform the mean shifted faces into the faces2 space

%########################
facesDB2 = eigvec' * facesDB;
%######################## 


k = 0;
for i = 1:length(foundFaces)
       
    
    % Crop Face
    croppedImg = foundFaces{i};
    
     
    % Save Vector for PCA
    faces(:,i) = croppedImg(:);
    
    %imshow(croppedImg); figure;
    %searchFace = reshape(mn+facesDB(:,i), imsize);
    
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
    %figure('Position',[100 500 1000 600]);
    
    for i=1:14
        subplot(3, 7, 4);
        imshow(croppedImg);
        
        subplot(3,7,i+7); 
        imshow((reshape(mn+facesDB(:,sortIndex(i)), imsize)));
        
        title(sprintf('Dist=%2.2f',sortedDistPC(i)));
    end;
    
    pause;
end



