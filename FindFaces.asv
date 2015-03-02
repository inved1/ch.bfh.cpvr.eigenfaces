clear all;
close all;
clc;


%std and mean
uMEAN = 100;
uSTD = 80;


% first get Image

img = imread('Images/cpvr_classes/2014HS/05.JPG');
imshow(img);title('one');

% make a color transformation from RGB to CIELAB color space
cform = makecform('srgb2lab');
img2 = applycform(img,cform);

imshow(img2);title('two');
figure;

% extract gray channel twice
img_gray1 = img2(:,:,1);
img_gray2 = img2(:,:,2);

imshow(img_gray1);title('three'); figure;
imshow(img_gray2);title('four'); figure;

%make threshhold gray
g_TS1 = graythresh(img_gray1);
g_TS2 = graythresh(img_gray2);

%now make BW image with this threshhold
imgBW1 = im2bw(img_gray1,g_TS1);
imgBW2 = im2bw(img_gray2,g_TS2);
%figure; imshow(imgBW1);title('five');
%figure; imshow(imgBW2);title('six');

O = imgBW1.*imgBW2;

imshow(O);title('seven'); figure; 

%remove small stuff less than 4000px
%this is maybe not so clever...
O = bwareaopen(O,4000);

%filling and closing
se = strel('disk',8);
O = imclose(O,se);
O = imfill(O,'holes');
imshow(O); title('eight'); figure;

[B,L] = bwboundaries(O,'noholes');
imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on
for i = 1: length(B)
    boundary = B{i};
    plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth',2)
end 
hold off
figure;

imgstats = regionprops(O);

for i=1:length(imgstats)
    
    boundary = B{i};
    
    delta_sq = diff(boundary).^2;
    perimeter = sum(sqrt(sum(delta_sq,2)));
    
    area = imgstats(i).Area;
    
    metric = 4*pi*area/perimeter^2;
    disp(metric);

    %expand the BB a bit, so that I get everything
    bboxArea = imgstats(i).BoundingBox;  
%     bboxArea(1) = bboxArea(1) - 80;
%     bboxArea(2) = bboxArea(2) - 80;
%     bboxArea(3) = bboxArea(3) + 160;
%     bboxArea(4) = bboxArea(4) + 160;
    
    

% what to check:
% get middle of image and make a region growing
% or
% get middle of image and use 
    crop = imcrop(img,bboxArea);
    subplot(2,11,i);
    imshow(crop);
    
    foundFaces{i} = crop;
        
end

iPCA = 1;

for i=1:length(foundFaces)
    foundFace = foundFaces{i}(:);
    foundFace = double(foundFace) /255;
    mean_foundFace = mean(foundFace,2);
    
    % only one face - i dont need to shift?!
    shifted_foundFace = foundFace;% - repmat(mean_foundFace,1,1);
    
    
    [U,E,V] = svd(shifted_foundFace,0);
    
    P = U(:,1:1);
    
    found_weight(:,i) =  P' * shifted_foundFace;
    
    
end



%now :
% 1. loop trough each folder and make eigenface
% 2. check eigenface with cropped image
% 3. mark them
for i=0:1:2 % 2 instead of 11
    k=0;
    w = 120;
    h = 160;
    
    for j=1:1:10 % 10 images per person
       filename  = sprintf('images/cpvr_faces_160/%04d/%02d.JPG',i,j);
       %disp(filename)
       image_data = double(imread(filename));
       k = k + 1;
       faces(:,k) = image_data(:);
    end;
    
    % mean face of faces
    %eigenface 
  
  
    
    clear faces;
    
    
    
end;




% old code

    %convert and normalize 
%     faces = double(faces) / 255;
%     mean_face = mean(faces,2);
%     
%     shifted_images = faces - repmat(mean_face,1,k);
%     [U,E,V] = svd(shifted_images,0);
%     
%     eigenVals = diag(E);
%     lmda = eigenVals(1:iPCA); % only 2 principal components
%     
%     %space face - face space, funny
%     P = U(:,1:iPCA);
%     
%     %weight
%     weight = P' * shifted_images;
% 
%     x = mean(weight);
%     disp(x);
%     
%     clear P;
%     clear weight;
%     clear faces;
%     clear mean_face;
%     clear shifted_images;







%shall I now get the round stuff or not ? (face is mostly round)
%or on the same height? (since we are all "tall"?)

%following code is to check if it's round.
% and then cut the 11 'roundest' stuff and work with this
% but since my skin-color is sometimes hidden by hair, this doesnt work
% now I think i just loop over the other skin parts as well



%imgstats = regionprops(O);
% 
% for i=1:length(imgstats)
%     
%     boundary = B{i};
%     
%     delta_sq = diff(boundary).^2;
%     perimeter = sum(sqrt(sum(delta_sq,2)));
%     
%     area = imgstats(i).Area;
%     
%     metric = 4*pi*area/perimeter^2;
%     disp(metric);
%     allmetrics(i) = metric;
%    
%         
% end
% 
% allmetrics = sort(allmetrics);
% allmetrics2 = allmetrics(length(allmetrics)-10:end); % 10: because of 11 stutends
% 
% for i=1:length(imgstats)
%     boundary = B{i};
%     
%     delta_sq = diff(boundary).^2;
%     perimeter = sum(sqrt(sum(delta_sq,2)));
%     
%     area = imgstats(i).Area;
%     
%     metric = 4*pi*area/perimeter^2;
%     
%     if any(metric==allmetrics2)== 1
%         bboxArea = imgstats(i).BoundingBox;  
%         bboxArea(1) = bboxArea(1) - 50;
%         bboxArea(2) = bboxArea(2) - 50;
%         bboxArea(3) = bboxArea(3) + 100;
%         bboxArea(4) = bboxArea(4) + 100;
%         crop = imcrop(img,bboxArea);
%         subplot(2,11,i);
%         imshow(crop);
%         
%     end
%     
%     
% end


%  if metric > threshhold
%         
%         %expand bounding box a littlebit

%         
%     end
%     


