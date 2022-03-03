clear;
files = strings(3064,1);
for i = 1:3064
    files(i) = append(string(i), ".mat");
end
npics = 3064;
imsz = 256; % Change to change images size
%%
ims = zeros(imsz, imsz, npics);
GT = zeros(imsz, imsz, npics);
label = zeros(npics, 1);
PID = strings(npics,1);
for i = 1:npics
    load(append("datasets/", files(i)));
    GT(:,:,i) = imresize(cjdata.tumorMask, [imsz imsz]);
    label(i) = cjdata.label;
    PID(i) = cjdata.PID;
    x = mat2gray(cjdata.image); % normalising
    %x = cjdata.image;
    x = adapthisteq(x); % Contrast enhancement
    x = imresize(x, [imsz imsz]); % Resizing
    ims(:,:,i) = x;
end
PID = categorical(PID);
PID = uint8(PID);
%%
str_ims = zeros(imsz, imsz, npics);
for i = 1:npics
    str_ims(:,:,i) = skullstrip(ims(:,:,i), 15,  0.2);
    
end

%% Segment out any tissue and take and add the tumour ground truth so that the pixel labels have three levels
gtarr = zeros(imsz, imsz, npics);
for i = 1:npics
    x = str_ims(:,:,i);
    binim = rescale(x);
    binim = binim > 0.1;
    binim = bwareaopen(binim, 50);
    binim = imfill(binim, 'holes');
    gtarr(:,:,i) = binim + GT(:,:,i);
end
gtarr = uint8(gtarr);
%% This is to write the ground truth to a file which you must to create an imagedatastore
% for i = 1:3064
%     x = str_ims(:,:,i);
%     x = repmat(x, [1 1 3]);
%     gt = gtarr(:, :, i);
%     if i < 10
%         imwrite(x, append('images/000', string(i), '.png'));
%         imwrite(gt, append('pixelLabel/000', string(i), '.png'));
%     end
%     if i > 9 & i < 100
%         imwrite(x, append('images/00', string(i), '.png'));
%         imwrite(gt, append('pixelLabel/00', string(i), '.png'));
%     end
%     if i > 99 & i < 1000
%         imwrite(x, append('images/0', string(i), '.png'));
%         imwrite(gt, append('pixelLabel/0', string(i), '.png'));
%     end
%     if i > 999
%         imwrite(x, append('images/', string(i), '.png'));
%         imwrite(gt, append('pixelLabel/', string(i), '.png'));
%     end
% end

%%
dataDir = fullfile(pwd);
imDir = fullfile(dataDir, 'images');
pxDir = fullfile(dataDir, 'pixelLabel');

%%
imds = imageDatastore(imDir);

%% Define the datastore for the pixel labels (ground truth)
classes = ["Tumour" "Normal" "Background"];
pixelLabelID = [2 1 0];
pxds = pixelLabelDatastore(pxDir, classes, pixelLabelID);

%%

I = readall(imds);
C = readall(pxds);

%% Show example of image and ground truth
n = 1266;
B = labeloverlay(I{n},C{n});
imshow(B)

%% Show class imbalance for the pixels
tbl = countEachLabel(pxds)

frequency = tbl.PixelCount/sum(tbl.PixelCount);

bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

%% Partition the data into training, testing and validation 
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionOnPat(imds,pxds, PID);



%% Create deeplab V 3+ network based on resnet18
% Specify the network image size
imageSize = [256 256 3];
numClasses = numel(classes);

lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet18");

%% Calculate weights for each class

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq)./ imageFreq;

%% Define the last classification layer

pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);

%% Options for sgdm
% Define validation data.
dsVal = combine(imdsVal,pxdsVal);

% Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'ValidationData',dsVal,...
    'MaxEpochs',30, ...  
    'MiniBatchSize',8, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 4);

%% Create training data
dsTrain = combine(imdsTrain, pxdsTrain);

%% Train network

doTraining = false; % set to true to train the network
if doTraining    
    [net, info] = trainNetwork(dsTrain,lgraph,options);
else
    data = load('patpartnet2.mat'); 
    net = data.net;
end


%% Check that it works
close all;
nim = 10; % Change to plot any image and its segmentation
I = readimage(imdsTest, nim);
C = semanticseg(I, net);
expected= readimage(pxdsTest, nim);
expected = uint8(expected);
B = labeloverlay(I,C,'Transparency',0.7);
x = labeloverlay(B, expected, 'Transparency',0.7);
imshow(x);

%% Check against groundtruth
% Not great 
expectedResult = readimage(pxdsTest, nim);
actual = uint8(C);
expected = uint8(expectedResult);
imshowpair(actual, expected)

iou = jaccard(C,expectedResult);
table(classes',iou)
%% Evaluation on test data

pxdsResults = semanticseg(imdsTest,net, ...
    'MiniBatchSize',4, ...
    'WriteLocation',tempdir, ...
    'Verbose',false);

%% Eval on all
pxdsResultsAll = semanticseg(imds,net, ...
    'MiniBatchSize',4, ...
    'WriteLocation',tempdir, ...
    'Verbose',false);

%% Get metrics for the semantic segmenation model - test data
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);
%% Check metrics of the semantic segmentation
metrics.DataSetMetrics
metrics.ClassMetrics


%% Save some images of the segmentation for report
% for i = 1:100
%     nim = i;
%     I = readimage(imdsTest, nim);
%     C = semanticseg(I, net);
%     expected= readimage(pxdsTest, nim);
%     expected = uint8(expected);
%     B = labeloverlay(I,C,'Transparency',0.7);
%     x = labeloverlay(B, expected, 'Transparency',0.7);
%     imwrite(x, append('segmented_test_images/',string(i),'.png'));
% end

%% Extract the segmented images from the network (all images, not just test set)
masks = zeros(256, 256, 3064);
for i = 1:3064
    x = uint8(readimage(pxdsResultsAll, i));
    x = x == 1;
    x = bwpropfilt(x, 'area', 1); % Keep only the largest object in the image
    masks(:,:,i) = x;
end
masks = logical(masks);
%% Extract features from full or segmented images

feat = zeros(3064, 14);

imtyp = "seg"; % set to "full" for feature extraction from full image

if imtyp == "seg"
    for i = 1:3064
        x = str_ims(:,:,i);
        x = rescale(x);
        mask = masks(:,:,i);
        if sum(mask, [1 2]) == 0
            mask(:,:) = true;
        end

        % Central moments
        rmx = x;
        rmx(~mask) = NaN;
        rmx = rmmissing(rmx(:));
        mu1 = mean(rmx); 
        mu2 = var(rmx);
        mu3 = skewness(rmx);
        mu4 = kurtosis(rmx);

        eta = SI_Moment(x, mask);
        hum = Hu_Moments(eta);

        warning('off', 'images:graycomatrix:scaledImageContainsNan')
        x(~mask) = NaN;
        glcm_x = graycomatrix(x); % Gives warning about not computing pairs with nan
        glcm_stats = graycoprops(glcm_x);

        feat(i,:) = [mu1, mu2, mu3, mu4, hum, ...
            glcm_stats.Correlation, glcm_stats.Energy, glcm_stats.Homogeneity];
    end
end

if imtyp == "full"
    for i = 1:3064
        x = str_ims(:,:,i);
        x = rescale(x);

        % Central moments
        rmx = x;
        rmx = rmmissing(rmx(:));
        mu1 = mean(rmx); 
        mu2 = var(rmx);
        mu3 = skewness(rmx);
        mu4 = kurtosis(rmx);

        eta = SI_Moment(x);
        hum = Hu_Moments(eta);
        
        glcm_x = graycomatrix(x); % Gives warning about not computing pairs with nan
        glcm_stats = graycoprops(glcm_x);

        feat(i,:) = [mu1, mu2, mu3, mu4, hum, ...
            glcm_stats.Correlation, glcm_stats.Energy, glcm_stats.Homogeneity];
    end
end

%% Data partition by patient and stratifed by class
patind = {};
patlab = [];
for i = 1:233
patind{i} = find(PID == i);
lab = find(PID == i);
patlab(i) = unique(label(lab));
end
rng(1234);
cv = cvpartition(patlab,'HoldOut',0.25, 'Stratify', true);
idx = cv.test;

trainingInd = patind(~idx);
trainingInd = cell2mat(trainingInd');

testInd = patind(idx);
testInd = cell2mat(testInd');

%% Partition the features and the labels
trainfeat = feat(trainingInd, :);
testfeat = feat(testInd, :);

trainLab = label(trainingInd);
testLab = label(testInd);

%% Train SVM and check 10 fold CV error

t = templateSVM('Standardize', 1, 'KernelFunction', 'gaussian');
C = fitcecoc(trainfeat, trainLab, 'Learners', t);
Ccv = crossval(C,'kfold', 10);
yhat = kfoldPredict(Ccv);
%yhat = predict(C, trainfeat);
confusionchart(categorical(trainLab), categorical(yhat));

genError = kfoldLoss(Ccv);
error = 1 - sum(trainLab == yhat)/length(trainLab)

%% Check error on test data

tstpred = predict(C, testfeat);
confusionchart(categorical(testLab), categorical(tstpred));
error = 1 - sum(testLab == tstpred)/length(testLab)