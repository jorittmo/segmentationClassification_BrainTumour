function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionOnPat(imds,pxds,PID)

patind = {};
for i = 1:233
patind{i} = find(PID == i);
end
%%
rng(0);
numpats = 233;
shuffleInd = randperm(numpats);
numTrain = round(0.60*numpats);
trainingInd = shuffleInd(1:numTrain);
trainingInd = patind(trainingInd);
trainingInd = cell2mat(trainingInd');

%%
numVal = round(0.20 * numpats);
valInd = shuffleInd(numTrain+1:numTrain+numVal);
valIndtot = patind(valInd);
valInd = cell2mat(valIndtot');
%%
testInd = shuffleInd(numTrain+numVal+1:end);
testInd = patind(testInd);
testInd = cell2mat(testInd');

%%
% Create image datastores for training and test.
trainingImages = imds.Files(trainingInd);
valImages = imds.Files(valInd);
testImages = imds.Files(testInd);

imdsTrain = imageDatastore(trainingImages);
imdsVal = imageDatastore(valImages);
imdsTest = imageDatastore(testImages);

% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = [2 1 0];

% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingInd);
valLabels = pxds.Files(valInd);
testLabels = pxds.Files(testInd);

pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end
