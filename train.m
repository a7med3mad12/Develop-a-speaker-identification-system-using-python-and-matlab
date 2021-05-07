ads = audioDatastore('data','IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames')
[adsTrain, adsTest] = splitEachLabel(ads,0.8);
trainDatastoreCount = countEachLabel(adsTrain)
%testDatastoreCount = countEachLabel(adsTest)
[sampleTrain, dsInfo] = read(adsTrain);
fs = dsInfo.SampleRate;
windowLength = round(0.03*fs);
overlapLength = round(0.025*fs);

features = [];
labels = [];
w = 0;
while hasdata(adsTrain)
    [audioIn,dsInfo] = read(adsTrain);
    lol = size(audioIn);
    if lol(2) == 1
        audioIn = [audioIn audioIn];
    end
    melC = mfcc(audioIn,fs,'Window',hamming(windowLength,'periodic'),'OverlapLength',overlapLength);
    f0 = pitch(audioIn,fs,'WindowLength',windowLength,'OverlapLength',overlapLength);
%     feat = [melC,f0];
a = size(melC);
feat = [reshape(melC,a(1),[]),f0];
feat
    
    voicedSpeech = isVoicedSpeech(2 * audioIn(1),fs,windowLength,overlapLength);
    
    feat(~voicedSpeech,:) = [];
    label = repelem(dsInfo.Label,size(feat,1));
    
    features = [features;feat];
    labels = [labels,label];
end
M = mean(features,1);
S = std(features,[],1);
features = (features-M)./S;
trainedClassifier = fitcknn( ...
    features, ...
    labels, ...
    'Distance','euclidean', ...
    'NumNeighbors',5, ...
    'DistanceWeight','squaredinverse', ...
    'Standardize',false, ...
    'ClassNames',unique(labels));
k = 5;
group = labels;
c = cvpartition(group,'KFold',k); % 5-fold stratified cross validation
partitionedModel = crossval(trainedClassifier,'CVPartition',c);
validationAccuracy = 1 - kfoldLoss(partitionedModel,'LossFun','ClassifError');
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);
validationPredictions = kfoldPredict(partitionedModel);
figure
cm = confusionchart(labels,validationPredictions,'title','Validation Accuracy');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
features = [];
labels = [];
numVectorsPerFile = [];

while hasdata(adsTest)
    [audioIn,dsInfo] = read(adsTest);
    
    lol = size(audioIn);
    if lol(2) == 1
        audioIn = [audioIn audioIn];
    end   
    
    melC = mfcc(audioIn,fs,'Window',hamming(windowLength,'periodic'),'OverlapLength',overlapLength);
    f0 = pitch(audioIn,fs,'WindowLength',windowLength,'OverlapLength',overlapLength);
%     feat = [melC,f0];
a = size(melC);
feat = [reshape(melC,a(1),[]),f0];
    
    voicedSpeech = isVoicedSpeech(2 * audioIn(1),fs,windowLength,overlapLength);
    
    feat(~voicedSpeech,:) = [];
    numVec = size(feat,1);
    label = repelem(dsInfo.Label,numVec);
    
    numVectorsPerFile = [numVectorsPerFile,numVec];
    features = [features;feat];
    labels = [labels,label];
    w=w+1;
end
features = (features-M)./S;
prediction = predict(trainedClassifier,features);
prediction = categorical(string(prediction));
figure('Units','normalized','Position',[0.4 0.4 0.4 0.4])
cm = confusionchart(labels,prediction,'title','Test Accuracy (Per Frame)');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';