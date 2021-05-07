function voicedSpeech = isVoicedSpeech(x,fs,windowLength,overlapLength)

pwrThreshold = -40;
[segments,~] = buffer(x,windowLength,overlapLength);
pwr = pow2db(var(segments));
isSpeech = (pwr > pwrThreshold);

zcrThreshold = 1000;
zeroLoc = (x==0);
crossedZero = logical([0;diff(sign(x))]);
crossedZero(zeroLoc) = false;
[crossedZeroBuffered,~] = buffer(crossedZero,windowLength,overlapLength);
zcr = (sum(crossedZeroBuffered,1)*fs)/(2*windowLength);
isVoiced = (zcr < zcrThreshold);

voicedSpeech = isSpeech & isVoiced;

end