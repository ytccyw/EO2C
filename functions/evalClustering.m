function results = evalClustering(truth, prediction)
%EVALCLUSTERING ACC, Kappa, arithmetic NMI, Purity and adjusted Rand index.
% Labels are finite numeric IDs. Apply any background mask before calling.
% results.label uses indices into class_ids; larger indices are unmatched.
% Kappa is NaN when both mapped partitions contain only one class.

validateattributes(truth, {'numeric','logical'}, {'vector','nonempty','real','finite'}, mfilename, 'truth');
validateattributes(prediction, {'numeric','logical'}, {'vector','nonempty','real','finite'}, mfilename, 'prediction');
assert(numel(truth)==numel(prediction), 'EO2C:LabelCount', 'Label vectors must have the same length.');
[classIds,~,truth] = unique(truth(:));
[~,~,prediction] = unique(prediction(:));
n = numel(truth);
nTrue = numel(classIds);
nPred = max(prediction);
counts = accumarray([truth,prediction],1,[nTrue,nPred]);
trueCounts = sum(counts,2);
predCounts = sum(counts,1)';

% Unmatched predicted clusters remain distinct and still count in the denominator.
pairs = matchpairs(-counts,n+1);
mapping = zeros(nPred,1);
mapping(pairs(:,2)) = pairs(:,1);
unmatched = find(mapping==0);
mapping(unmatched) = nTrue+(1:numel(unmatched));
mapped = mapping(prediction);
correct = mapped==truth;
acc = sum(correct)/n;
chance = sum(trueCounts(pairs(:,1)).*predCounts(pairs(:,2)))/n^2;
if chance==1
    kappa = NaN;
else
    kappa = (acc-chance)/(1-chance);
end

[rows,cols,values] = find(counts);
rows = rows(:); cols = cols(:); values = values(:);
mutualInformation = sum((values/n).*log((n*values)./(trueCounts(rows).*predCounts(cols))));
pTrue = trueCounts/n;
pPred = predCounts/n;
entropySum = -sum(pTrue.*log(pTrue))-sum(pPred.*log(pPred));
if entropySum==0
    nmi = 1;
else
    nmi = min(1,max(0,2*mutualInformation/entropySum));
end
purity = sum(max(counts,[],1))/n;

pairCount = n*(n-1)/2;
sameBoth = sum(values.*(values-1)/2);
sameTrue = sum(trueCounts.*(trueCounts-1)/2);
samePred = sum(predCounts.*(predCounts-1)/2);
if pairCount==0
    ari = 1;
else
    expected = sameTrue*samePred/pairCount;
    maximum = (sameTrue+samePred)/2;
    if maximum==expected
        ari = 1;
    else
        ari = (sameBoth-expected)/(maximum-expected);
    end
end

results.metric = [acc,kappa,nmi,purity,ari];
results.p_acc = accumarray(truth,double(correct),[nTrue,1])./trueCounts;
results.class_ids = classIds(:)';
results.label = mapped;
end
