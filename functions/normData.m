function normalizedData = normData(X)
%NORMDATA L2-normalize each feature column.

columnNorms = vecnorm(X, 2, 1);
columnNorms(columnNorms < 1e-12) = 1;
normalizedData = X ./ columnNorms;
end
