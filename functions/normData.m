function normalizedData = normData(X)
%NORMDATA L2-normalize each feature column.

normalizedData = zeros(size(X), 'like', X);
for column = 1:size(X, 2)
    normalizedData(:, column) = X(:, column) ./ max(1e-12, norm(X(:, column)));
end
end
