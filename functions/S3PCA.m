function filteredData = S3PCA(data, dk, labels)
%S3PCA Smooth each superpixel with a local neighborhood mean.

[nRow, nCol, dim] = size(data);
segments = segSP(data, labels);

numSegments = numel(segments.Y);
filteredPixels = zeros(nRow * nCol, dim);

for i = 1:numSegments
    smoothedPixels = localMean(segments.Y{1, i}, segments.coordinates{1, i}, dk);
    filteredPixels(segments.index{1, i}, :) = smoothedPixels;
end

filteredData = reshape(filteredPixels, [nRow, nCol, dim]);
end
