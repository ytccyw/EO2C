function segments = segSP(data, labels)
%SEGSP Group pixel vectors and coordinates by superpixel label.

[nRow, nCol, dim] = size(data);
pixels = reshape(data, nRow * nCol, dim);
flatLabels = reshape(labels, [1, nRow * nCol]);
segmentIds = unique(flatLabels);

segments.index = cell(1, numel(segmentIds));
segments.coordinates = cell(1, numel(segmentIds));
segments.Y = cell(1, numel(segmentIds));

for i = 1:numel(segmentIds)
    segments.index{1, i} = find(flatLabels == segmentIds(i));
    [row, col] = find(labels == segmentIds(i));
    segments.coordinates{1, i} = [row, col];
    segments.Y{1, i} = pixels(flatLabels == segmentIds(i), :);
end
end
