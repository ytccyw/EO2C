function smoothedPixels = localMean(pixels, coordinates, neighborCount)
%LOCALMEAN Smooth pixels with nearest neighbors in image space.

if isempty(coordinates)
    coordinates = pixels;
end

neighborCount = min(neighborCount, size(pixels, 1) - 1);
[numPixels, dim] = size(pixels);
smoothedPixels = zeros(numPixels, dim);

if neighborCount < 1
    smoothedPixels = pixels;
    return;
end

[~, neighborIndex] = pdist2(coordinates, coordinates, 'euclidean', 'Smallest', neighborCount + 1);
neighborIndex = neighborIndex';

for i = 1:numPixels
    smoothedPixels(i, :) = mean(pixels(neighborIndex(i, :), :), 1);
end
end
