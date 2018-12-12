close all
clear all
clc

global DOOR_WIDTH
global DOOR_HEIGHT

DOOR_WIDTH = 80;
DOOR_HEIGHT = 80;

% build the Resources struct array
Resources(1) = struct('OriginalImage', [], 'GTImage', []);
NUM_GT_IMAGES = 80;
for i = 1:NUM_GT_IMAGES
    Resources(i).OriginalImage = imread(char(i + ".png"));
    Resources(i).GTImage = 1-imread(char(i + "-Door.png"));
end

AllPositiveExamples = [];
AllNegativeExamples = [];
for i = 1:NUM_GT_IMAGES
    OriginalImage = Resources(i).OriginalImage;
    GTImage = Resources(i).GTImage;

    [PositiveExamples, NegativeExamples] = GenerateTrainingSetFromImage(OriginalImage, GTImage);
    
    Resources(i).PositiveExamples = PositiveExamples;
    Resources(i).NegativeExamples = NegativeExamples;

    AllPositiveExamples = [AllPositiveExamples; single(PositiveExamples)/255.0 ones(size(PositiveExamples, 1), 1)];
    AllNegativeExamples = [AllNegativeExamples; single(NegativeExamples)/255.0 zeros(size(NegativeExamples, 1), 1)];
end

NUM_POS_EXAMPLES = size(AllPositiveExamples, 1);
NUM_NEG_EXAMPLES = size(AllNegativeExamples, 1);

AllPositiveExamples = AllPositiveExamples(randsample(NUM_POS_EXAMPLES, NUM_POS_EXAMPLES), :);
AllNegativeExamples = AllNegativeExamples(randsample(NUM_NEG_EXAMPLES, NUM_NEG_EXAMPLES), :);

disp(NUM_POS_EXAMPLES)
disp(NUM_NEG_EXAMPLES)

csvwrite('doornegex.txt', AllNegativeExamples)
csvwrite('doorposex.txt', AllPositiveExamples)

ShuffledPositiveExamples = AllPositiveExamples(randsample(size(AllPositiveExamples, 1), size(AllNegativeExamples, 1)), :);
ShuffledNegativeExamples = AllNegativeExamples(randsample(size(AllNegativeExamples, 1), size(AllNegativeExamples, 1)), :);

AllExamples = [ShuffledPositiveExamples; ShuffledNegativeExamples];
AllExamples = AllExamples(randsample(size(AllExamples, 1), size(AllExamples, 1)), :);
csvwrite(char('doorex.txt'), AllExamples)

% figure
% for i = 1:NumPosExamples
%     subplot(ceil(sqrt(NumPosExamples)), ceil(sqrt(NumPosExamples)), i);
%     Example = reshape(PositiveExamples(i, :), 50, 50);
%     imshow(Example)
% end
% 
% figure
% for i = 1:NumNegExamples
%     subplot(ceil(sqrt(NumNegExamples)), ceil(sqrt(NumNegExamples)), i);
%     Example = reshape(NegativeExamples(i, :), 50, 50);
%     imshow(Example)
% end

function [PositiveExamples, NegativeExamples] = GenerateTrainingSetFromImage(OriginalImage, GroundTruthImage)
    global DOOR_HEIGHT
    global DOOR_WIDTH

    NUM_EXAMPLES_TO_GENERATE = 300;
    [IMAGE_HEIGHT, IMAGE_WIDTH] = size(OriginalImage);
    OriginalImage = OriginalImage;
    
    % generate random centroids to serve as negative examples
    RandomCentroids = [randi([DOOR_WIDTH IMAGE_WIDTH-DOOR_WIDTH], NUM_EXAMPLES_TO_GENERATE, 1) randi([DOOR_HEIGHT IMAGE_HEIGHT-DOOR_HEIGHT], NUM_EXAMPLES_TO_GENERATE, 1)];

    % get the centroids of the ground-truth masks
    GroundTruthCentroids = GetGroundTruthCentroids(GroundTruthImage);
        
    % remove the randomly generated centroids that are too close to the
    % ground-truth centroids
    DIST_THRESHOLD_FOR_REMOVAL = DOOR_HEIGHT;
    RandomCentroidsToRemove = [];
    for i = 1:length(RandomCentroids)
        for j = 1:size(GroundTruthCentroids, 1)
            Dist = sqrt(sum((RandomCentroids(i, :)-GroundTruthCentroids(j, :)).^2));
            if Dist < DIST_THRESHOLD_FOR_REMOVAL
                RandomCentroidsToRemove = [RandomCentroidsToRemove i];
            end
        end
    end
    
    RandomCentroids(RandomCentroidsToRemove, :) = [];
    
    % for each ground-truth centroid, get the 50x50 subimage with the same
    % centroid - these will serve as the positive example images
    PositiveExamples = [];
    for idx = 1:size(GroundTruthCentroids, 1)
        centroid = GroundTruthCentroids(idx, :);
        centroid = ceil(centroid);
        x1 = int32(centroid(1) - (DOOR_WIDTH/2)+1);
        x2 = int32(centroid(1) + (DOOR_WIDTH/2));
        y1 = int32(centroid(2) - (DOOR_HEIGHT/2)+1);
        y2 = int32(centroid(2) + (DOOR_HEIGHT/2));
        
        NewPositiveExamples = zeros(36, DOOR_WIDTH*DOOR_HEIGHT, 'uint8');
        k = 0;
        for dx = -10:10:10
            for dy = -10:10:10
                k = k+1;
                
                xrange = (x1+dx):(x2+dx);
                yrange = (y1+dy):(y2+dy);
                DoorImage0Degrees = OriginalImage(yrange, xrange);
                DoorImage90Degrees = rot90(DoorImage0Degrees);
                DoorImage180Degrees = rot90(DoorImage90Degrees);
                DoorImage270Degrees = rot90(DoorImage180Degrees);
                NewPositiveExamples(4*(k-1)+1, :) = DoorImage0Degrees(:);
                NewPositiveExamples(4*(k-1)+2, :) = DoorImage90Degrees(:);
                NewPositiveExamples(4*(k-1)+3, :) = DoorImage180Degrees(:);
                NewPositiveExamples(4*(k-1)+4, :) = DoorImage270Degrees(:);
            end
        end
        PositiveExamples = [PositiveExamples; NewPositiveExamples];
    end
        
    % for each ground-truth centroid, get the 50x50 subimage with the same
    % centroid - these will serve as the negative example images
    NegativeExamples = zeros(size(RandomCentroids, 1), DOOR_WIDTH*DOOR_HEIGHT, 'uint8');
    NegativeExamplesToDrop = zeros(size(RandomCentroids, 1), 1);
    for idx = 1:size(RandomCentroids, 1)
        centroid = RandomCentroids(idx, :);
        centroid = ceil(centroid);

        x1 = int32(centroid(1) - (DOOR_WIDTH/2)+1);
        x2 = int32(centroid(1) + (DOOR_WIDTH/2));
        y1 = int32(centroid(2) - (DOOR_HEIGHT/2)+1);
        y2 = int32(centroid(2) + (DOOR_HEIGHT/2));
                
        NonDoorImage = OriginalImage(y1:y2, x1:x2);
        NegativeExamples(idx, :) = NonDoorImage(:);
        NumWhitePixels = sum(NonDoorImage(:) > 250);
        NumPixels = numel(NonDoorImage(:));
        if NumWhitePixels >= 0.99*NumPixels
            NegativeExamplesToDrop(idx) = 1;
        end
    end
    % get rid of negative examples that are just white pixels
    NegativeExamples(find(NegativeExamplesToDrop), :) = [];
end

function Centroids = GetGroundTruthCentroids(Image)
    Doors = get_doors(Image);
    
    X = cumsum(ones(size(Image)), 2);
    Y = cumsum(ones(size(Image)), 1);
    
    Centroids = zeros(size(Doors, 3), 2);
    for door_idx = 1:size(Doors, 3)
        CurrDoor = Doors(:, :, door_idx);
        %figure
        %imshow(CurrDoor)
        %title("DOOR " + door_idx)
        NonZeroX = X .* CurrDoor;
        NonZeroY = Y .* CurrDoor;
        x_centroid = sum(sum(NonZeroX)) / sum(sum(NonZeroX > 0));
        y_centroid = sum(sum(NonZeroY)) / sum(sum(NonZeroY > 0));
        Centroids(door_idx, :) = [x_centroid, y_centroid];
    end
    
    %figure
    %imshow(Image)
    %hold on
    %plot(Centroids(:, 1), Centroids(:, 2), 'r*')
end

function C = get_doors(A)
    [B, num_rooms] = flood_fill_all(A);
    %disp(num_rooms)
    C = zeros(size(B, 1), size(B, 2), num_rooms);
    for i = 1:num_rooms
        R = (B == (i+1));
        C(:, :, i) = R;
    end
end

function [B, num_sections] = flood_fill_all(A)
    i = 2;
    C = A;
    B = zeros(size(A));
    while ~isequal(B, C)
        B = C;
        C = flood_fill_with(C, i);
        i = i+1;
    end
    num_sections = i-3;
end

function B = flood_fill_with(A, fill_value)
    assert(fill_value ~= 0)

    [ROWS, COLS] = size(A);
    % scan to find a (row, col) where matrix entry is 0 (black)
    for i = 1:ROWS
        for j = 1:COLS
           if A(i, j) == 0
               break
           end
        end 
        if A(i,j) == 0
            break
        end
    end
    
    if A(i,j) ~= 0
        B = A;
        return
    end
    
    % perform breadth-first fill of zero entries in A, starting at (i, j)
    in_queue = [i, j];
    out_queue = [];
    while ~isempty(in_queue)
        out_queue = [];
        for row = 1:numrows(in_queue)
            i = in_queue(row, 1);
            j = in_queue(row, 2);
            for di = -1:1
                for dj = -1:1
                    %fprintf('i=%f, i+di=%f\n', i, i+di)
                    if (i+di > 0 && j+dj > 0) && (i+di <= numrows(A) && j+dj<=numcols(A)) && (A(i+di, j+dj) == 0)
                        A(i+di, j+dj) = fill_value;
                        out_queue = [out_queue; i+di, j+dj];
                        % now check that the 'channel' is wide enough
                        if (i+di-1 > 0 && j+dj-1 > 0) && (i+di+1 <= numrows(A) && j+dj+1<numcols(A))
                            sum_around_point = sum(sum(A(i+di-1:i+di+1,j+dj-1:j+dj+1)));
                            if sum_around_point < 600
                                A(i+di, j+dj) = fill_value;
                                out_queue = [out_queue; i+di, j+dj];
                            end
                        end
                    end
                end
            end
        end
        in_queue = out_queue;
    end

    B = A;
end

function cols = numcols(A)
    if isempty(A)
        cols = 0;
    else
        [rows cols] = size(A);
    end
end

function rows = numrows(A)
    if isempty(A)
        rows = 0;
    else
        [rows cols] = size(A);
    end
end