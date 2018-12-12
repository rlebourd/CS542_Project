close all

A = [1 1 1 1 1 1 1 1 1 1;
     1 1 0 0 0 1 1 1 0 1;
     1 0 0 1 1 1 1 0 0 1;
     1 0 1 1 0 1 1 0 0 1;
     1 0 1 0 0 1 1 0 0 1;
     1 0 0 0 0 1 0 0 0 1;
     1 1 1 1 0 1 0 0 0 1;
     1 1 0 0 0 1 1 0 0 1;
     1 1 1 1 1 1 1 1 1 1];

%A = mat2gray(imread('1-Room.png'));
Doors = imread('1-Door.png');
Walls = imread('1-Wall.png');
Windows = imread('1-Window.png');
Separations = imread('1-Separation.png');

Combined = (Doors+Walls+Windows+Separations);
Combined(find(Combined>1)) = 1;
A = Combined;

% use the get_rooms function to split a 2D black-and-white image of rooms 
% into a 3D arrange, where each room is separated along the 3rd dimension 
R = get_rooms(A);

figure
%subplot(3, 2, 1)
figure
imshow(1-R(:, :, 1)-A)
title('Floor mask image')
total_area = sum(sum((1-R(:, :, 1)-A)>0))
for room_number = 1:size(R, 3)
    if room_number > 1
        %subplot(3, 2, room_number)
        figure
        imshow(R(:, :, room_number))
        room_area = sum(sum(R(:, :, room_number)>0))
        relative_room_area = room_area/total_area;
        title(sprintf('Room mask #%d - Area = %.2f%%', room_number-1, relative_room_area*100))
    end
end

function C = get_rooms(A)
    [B, num_rooms] = flood_fill_all(A);
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
