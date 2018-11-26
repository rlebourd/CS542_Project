A = [1 1 1 1 1 1 1 1 1 1;
     1 1 0 0 0 1 1 1 0 1;
     1 0 0 1 1 1 1 0 0 1;
     1 0 1 1 0 1 1 0 0 1;
     1 0 1 0 0 1 1 0 0 1;
     1 0 0 0 0 1 0 0 0 1;
     1 1 1 1 0 1 0 0 0 1;
     1 1 0 0 0 1 1 0 0 1;
     1 1 1 1 1 1 1 1 1 1];

A = 1-mat2gray(imread('2-Room.png'));

% use the get_rooms function to split a 2D black-and-white image of rooms 
% into a 3D arrange, where each room is separated along the 3rd dimension 
R = get_rooms(A);

subplot(5, 3, 1)
imshow(1-A)
total_area = sum(1-A);
for room_number = 1:size(R, 3)
    subplot(5, 3, room_number+1)
    imshow(R(:, :, room_number))
    room_area = sum(R(:, :, room_number));
    relative_room_area = room_area/total_area;
    title(sprintf('room #%d - area = %.2f%%', room_number, relative_room_area*100))
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
    % scan to find a (row, col) where matrix entry is 0
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
                    if A(i+di, j+dj) == 0
                        A(i+di, j+dj) = fill_value;
                        out_queue = [out_queue; i+di, j+dj];
                    end
                end
            end
        end
        in_queue = out_queue;
    end

    B = A;
end

function rows = numrows(A)
    if isempty(A)
        rows = 0;
    else
        [rows cols] = size(A);
    end
end