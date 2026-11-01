
function [coordinates, ppb] = location(Dat,grid)
% Dat is the npts by 2 matrix.

%% Output
% coordinates is a npts by 4 matrix.
% First column: Rank of the point
% Second Column: Row of the point in partition.
% Third Column: Column of the point in partition.
% Forth Column: Number of square which point lies.
% ppb: vector with N^2 length, which gives the number of points in cells.
 
npts = length(Dat(:,1));
coordinates = zeros(npts, 4);
ppb  = zeros(grid^2, 2);

min_v1 = min(Dat(:,1));
max_v1 = max(Dat(:,1));
Dat(:,1) = (Dat(:,1) - min_v1)/(max_v1 - min_v1);

min_v2 = min(Dat(:,2));
max_v2 = max(Dat(:,2));
Dat(:,2) = (Dat(:,2) - min_v2)/(max_v2 - min_v2);

Dat = Dat*grid;

Dat (Dat == 0) = 1e-15;
Dat(Dat == 1) = 1 - 1e-15;

    for i=1:npts
        coordinates(i,1) = i;
        coordinates(i,2) = ceil(Dat(i,1));
        coordinates(i,3) = ceil(Dat(i,2));
        coordinates(i,4) = coordinates(i,3) + (coordinates(i,2)-1)*grid;
    end

%%% Computation of ppb

nob = grid^2;
boxcounter = zeros(nob,1);


    for i=1:npts
    boxcounter(coordinates(i,4)) = boxcounter(coordinates(i,4)) + 1; 
    end
    
ppb(:,1) = (1:nob)';
ppb(:,2) = boxcounter;

end
