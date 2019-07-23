% Adds path of the other files

           
path = cd;
path = fullfile(path, '..');

addpath(genpath(path));
