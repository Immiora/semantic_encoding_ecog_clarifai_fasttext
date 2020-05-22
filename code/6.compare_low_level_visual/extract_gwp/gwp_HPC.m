function [simple, complex] = gwp_HPC(img_indices)

hdir = '/vol/ccnlab-scratch1/julber/clarifai_wordvec_decoding/';
imdir = [hdir, 'data/chill_frames/'];
addpath(genpath('/vol/ccnlab-scratch1/julber/clarifai_wordvec_decoding/code/make_baseline_vision/neural-coding-master')) % https://github.com/artcogsys/Neural-coding

b      = 1;
FOV    = -63.5 : 63.5;
gamma  = 1;
lambda = 2 .^ (2 : 7);
sigma  = [];
theta  = 0 : pi / 8 : 7 * pi / 8;

G = Gabor_wavelet_pyramid(b, FOV, gamma, lambda, sigma, theta);

g    = cell(2, 1);
g{1} = reshape(G{1}, size(G{1}, 1) , size(G{1}, 2), size(G{1}, 3));
g{2} = reshape(G{2}, size(g{1}));


Nimages = length(img_indices);
Nfeatures = size(g{1}, 1);
simple = zeros([Nimages Nfeatures]); 
complex = zeros([Nimages Nfeatures]); 
if size(img_indices, 2) == 1, img_indices = img_indices'; end

counter = 1;
for i = img_indices
   fprintf(['frame ', num2str(i), '\n'])
   img = double(rgb2gray(imread([imdir, 'frame', num2str(i), '.jpg'])));
   img = imresize(img, [length(FOV) length(FOV)]);
   x = reshape(img, size(img, 1) * size(img, 2), 1);
   simple(counter, :) = g{1} * x; 
   complex(counter, :) = sqrt((g{1} * x) .^ 2 + (g{2} * x) .^ 2);
   counter = counter + 1;
end
