%% Image data
Nimages = 9749;
hdir = '/clarifai_wordvec_decoding/';
imdir = [hdir, 'data/chill_frames/'];
addpath(genpath('/clarifai_wordvec_decoding/code/make_baseline_vision/neural-coding-master')) % https://github.com/artcogsys/Neural-coding
addpath /vol/optdcc/fieldtrip-latest/fieldtrip/qsub

%% GWP Parameters:
b      = 1;
FOV    = -63.5 : 63.5;
gamma  = 1;
lambda = 2 .^ (2 : 7);
sigma  = [];
theta  = 0 : pi / 8 : 7 * pi / 8;

G = Gabor_wavelet_pyramid(b, FOV, gamma, lambda, sigma, theta);

%% Loop on cluster
Indices = repmat_part(Nimages, 1000);
v = version;
v = v(find(v == 'R'):find(v == ')')-1);

success = 0;
while ~success
    try
        [out1, out2] = qsubcellfun(@gwp_HPC, Indices, ...
                     'memreq', 20 * 1024^3, 'timreq', 25000, ...
                     'matlabcmd',['/opt/matlab-', v, '/bin/matlab']);
      success = 1;
    catch err
   end
end


simple = vertcat(out1{:});
complex = vertcat(out2{:});



%%
save([hdir, '/data/chill_gwp_features.mat'], 'simple', 'complex', 'G', '-v7.3')

%%
save([hdir, '/data/chill_gwp_features_simple.mat'], 'simple')

%%
save([hdir, '/data/chill_gwp_features_complex.mat'], 'complex')

