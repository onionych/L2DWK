% example

%% add path
cd('../data/');             unzip('data.zip');   addpath(cd);
cd('../Base_classifier');   addpath(cd);
cd('../L2DWK');             addpath(cd);

cd('../tests');

example_dataset = 'autos';
%% generate base classifier
generate_bagging_CV(example_dataset,101,1,'tree');

%% learn L2DWK
load(example_dataset);
target = target(target>0)';

cd(strcat(example_dataset,'/valid1/'));
load('valid1');
load('classifier_tree_101_Re');                         % load matrix R: the classification result for each data and classifier 

fprintf('\nL2DWK starts.\n');
W1 = learn_L2DWK(R,traintarget,'linear',[],'dis',0.8);   % linear kernel, disagreement diversity, lambda=0.8
fprintf('\nL2DWK ends.\n');

%% predict L2DWK
% clc;
load('classifier_tree_101_rec');
fprintf('rec of bag = %.2f percent\n',rec_e(end)*100);

C = unique(traintarget);
y1 = predict_L2DWK(predict,W1,C);
fprintf('rec of L2DWK = %.2f percent\n',mean(y1==testtarget')*100);

cd ../..
