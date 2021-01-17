disp('Start Timer');
disp(datetime('now'));

number_of_classes = 10;

data_raw = prnist((0:number_of_classes - 1) , (1:10:1000));

preproc = im_box([],0,1)*im_rotate*im_box([],1,0);
proc_data = data_raw * preproc;
% show(proc_data);

dataset = data2im(proc_data);
dataset = transpose(dataset);

labelArray = [];

for i = 1:length(dataset)
    class = i / 100;
    class = floor(class - 0.01);
    labelArray = [labelArray; class];
end

pixel_matrix_Dataset = prdataset(dataset, labelArray);

%Parzen & KNN
% Parzen Param -> h - 
% KNN Param -> k - 

[training, testing] = gendat(pixel_matrix_Dataset, 0.7);

% training = pcam(pixel_matrix_Dataset);
% testing = 

% h = 0.25;
% k = 2;
% 
% % Non-Parametric (knnc, parzenc)
% 
% w = parzenc(training, h);
% E_parzen = testp(testing, h);
% 
% v = knnc(training, k);
% E_knn = testk(testing, k);
% 
% W = svc(training);
% E_svc = testc(testing, W);
% 
% comb_cl = [parzenc*classc knnc*classc] * svc;
% comb = comb_cl(training);
% E_comb = testc(testing, comb);

%-----Test - Non -Parametric


%{




feat_data = im_features(proc_data, 'all');

[W, R] = featselm(feat_data, 'eucl-m', 'forward', 6);

featureSel = feat_data*W;

J = feateval(featureSel, 'eucl-m'); 
J_All = feateval(feat_data, 'eucl-m'); 

% Complex Classifiers - SVM & Combiners

%svm_c = featureSel * svc(proxm('polynomial',3));

%-----Test - Complex

%E_svm_radical = testc(featureSel, svm_c);
%}



disp('End Timer');
disp(datetime('now'));