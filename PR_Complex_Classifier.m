disp('Start Timer');
disp(datetime('now'));

number_of_classes = 10;

data_raw = prnist((0:number_of_classes - 1) , (1:100:1000));

preproc = im_box([],0,1)*im_rotate*im_resize([],[20 20])*im_box([],1,0);
proc_data = data_raw * preproc;
show(proc_data);

dataset = data2im(proc_data);
dataset = transpose(dataset);

labelArray = [];

for i = 1:length(dataset)
    class = i / 10;
    class = floor(class - 0.01);
    labelArray = [labelArray; class];
end

pixel_matrix_Dataset = prdataset(dataset, labelArray);

%Parzen & KNN
% Parzen Param -> h - 
% KNN Param -> k - 

h = 0.25;
k = 2;

% Non-Parametric (knnc, parzenc)

w = parzenc(pixel_matrix_Dataset, h);
v = knnc(pixel_matrix_Dataset, k);

%-----Test - Non -Parametric

E_parzen = testp(pixel_matrix_Dataset, h);
E_knn = testk(pixel_matrix_Dataset, k);

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