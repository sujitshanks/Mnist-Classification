disp('Start Timer');
disp(datetime('now'));

data_raw = prnist((0:9) , (1:200:1000));
dataset = data2im(data_raw);

preproc = im_box([],0,1)*im_rotate*im_resize([],[20 20])*im_box([],1,0);

proc_data = data_raw * preproc;

feat_data = im_features(proc_data, 'all');

[W, R] = featselm(feat_data, 'eucl-m', 'float', 6);

featureSel = feat_data*W;

[training, testing] = gendat(featureSel, 0.6);

%Parzen & KNN
% Parzen Param -> h - 
% KNN Param -> k - 

h = 0.75;
k = 10;

% Non-Parametric (knnc, parzenc)

w = parzenc(training, h);
v = knnc(training, k);

%-----Test - Non -Parametric

E_parzen = testp(testing, h);
E_knn = testk(testing, k);

% Parametric (nmc,ldc, qdc, fisherc, loglc)

fis_c = fisherc(training);
E_fisher = testc(testing, fis_c);

nme_c = nmc(training);
E_nrmean = testc(testing, nme_c);

lin_c = ldc(training);
E_linear = testc(testing, lin_c);

qad_c = qdc(training);
E_quadra = testc(testing, qad_c);

log_c = loglc(training);
E_logist = testc(testing, log_c);

%-----Test - Parametric


% Complex Classifiers - SVM & Combiners

% svm_c = featureSel * svc(proxm('polynomial',3));
W = svc(training, 'radial_basis');
W_svc9 = svc(training, proxm('p', 9));
% Wnn = bpxnc(training);

%-----Test - Complex

% E_svm_radical = testc(featureSel, svm_c);
error = testc(testing, W);
error_svc9 = testc(testing, W_svc9);
% error_nn = testc(testing, Wnn);

disp('End Timer');
disp(datetime('now'));


% [training, testing] = gendat(features, 0.7);


