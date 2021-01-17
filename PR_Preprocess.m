disp('Start Timer');
disp(datetime('now'));

data_raw = prnist((0:9) , (1:100:1000));
dataset = data2im(data_raw);

preproc = im_box([],0,1)*im_rotate*im_resize([],[20 20])*im_box([],1,0);

proc_data = data_raw * preproc;

feat_data = im_features(proc_data, 'all');

[W, R] = featselm(feat_data, 'eucl-m', 'forward', 3);

featureSel = feat_data*W;

%Parzen & KNN
% Parzen Param -> h - 
% KNN Param -> k - 

h = 0.75;
k = 10;

% Non-Parametric (knnc, parzenc)

w = parzenc(featureSel, h);
v = knnc(featureSel, k);

%-----Test - Non -Parametric

E_parzen = testp(featureSel, h);
E_knn = testk(featureSel, k);

% Parametric (nmc,ldc, qdc, fisherc, loglc)

fis_c = fisherc(featureSel);
nme_c = nmc(featureSel);
lin_c = ldc(featureSel);
qad_c = qdc(featureSel);
log_c = loglc(featureSel);

%-----Test - Parametric

E_fisher = testc(featureSel, fis_c);
E_nrmean = testc(featureSel, nme_c);
E_linear = testc(featureSel, lin_c);
E_quadra = testc(featureSel, qad_c);
E_logist = testc(featureSel, log_c);

% Complex Classifiers - SVM & Combiners

svm_c = featureSel * svc(proxm('polynomial',3));

%-----Test - Complex

E_svm_radical = testc(featureSel, svm_c);

disp('End Timer');
disp(datetime('now'));