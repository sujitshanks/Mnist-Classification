% W_pca_norm = W_pca * classc;
% W_svc_norm = W_svc * classc;
% e_svc = 0.028;
% disp(['SVC error ', num2str(e_svc)]);
% 

% e_nn = testc(testing, W_bpxnc);
% disp(['NN error ', num2str(e_nn)]);

image_size_pixel = 32;

data_raw = prnist((0:9) , (1:100:1000));
preproc = im_resize([],[image_size_pixel image_size_pixel]);

dataset = data_raw*preproc; 

show(dataset);

dataset = prdataset(dataset ,getnlab(dataset));
[training, testing] = gendat(dataset, 0.7);

W_pca    = pcam(training, 100);
training = training*W_pca;
testing  = testing *W_pca;

W_svc_5 = svc(training, proxm('p', 5));
W_svc_9 = svc(training, proxm('p', 9));

[d, t] = gendat(training, 0.8);
l = [64 32];
W_bpxnc = bpxnc(d, l, 1000, [], t);

w = [W_svc_5, W_svc_9]*meanc;

e_combiner = testc(testing, w);
e_NNet = testc(testing, W_bpxnc);
e_svc_5 = testc(testing, W_svc_5);
e_svc_9 = testc(testing, W_svc_9);

disp(['Combiner error - ', num2str(e_combiner)]);
disp(['SVC error 5 - ', num2str(e_svc_5)]);
disp(['SVC error 9 - ', num2str(e_svc_9)]);
disp(['NNET error - ', num2str(e_NNet)]);

% votec:   0.043667
% minc:    0.024333
% maxc:    0.019333
% medianc: 0.018667
% meanc:   0.018667