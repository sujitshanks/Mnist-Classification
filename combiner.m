image_size_pixel = 32;
number_of_classes = 10;

data_raw = prnist((0:number_of_classes-1) , (1:1:1000));
% boxed = im_box(data_raw, [], 1);
% resized = im_resize(boxed, [image_size_pixel, image_size_pixel]);
% rotated = im_rotate(resized);
% dataset = rotated;

preproc = im_resize([],[image_size_pixel image_size_pixel]);
dataset = data_raw*preproc; 

% pixel_data = data2im(dataset);
% pixel_data = transpose(pixel_data);
% labelArray = [];
% 
% for i = 1:length(pixel_data)
%     class = i / 100;
%     class = floor(class - 0.01);
%     labelArray = [labelArray; class];
% end


dataset = prdataset(dataset ,getnlab(dataset));

% [W, R] = featselm(dataset, 'eucl-m', 'float', 20);
% dataset = dataset*W;

% feat_data = im_features(data_raw, 'all');

% 
% dataset = feat_data;

% show(rotated)

% features = im_features(rotated, 'all');
% [training, testing] = gendat(features, 0.7);

[training, testing] = gendat(dataset, 0.7);
training_dataset = prdataset(training, getnlab(training));
testing_dataset = prdataset(testing, getnlab(testing));


% pca_vecs = fisherm(training_dataset, 9);
% training_dataset = training_dataset * pca_vecs;
% 
% pca_vecs = fisherm(testing_dataset, 9);
% testing_dataset = testing_dataset * pca_vecs;

[ts1,ts2] = gendat(training_dataset, 0.8);

%---------------------------------------------------------------%

% [training1, training2] = gendat(training_dataset, 0.5);



bxpnc_norm = bpxnc*classc;

W_bxpnc_norm_1 = bxpnc_norm(training_dataset, [image_size_pixel, image_size_pixel/2] , 3000, [], ts2);
% W_bxpnc_norm_2 = bxpnc_norm(training2, [image_size_pixel] , 3000);

% C_bxpnc_norm = testing_dataset * W_bxpnc_norm * classc;

W_svc_norm_1 = svc(training_dataset, proxm('p', 5)) * classc;
% W_svc_norm_2 = svc(training2, proxm('p', 9)) * classc;

% W_svc_norm = svc_norm(training_dataset);
% C_svc_norm = testing_dataset * W_svc_norm * classc;

% W_final_1 = [W_svc_norm_1   W_svc_norm_2]*medianc;
% W_final_2 = [W_bxpnc_norm_1 W_bxpnc_norm_2]*medianc;

W_final = [W_bxpnc_norm_1 W_svc_norm_1] * minc;

[error_combiner, steps] = testc(testing_dataset, W_final);
[error_nnet, steps] = testc(testing_dataset, W_bxpnc_norm_1);
[error_svc, steps] = testc(testing_dataset, W_svc_norm_1);

disp(['Combiner - ', num2str(error_combiner)]);
disp(['SVC - ', num2str(error_svc)]);
disp(['NNet - ', num2str(error_nnet)]);

% bxpnc_norm(training_dataset, (32), 3000);


% v = (bxpnc_norm())*ldc;
% W = training_dataset * v;
% error_alone = testc(testing_dataset, W);


% pca_vecs = pcam(training, 20);
% 
% mapped_training = training * pca_vecs;
% W = bpxnc(mapped_training, [15 15], 3500);
% 
% mapped_testing = testing * pca_vecs;
% error = testc(mapped_testing, W)