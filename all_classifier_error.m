% Pixel Error Reporting Experiments - 
%
% For Classifiers : 
% 1. parametric (nmc, ldc, qdc, fisherc, loglc) - 5
% 2. non-parametric (knnc, parzenc) - 2
% 3. advanced (neural networks, support vector classifiers);  - 2

disp('Start Timer');
disp(datetime('now'));

fisherc_error_array = [];
nmc_error_array = [];
ldc_error_array = [];
qdc_error_array = [];
loglc_error_array = [];

parzenc_error_array = [];
knnc_error_array = [];

nnet_error_array = [];
svc_error_array = [];

current_dataset = dataset_1;


for v = 1:1:10
   switch v
    case 1
        disp('1')
        current_dataset = dataset_1;
    case 2
        disp('2')
        current_dataset = dataset_2;
    case 3
        disp('3')
        current_dataset = dataset_3;
    case 4
        disp('4')
        current_dataset = dataset_4;
    case 5
        disp('5')
        current_dataset = dataset_5;
    case 6
        disp('6')
        current_dataset = dataset_6;
    case 7
        disp('7')
        current_dataset = dataset_7;
    case 8
        disp('8')
        current_dataset = dataset_8;
    case 9
        disp('9')
        current_dataset = dataset_9;
    case 10
        disp('10')
        current_dataset = dataset_10;
    
        c =  [c next];
        
    otherwise
        disp('DONE DONE')
   end 
   
   % Split Train/Test Data
    current_dataset = prdataset(current_dataset ,getnlab(current_dataset));  
    [training, testing] = gendat(current_dataset, 0.7);
   
   % Feature selection or extraction code
   
    W_kernel_sel = pcam(training,'eucl-m');
    training = training*W_kernel_sel;
    testing  = testing *W_kernel_sel;
   
   % Classifier Test and Train
   
    % Parametric Classifier
    fis_c = fisherc(training);
    E_fisher = testc(testing, fis_c);
    fisherc_error_array = [fisherc_error_array E_fisher];

    nme_c = nmc(training);
    E_nrmean = testc(testing, nme_c);
    nmc_error_array = [nmc_error_array E_nrmean];
    
    lin_c = ldc(training);
    E_linear = testc(testing, lin_c);
    ldc_error_array = [ldc_error_array E_linear];
    
    qad_c = qdc(training);
    E_quadra = testc(testing, qad_c);
    qdc_error_array = [qdc_error_array E_quadra];
    
    log_c = loglc(training);
    E_logist = testc(testing, log_c);
    loglc_error_array = [loglc_error_array E_logist];
    
    
    % Non-Parametric (knnc, parzenc)

    parzen_c = parzenc(training, 10);
    E_parzen = testc(testing, parzen_c);
    parzenc_error_array = [parzenc_error_array E_parzen];
    
    knn_c = knnc(training);
    E_knnc = testc(testing, knn_c);
    knnc_error_array = [knnc_error_array E_knnc];
    
    
    % Advanced (svm, nn)
    
    W_svc = svc(training, proxm('p', 2)) * classc;
    [error_svc, steps] = testc(testing, W_svc);
    svc_error_array = [svc_error_array error_svc];
    
    W_bxpnc = bpxnc(training,(32));
    E_bxpnc = testc(testing, W_bxpnc);
    nnet_error_array = [nnet_error_array E_bxpnc];

end

disp('END Timer');
disp(datetime('now'));
