close all
clear all
clc

%% Load the extracted features and initiate the Vl_feat library

load Action3D_Features
run vl_feat/vlfeat-0.9.20/toolbox/vl_setup.m
disp('Data Loaded');

%% Create an object from class description and classification %%

DC_Class = Description_Classification ;
RBF= 6.8;  Regularization= 100;
Need_to_Train_Visual_Word = 0;

%% Divide data to train and test set

[TF_HOG,TS_HOG,TT_HOG, TeF_HOG, TeS_HOG, TeT_HOG, Tr, Te] = DC_Class.Divide_Train_Test(Total_Frontal_RHOG, Total_Side_RHOG, Total_Top_RHOG, Sample_label, Subject_Index);
[TF_LBP,TS_LBP,TT_LBP, TeF_LBP, TeS_LBP, TeT_LBP, Tr, Te] = DC_Class.Divide_Train_Test(Total_Frontal_RLBP, Total_Side_RLBP, Total_Top_RLBP, Sample_label, Subject_Index);

%% Train the Visual Words or use the trained visual words with 80 visual words
   %%% 0 stand for using the pretrained visual words and 1 for training %%%

if Need_to_Train_Visual_Word == 1

[VW_F_HOG, VW_S_HOG, VW_T_HOG ]= DC_Class.Generate_Visual_Words(TF_HOG, TS_HOG, TT_HOG, 1);
[VW_F_LBP, VW_S_LBP, VW_T_LBP ]= DC_Class.Generate_Visual_Words(TF_LBP, TS_LBP, TT_LBP, 2);

else

load Trained_VW
% Parameters that used for learned visual words
DC_Class.N_Visual_Words_HOG= 80;
DC_Class.N_Visual_Words_LBP= 80;
DC_Class.Numer_PCA_HOG = 130;
DC_Class.Numer_PCA_LBP = 130;

end


%% Extract Vlad representation

[Train_HOG, Test_HOG]= DC_Class.Extract_Description(VW_F_HOG, VW_S_HOG, VW_T_HOG, TF_HOG, TS_HOG, TT_HOG, TeF_HOG, TeS_HOG, TeT_HOG, 2);
[Train_LBP, Test_LBP]= DC_Class.Extract_Description(VW_F_LBP, VW_S_LBP, VW_T_LBP, TF_LBP, TS_LBP, TT_LBP, TeF_LBP, TeS_LBP, TeT_LBP, 1);

%% Applying the PCA on both LBP and HOG descriptions

    [disc_set, Dim] = DC_Class.Eigenface_f(Train_HOG, 2);
    Train_HOG = disc_set'*Train_HOG;
    Test_HOG  = disc_set'*Test_HOG;
    F_train_a = Train_HOG./(repmat(sqrt(sum(Train_HOG.*Train_HOG)), [Dim,1]));
    F_test_a  = Test_HOG./(repmat(sqrt(sum(Test_HOG.*Test_HOG)), [Dim,1]));


    [disc_set, Dim]  = DC_Class.Eigenface_f(Train_LBP, 1);
    Train_LBP = disc_set'*Train_LBP;
    Test_LBP  = disc_set'*Test_LBP;
    F_train_b = Train_LBP./(repmat(sqrt(sum(Train_LBP.*Train_LBP)), [Dim,1]));
    F_test_b  = Test_LBP./(repmat(sqrt(sum(Test_LBP.*Test_LBP)), [Dim,1]));



%% Concatination of the descriptions

    temp_train=[F_train_a; F_train_b];
    temp_test=[F_test_a; F_test_b];
    
    train_data = [Tr, temp_train'];
    test_data = [Te, temp_test'];

%% Classification and results

    [TrainingTime, TestingTime, TrainAC, TestAC, TY, ConfusMatrix] = elm_kernel(train_data, test_data, 1, Regularization, 'RBF_kernel',RBF);


    disp ('Accuracy is:');
    disp(TestAC);





