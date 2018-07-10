%%% Dynamic 3D Hand Gesture and Action Recognition with Learning Spatio-Temporal Aggregation from Different Representation %%%
			       %## R. Azad, M. Asadi and S. Kasaei ##%
            % will be submitted to IEEE Transactions on CSVT, 2017 %   
                   
    %%% For Runing the code, follow the below steps %%%
%%% 1- Clone the code "https://github.com/rezazad68/Dynamic-3D-Action-Recognition-on-RGB-D-Videos"
%%% 2- Extract the Vl_feat library that exist in vl_feat foldoer
%%% 2- Download the MSR Action 3D Dataset (already exist in MSR-Action3D foldor)
%%% 3- Run the Main_MSRAction3D function for both feature extraction and classification
     %%@@@@ Please see our page for more detail and references @@@@%%
clc
clear all
close all

%% Put Flag value can be 0, 1 or 2, 0 for only extracing features from 3D videos; 1 only for classification step with using extracted features and 2 for both feature extraction and classification step.

Flag = 0; %{0, 1, 2}



switch Flag

  case 0

    disp('Feature Extraction Step');
    Step1_Extract_Features

  case 1

    disp('Description and Classification Step')
    Step2_Description_Classification

  case 2
  
    disp('Both Feature Extraction and Classification Step');
    Step1_Extract_Features     
    Step2_Description_Classification

 end


