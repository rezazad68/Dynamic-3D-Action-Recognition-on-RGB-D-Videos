%%% Dynamic 3D Hand Gesture and Action Recognition with Learning Spatio-Temporal Aggregation from Different Representation %%%
						%## R. Azad, M. Asadi and S. Kasaei ##%
                   % will be submitted toIEEE Transactions on CSVT % 
                   
          %% Description and Classification class for feature description %%
          
% This Class contain function as follow:
% 1- Divide_Train_Test: dividing data set into train and test set
% 2- Generate_Visual_Words: learning visual words from train data, sample
% of learned visual words is also available "Trained_VW.mat"
% 3-  Extract_Description: describing data with use of Vlad representation
% 4- Eigenface_f and Find_K_Max_Eigen: for PCA purpose


classdef Description_Classification
    
    properties
    
     Number_Actions = 20;
     N_Visual_Words_HOG = 60;
     N_Visual_Words_LBP = 60;
     Numer_PCA_HOG = 110; % both 110 > 94.14
     Numer_PCA_LBP = 110;
     Train_idx = [1 3 5 7 9];
     Front_SHOG = 72;
     Front_SLBP = 59;
     Side_SHOG = 72;
     Side_SLBP = 59;
     Top_SHOG = 72;
     Top_SLBP = 59;

    end
    
    methods
       
        function [Train_Front,Train_Side,Train_Top, Test_Front, Test_Side, Test_Top, Train_Labeles, Test_Labeles] = Divide_Train_Test(obj,Frontals, Sides, Tops, Sample_label, sub_index)
            
          Train_Size = zeros(1, obj.Number_Actions);
          Test_Size  = zeros(1, obj. Number_Actions);
          Train_Front = cell(1,1);
          Test_Front = cell(1,1);
          Train_Side = cell(1,1);
          Test_Side = cell(1,1);
          Train_Top = cell(1,1);
          Test_Top = cell(1,1);
          Train_Labeles = [];
          Test_Labeles = [];
          
         
          for idx = 1:obj.Number_Actions 
            
           F1 = Frontals(Sample_label==idx);
           F2 = Sides(Sample_label==idx);
           F3 = Tops(Sample_label==idx); 
           ID = sub_index(Sample_label==idx);
              
           for k = 1:length(obj.Train_idx)
              
               ID(ID == obj.Train_idx(k)) = 0;
               
           end
           
            Train_Front = [Train_Front, F1(ID == 0)];
            Test_Front  = [Test_Front, F1(ID > 0)];
    
            Train_Side = [Train_Side, F2(ID == 0)];
            Test_Side  = [Test_Side, F2(ID > 0)];    
    
            Train_Top = [Train_Top, F3(ID == 0)];
            Test_Top  = [Test_Top, F3(ID > 0)];
    
            Train_Size(idx) = sum(ID == 0);
            Test_Size(idx)  = size(F1,2) - Train_Size(idx);
            Train_Labeles = [Train_Labeles; idx * ones(Train_Size(idx), 1)];
            Test_Labeles = [Test_Labeles; idx * ones(Test_Size(idx), 1)];
    
           end

           Train_Front = Train_Front(2:end);
           Test_Front = Test_Front(2:end);
           Train_Side = Train_Side(2:end);
           Test_Side = Test_Side(2:end);
           Train_Top = Train_Top(2:end);
           Test_Top = Test_Top(2:end); 
            
            
        end
        
        function [VW_Front, VW_Side, VW_Top ]= Generate_Visual_Words(obj, Fronts, Sides, Tops, Type)
            
 	    if Type==1

	    NVW = obj.N_Visual_Words_HOG;

            else

            NVW = obj.N_Visual_Words_LBP;

	    end

            Mat_Front = cell2mat(Fronts);
            Mat_Side = cell2mat(Sides);
            Mat_Top = cell2mat(Tops);
            
            VW_Front = vl_kmeans(Mat_Front, NVW);
            VW_Side = vl_kmeans(Mat_Side, NVW);
            VW_Top = vl_kmeans(Mat_Top, NVW);

        
        end
        
        function [Train_Desc, Test_Desc]= Extract_Description(obj, VW_Front, VW_Side, VW_Top, F, S, T, Ft, St, Tt, Type)
            
         kdtree_f = vl_kdtreebuild(VW_Front);
         kdtree_s = vl_kdtreebuild(VW_Side);
         kdtree_t = vl_kdtreebuild(VW_Top);

         if Type==1
    
          Front_Size = obj.Front_SLBP;
          Side_Size = obj.Side_SLBP;
          Top_Size =  obj.Top_SLBP;
          NVW = obj.N_Visual_Words_LBP;
          
         else
             
          Front_Size = obj.Front_SHOG;
          Side_Size = obj.Side_SHOG;
          Top_Size =  obj.Top_SHOG;    
          NVW = obj.N_Visual_Words_HOG;
              
         end
    
         Tr_f = zeros(Front_Size*NVW, length(F));
         Tr_s = zeros(Side_Size*NVW, length(S));
         Tr_t = zeros(Top_Size*NVW, length(T));

         for idx = 1:length(F)

          nn_f = vl_kdtreequery(kdtree_f, VW_Front, F{idx}) ;
          nn_s = vl_kdtreequery(kdtree_s, VW_Side, S{idx}) ;
          nn_t = vl_kdtreequery(kdtree_t, VW_Top, T{idx}) ;
         
          assignments_f = zeros(NVW,numel(nn_f));
          assignments_s = zeros(NVW,numel(nn_s));
          assignments_t = zeros(NVW,numel(nn_t));
        
          assignments_f(sub2ind(size(assignments_f), nn_f, 1:length(nn_f))) = 1;
          assignments_s(sub2ind(size(assignments_s), nn_s, 1:length(nn_s))) = 1;
          assignments_t(sub2ind(size(assignments_t), nn_t, 1:length(nn_t))) = 1;
       
          vlad_f = vl_vlad(F{idx},VW_Front, assignments_f);
          vlad_s = vl_vlad(S{idx},VW_Side, assignments_s);
          vlad_t = vl_vlad(T{idx},VW_Top, assignments_t);
        
          Tr_f(:,idx) = vlad_f;
          Tr_s(:,idx) = vlad_s;
          Tr_t(:,idx) = vlad_t;
         end

         %%%%% testing samples %%%%%%%
         Te_f = zeros(Front_Size*NVW, length(Ft));
         Te_s = zeros(Side_Size*NVW, length(St));
         Te_t = zeros(Top_Size*NVW, length(Tt));

         for idx = 1:length(Ft)
   
          nn_f = vl_kdtreequery(kdtree_f, VW_Front, Ft{idx}) ;
          nn_s = vl_kdtreequery(kdtree_s, VW_Side, St{idx}) ;
          nn_t = vl_kdtreequery(kdtree_t, VW_Top, Tt{idx}) ;
        
          assignments_f = zeros(NVW,numel(nn_f));
          assignments_s = zeros(NVW,numel(nn_s));
          assignments_t = zeros(NVW,numel(nn_t));
        
          assignments_f(sub2ind(size(assignments_f), nn_f, 1:length(nn_f))) = 1;
          assignments_s(sub2ind(size(assignments_s), nn_s, 1:length(nn_s))) = 1;
          assignments_t(sub2ind(size(assignments_t), nn_t, 1:length(nn_t))) = 1;
       
          vlad_f = vl_vlad(Ft{idx},VW_Front,assignments_f);
          vlad_s = vl_vlad(St{idx},VW_Side,assignments_s);
          vlad_t = vl_vlad(Tt{idx},VW_Top,assignments_t);
  
          Te_f(:,idx) = vlad_f;
          Te_s(:,idx) = vlad_s;
          Te_t(:,idx) = vlad_t;
         end


         Train_Desc = [Tr_f; Tr_s; Tr_t];
         Test_Desc  = [Te_f; Te_s; Te_t]; 
            
            
        end
        
        function [disc_set, Eigen_NUM] = Eigenface_f(obj, Train_SET, Type)
          
         if Type==1
             
           Eigen_NUM = obj.Numer_PCA_LBP;
           
         else
           
             Eigen_NUM = obj.Numer_PCA_HOG;
             
         end
         [NN,Train_NUM]=size(Train_SET);

         if NN<=Train_NUM % not small sample size case
    
          Mean_Image=mean(Train_SET,2);  
          Train_SET=Train_SET-Mean_Image*ones(1,Train_NUM);
          R=Train_SET*Train_SET'/(Train_NUM-1);
   
          [V,S]=obj.Find_K_Max_Eigen(R,Eigen_NUM);
          disc_value=S;
          disc_set=V;

         else % for small sample size case
     
          Train_SET=Train_SET-(mean(Train_SET,2))*ones(1,Train_NUM);
          R=Train_SET'*Train_SET/(Train_NUM-1);
         [V,S]=obj.Find_K_Max_Eigen(R,Eigen_NUM);
         clear R
         disc_set=zeros(NN,Eigen_NUM);
         Train_SET=Train_SET/sqrt(Train_NUM-1);
  
         for k=1:Eigen_NUM
          a = Train_SET*V(:,k);
          b = (1/sqrt(S(k)));
          disc_set(:,k)=b*a;
         end

        end

        
        end  
        
        function [Eigen_Vector,Eigen_Value]=Find_K_Max_Eigen(~, Matrix,Eigen_NUM)

         [NN,NN]=size(Matrix);
         [V,S]=eig(Matrix);        %Note this is equivalent to; [V,S]=eig(St,SL); also equivalent to [V,S]=eig(Sn,St); %
         S=diag(S);
         [S,index]=sort(S);

         Eigen_Vector=zeros(NN,Eigen_NUM);
         Eigen_Value=zeros(1,Eigen_NUM);

         p=NN;
         for t=1:Eigen_NUM
          Eigen_Vector(:,t)=V(:,index(p));
          Eigen_Value(t)=S(p);
          p=p-1;
         end
       end 
        
    end
    
    
end