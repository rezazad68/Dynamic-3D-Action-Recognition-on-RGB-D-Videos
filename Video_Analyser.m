%%% Dynamic 3D Hand Gesture and Action Recognition with Learning Spatio-Temporal Aggregation from Different Representation %%%
						%## R. Azad, M. Asadi and S. Kasaei ##%
                   % will be submitted toIEEE Transactions on CSVT % 
                   
          %% Video Analysis class for extracting features %%
          
% This Class contain function as follow:
% 1- Get_Video_List: for extracting list of videos that exist in MSR Action 3D data set
% 2- Read_Video: reading one video with specified address
% 3- Remove_empty_Frames: some 3D depth videos contains empty value, this
% function is defined for removing those frames
% 4- Summarizer: summarizing video in 3 different representation
% 5- Image_resize: resizing image size
% 6- Forward_Backward_Motion: extracting forward and backward motion for
% each frame of video
% 7- bounding_box: for extracting object position
% 8- Temporal_Sequences: extracting temporal videos with overlaping,
% parameters such as Hierarchical_{TL_LBP, NT_LBP, TL_HOG and NT_HOG}
% relats to this function
% 9- Binary_Weighted_Mapping: encoding video with binary weighter mapping
% 10- Total_RHOG and Total_RLBP for extracting regional features


classdef Video_Analyser
    
   properties
       
       Data_sufix = '/*.mat';
       Vid_prefix = 'MSR-Action3D';
       Fix_Size = [100 100];
       Front_view= [102 54];
       Side_view = [102 75];
       Top_view = [75 54];
       Block_Size_LBP = [4, 4, 3; 2, 3, 2];
       Block_Size_HOG = [4, 4, 3; 2, 3, 2];
       Cell_Size = [9, 8, 9; 9, 9, 9];
       Shift_LBP = [7, 7, 7];
       Shift_HOG = [7, 7, 7];
       LBPRadius = 4;
       Data_set_address = 'MSR-Action3D'; 
       Hierarchical_TL_LBP = [8, 8, 16]; 
       Hierarchical_NT_LBP = [5, 5, 5];
       Hierarchical_TL_HOG = [4, 8, 16]; 
       Hierarchical_NT_HOG = [5, 5, 5];


   end
   
   methods

       function [ Videos_List,Video_Count ] = Get_Video_List( obj)
        
           Videos_List = [];
           Address = strcat(obj.Data_set_address,obj.Data_sufix);
           temp_List = dir(Address);
           [Video_Count,~] = size(temp_List);
           for idx = 1:Video_Count
             Videos_List{idx} = temp_List(idx,1).name;  
           end

       end
       function [Video] = Read_Video (obj, Vide_name)
     
          load (strcat(obj.Vid_prefix,'/',Vide_name))
          Video = depth; 
       end
       function [] = Show_Text (~, Total,now)
               disp(strcat('Total Videos: ',int2str(Total),', Video Under Process is: ', int2str(now)));
       end
       function [New_Video] = Remove_empty_Frames (~, Input_Video)
           
           [~,~,count]=size(Input_Video);
           T=1;
           New_Video = [];
            for i=1:count
                if sum(sum(Input_Video(:,:,i)))>0   
                    New_Video(:,:,T)=Input_Video(:,:,i);T=T+1;
                end
            end
       
       end
       function [long_vid, mid_vid,short_vid] = Summarizer (obj, Input_Video)         
            New_video = obj.Remove_empty_Frames (Input_Video);
            [~,~,count]=size(New_video);
            long_vid = New_video;
            if count > 10
               
              [Forward_energy,Backward_energy ] = obj.Forward_Backward_Motion(New_video);
              
              for i=1:count-3
                    
                  Diff_Forw_En(i)=abs(Forward_energy(i)-Forward_energy(i+1));

              end
              Diff_Forw_En(count-3)=0;
              Diff_Forw_En(1)=0;
              
              %% Get Mid and Short Video
            
              Video_counter = 1;
              [~, vid_len]= size(Diff_Forw_En);
              
              for idx=1:2
                Convert_size = floor(vid_len/(idx + 1));
                Diff_Forw_En_Temp=Diff_Forw_En;
                f_cand=[];
                candidate_list=[];
                new_vid=[];

                for idy=1:Convert_size
                    [~,index_max]=max(Diff_Forw_En_Temp);
                    candidate_list(idy)=index_max;
                    Diff_Forw_En_Temp(index_max)=0;
                end

                candidate_list=sort(candidate_list);
                f_cand(1)=1;f_cand(2:Convert_size+1)=candidate_list;f_cand(Convert_size+2)=count-2;

                for i=1:Convert_size
                    new_vid(:,:,i)=New_video(:,:,f_cand(i)+1);
                end

                total_vid{Video_counter}=  new_vid ;
                Video_counter = Video_counter +1;

              end
            mid_vid = total_vid{1};
            short_vid = total_vid{2};
           else
            mid_vid = New_video;
            short_vid = New_video;
           end

       end
       function [New_img] = Image_resize(~, Img, new_size)
           
           New_img = imresize(Img,new_size, 'bicubic');
           New_img = New_img(:);
           mask = New_img;
           if min(New_img) < 0
           New_img = New_img + abs(min(New_img)) * 2;
           end
           New_img(mask==0) = 0;
           New_img = (New_img-min(New_img)) ./ (max(New_img)-min(New_img));
           New_img = reshape(New_img, new_size(1), new_size(2));
           
       end
       function [Forward_energy, Backward_energy ] = Forward_Backward_Motion(obj, Video)
          
          [~,~,count]=size(Video);

          for idx = 1 :  count - 2
           
              Motion_T1 = abs(Video(:, :, idx+1) - Video(:, :, idx));
              Motion_T1 = obj.bounding_box(Motion_T1);
              Motion_T2 = abs(Video(:, :, idx+2) - Video(:, :, idx+1));
              Motion_T2 = obj.bounding_box(Motion_T2);
              
              Motion_T1 = obj.Image_resize(Motion_T1, obj.Fix_Size);
              Motion_T2 = obj.Image_resize(Motion_T2, obj.Fix_Size);
              
              Forward_motion= zeros(obj.Fix_Size);
              Backward_motion = zeros(obj.Fix_Size);
              
              for i= 1: obj.Fix_Size(1)
                for j= 1: obj.Fix_Size(2)
        
                  if Motion_T1(i,j)==0 && Motion_T2(i,j)~=0
                    Forward_motion(i,j)=Motion_T2(i,j); 
                  end
                 
                  if Motion_T1(i,j)~=0 && Motion_T2(i,j)==0
                    Backward_motion(i,j)=Motion_T1(i,j); 
                  end
                 
                end
              end
              
              Forward_energy(idx)=sum(sum(Forward_motion));
              Backward_energy(idx)=sum(sum(Backward_motion));
             
          end 
           
       end
       function [Output_Image] = bounding_box(~, Input_Image)

       rows_sum = sum(Input_Image');
       cloumn_sum = sum(Input_Image);
       
       idx = size(cloumn_sum,2);
       while cloumn_sum(idx)<=0
           idx = idx-1;          
       end
       Right = idx;
       
       idx = 1;
       while cloumn_sum(idx)<=0
           idx = idx+1;          
       end
       Left = idx;
       
       idx = 1;
       while rows_sum(idx)<=0
           idx = idx+1;          
       end
       Top = idx;
       
       idx = size(rows_sum,2);
       while rows_sum(idx)<=0
           idx = idx-1;          
       end
       Bottom = idx;
       
       Output_Image = Input_Image(Top:Bottom, Left:Right);
       end
       function [ Temporal_Videos] = Temporal_Sequences( obj, short_vid, mid_vid, long_vid, Type)
           %Front_BWM, Side_BWM, Top_BWM
           %% Extracting Temporal Sequences
           Hierarchical_Videos{1} = short_vid;
           Hierarchical_Videos{2} = mid_vid;
           Hierarchical_Videos{3} = long_vid;
           [~, Hierarchical_count] = size(Hierarchical_Videos);
           Temporal_Video_Count = 0;
           
           for idx=1:Hierarchical_count

            if Type==1   
                
             Temporal_length = obj.Hierarchical_TL_LBP (idx);
             Temporal_Next_Start = obj.Hierarchical_NT_LBP (idx);
            
            else
                
             Temporal_length = obj.Hierarchical_TL_HOG (idx);
             Temporal_Next_Start = obj.Hierarchical_NT_HOG (idx);
                
            end
            
            
            Video = Hierarchical_Videos{idx};
            [~,~, Frames_Count] = size(Video);
        
            if Temporal_length > Frames_Count
            Temporal_length = Frames_Count;
            end
            
            idy = 1;
            while ( (idy + Temporal_length- 1) <= Frames_Count)
            
            Temporal_Video_Count = Temporal_Video_Count + 1;
            Temporal_Videos{Temporal_Video_Count} = Video(:, :, idy: idy -1 + Temporal_length);
            idy = idy + Temporal_Next_Start;
            
            end
        

           end
            
           
           
       end
       function [Frontal_pak, Side_pak, Top_pak] = Temporal_Mapping(obj, Temporal_Videos)
       
          [~, Total_Videos] = size(Temporal_Videos);
          
          for idx=1: Total_Videos
           
            Video = Temporal_Videos {idx};
            
            [Front_BWM, Side_BWM, Top_BWM] = obj.Binary_Weighted_Mapping(Video);
            Frontal_pak(:, :, idx) = obj.Image_resize(Front_BWM, obj.Front_view);  
            Side_pak(:, :, idx) = obj.Image_resize(Side_BWM, obj.Side_view); 
            Top_pak(:, :, idx) = obj.Image_resize(Top_BWM, obj.Top_view); 
                        
          end
 
       end
       function [Front_BWM, Side_BWM, Top_BWM] = Binary_Weighted_Mapping(obj, Video)
           
           %% BWM is a revised version of DMM
           [rows,cols,D] = size(Video);
           X2D = reshape(Video, rows*cols, D);
           max_depth = max(X2D(:));

           Front_BWM = zeros(rows, cols);
           Side_BWM = zeros(rows, max_depth);
           Top_BWM = zeros(max_depth, cols);

           for k = 1:D   
            front = Video(:,:,k);
            side = zeros(rows, max_depth);
            top = zeros(max_depth, cols);
    
            for i = 1:rows
             for j = 1:cols
              if front(i,j) ~= 0
                side(i,front(i,j)) = j;   % side view projection (y-z projection)
                top(front(i,j),j)  = i;   % top view projection  (x-z projection)
              end
             end
            end
            front = front >0;
            side = side >0;
            top = top >0;
    
            if k > 1
             Front_BWM = Front_BWM + (abs(front - front_pre)*2^(k-1));
             Side_BWM = Side_BWM + (abs(side - side_pre)*2^(k-1));
             Top_BWM = Top_BWM + (abs(top - top_pre)*2^(k-1));
            end   
        
            front_pre = front;
            side_pre  = side;
            top_pre   = top;
           
           end

            Front_BWM = obj.bounding_box(Front_BWM);
            Side_BWM = obj.bounding_box(Side_BWM);
            Top_BWM = obj.bounding_box(Top_BWM);
           
           
       end
       function [Subject_Index, Sample_Label]= Instance_Lebels(~, Text_Sequence)
           
          Subject_Index = str2double(Text_Sequence(6:7));
          Sample_Label =  str2double(Text_Sequence(2:3)); 
           
       end
       function [RHOG] = Regional_HOG(~, Image, Block_Size, Cell_Size, Shift)
           
           [m,n] = size(Image);
           xsize = round(m/Block_Size(1));
           ysize = round(n/Block_Size(2));
           s = 1;
           while s+xsize-1<=m
                s = s + Shift;
           end
           m = s+xsize-1;

           s = 1;
           while s+ysize-1<=n
            s = s + Shift;
           end
           n = s+ysize-1;

           Image = imresize(Image, [m,n], 'bicubic');
           RHOG = [];
           for i = 1:Shift:m-xsize+1
            for j = 1:Shift:n-ysize+1
                
             blk = Image(i:i+xsize-1, j:j+ysize-1);
             HOG = double(extractHOGFeatures(blk,'CellSize',Cell_Size'));
             RHOG = [RHOG, HOG(:)];
            
            end
           end
 
       end
       function [HOG_Description] = Total_RHOG(obj, Image_Pak, Pak_View)
          
           [~, ~, Total_Images] = size (Image_Pak);
           HOG_Description = [];
           
           for idx=1:Total_Images
               
              RHOG = obj.Regional_HOG(Image_Pak(:, :, idx), obj.Block_Size_HOG(:, Pak_View), obj.Cell_Size(:, Pak_View), obj.Shift_HOG(Pak_View));
               
              HOG_Description = [HOG_Description, RHOG]; 
           end
           
           
           
       end
       function [LBP_Description] = Total_RLBP(obj, Image_Pak, Pak_View)
          
           [~, ~, Total_Images] = size (Image_Pak);
           LBP_Description = [];
           
           for idx=1:Total_Images
               
              RLBP = obj.Regional_LBP(Image_Pak(:, :, idx), obj.Block_Size_LBP(:, Pak_View), obj.Shift_LBP(Pak_View));
               
              LBP_Description = [LBP_Description, RLBP]; 
           end
             
           
           
       end
       function [RLBP] = Regional_LBP(obj, Image, Block_Size, Shift)
           
           [m,n] = size(Image);
           xsize = round(m/Block_Size(1));
           ysize = round(n/Block_Size(2));
           s = 1;
           while s+xsize-1<=m
                s = s + Shift;
           end
           m = s+xsize-1;

           s = 1;
           while s+ysize-1<=n
            s = s + Shift;
           end
           n = s+ysize-1;

           Image = imresize(Image, [m,n], 'bicubic');
           RLBP = [];
           for i = 1:Shift:m-xsize+1
            for j = 1:Shift:n-ysize+1
                
             blk = Image(i:i+xsize-1, j:j+ysize-1);
             LBP = double(extractLBPFeatures(blk, 'Radius', obj.LBPRadius));
             RLBP = [RLBP, LBP(:)];
            
            end
           end
 
       end
       
       
       
   end
   
   
end



