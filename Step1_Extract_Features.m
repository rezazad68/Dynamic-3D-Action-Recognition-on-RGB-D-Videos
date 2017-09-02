close all
clear all
clc

%% Create an object from class Video Analyser %%

Vid_Analyser = Video_Analyser ;
Subject_Index =[];Sample_label=[];

%% Extract video list for MSR Action 3D dataset that contains 557 3D videos%%

[ Videos_List,Video_Count ] = Vid_Analyser.Get_Video_List();
 
for idx = 1:Video_Count
  
  %% Showing the progress %%
  
  Vid_Analyser.Show_Text(Video_Count, idx);
  Video = Vid_Analyser.Read_Video(Videos_List{idx});
  
  %% Summarizing video in 3 different representation %%
  
  [long_vid, mid_vid,short_vid] = Vid_Analyser.Summarizer (Video);
  
  %% Extracting Temporal Videos from each of summarized video
  
  [ Temporal_Videos_LBP] = Vid_Analyser.Temporal_Sequences( short_vid, mid_vid, long_vid, 1);
  [ Temporal_Videos_HOG] = Vid_Analyser.Temporal_Sequences( short_vid, mid_vid, long_vid, 2);
  
  %% Binary waited mapping of temporal videos
  
  [Frontal_pak_LBP, Side_pak_LBP, Top_pak_LBP] = Vid_Analyser.Temporal_Mapping(Temporal_Videos_LBP);
  [Frontal_pak_HOG, Side_pak_HOG, Top_pak_HOG] = Vid_Analyser.Temporal_Mapping(Temporal_Videos_HOG);
  
  %% Extracting regional local binary patterns histograms
  
  [Frontal_RLBP] = Vid_Analyser.Total_RLBP(Frontal_pak_LBP, 1);
  [Side_RLBP] = Vid_Analyser.Total_RLBP(Side_pak_LBP, 2);
  [Top_RLBP] = Vid_Analyser.Total_RLBP(Top_pak_LBP, 3);
   
  %% Extracting regional histogram of oriented gradiants histograms
  
  [Frontal_RHOG] = Vid_Analyser.Total_RHOG(Frontal_pak_HOG, 1);
  [Side_RHOG] = Vid_Analyser.Total_RHOG(Side_pak_HOG, 2);
  [Top_RHOG] = Vid_Analyser.Total_RHOG(Top_pak_HOG, 3);
  
  %% Paking the Extracted features
  
  Total_Frontal_RLBP {idx} = Frontal_RLBP;
  Total_Side_RLBP {idx} = Side_RLBP;
  Total_Top_RLBP {idx} = Top_RLBP;
  Total_Frontal_RHOG {idx} = Frontal_RHOG;
  Total_Side_RHOG {idx} = Side_RHOG;
  Total_Top_RHOG {idx} = Top_RHOG;
  
  %% Extract data label and its related actor
  
  [Index, Label]= Vid_Analyser.Instance_Lebels(Videos_List{idx});
  Subject_Index = [Subject_Index, Index];
  Sample_label = [Sample_label, Label];  
  
  
end

%% Save extrated features

save Action3D_Features -v7.3 





