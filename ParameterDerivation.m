clc
clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Parameter names array
Par_list=["\lambda_{T_hM}", "\lambda_{T_hD}", "\lambda_{T_hH}", "\lambda_{T_hI_{2}}", "\delta_{T_hT_r}","\delta_{T_hIL_{10}}","\delta_{T_h}",...
          "\lambda_{T_cT_h}","\lambda_{T_cI_{2}}", "\lambda_{T_cD}", "\lambda_{T_cI_{\gamma}}","\delta_{T_cIL_{10}}","\delta_{T_CT_r}","\delta_{T_c}",...
          "\lambda_{T_rD}", "\lambda_{T_rI_2}", "\delta_{T_r}",...
          "A_{T_N}","\delta_{T_N}",...
          "\lambda_{MT_h}", "\lambda_{MI_\gamma}","\lambda_{MIL_{10}}","\delta_{M}",...
          "A_{M_N}","\delta_{M_N}",...
          "\lambda_{DH}","\lambda_{DC}","\delta_{DC}","\delta_{D}",...
          "A_{D_N}", "\delta_{D_N}",...
          "\lambda_{C}","\lambda_{CIL_6}","C_0","\delta_{CT_c}","\alpha_{T_c}","\beta_C","\delta_{CI_\gamma}","\delta_{C}",...
          "\alpha_{NC}","\delta_{N}",...
          "\lambda_{I_\gammaT_c}","\lambda_{I_\gammaT_h}","\lambda_{I_\gammaD}","\delta_{I_\gamma}",...
          "\lambda_{HT_c}", "\lambda_{HT_r}","\lambda_{HT_h}", "\lambda_{HN}","\lambda_{HM}", "\lambda_{HC}","\delta_{H}",...
          "\lambda_{IL_{10}T_h}", "\lambda_{IL_{10}T_c}", "\lambda_{IL_{10}D}", "\lambda_{IL_{10}M}","\delta_{IL_{10}}",...
          "\lambda_{I_2T_c}","\lambda_{I_2T_h}", "\lambda_{I_2D}", "\lambda_{I_2M}","\delta_{I_{2}}",...
          "\lambda_{IL_6C}","\lambda_{IL_6M}","\lambda_{IL_6T_h}","\lambda_{IL_6D}","\delta_{IL_6}"];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Reading the parameter values acquired by scaling deltaDC-deltaD-Assumption
%by 1,0.2 and 5 respectively
for i=1:4
    M1(:,i) = csvread("Cluster"+string(i)+"-deltaDC-deltaD-Assumption Scale-"+string(1)+".csv");
    M2(:,i) = csvread("Cluster"+string(i)+"-deltaDC-deltaD-Assumption Scale-"+string(0.2)+".csv");
    M3(:,i) = csvread("Cluster"+string(i)+"-deltaDC-deltaD-Assumption Scale-"+string(5)+".csv");
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Reading the parameter values acquired by scaling lambdaIL6Th-lambdaIL6C
%assumption by 1,0.2 and 5 respectively
for i=1:4
    T1(:,i) = csvread("Cluster"+string(i)+"-lambdaIL6Th-lambdaIL6C-Assumption Scale-"+string(1)+".csv");
    T2(:,i) = csvread("Cluster"+string(i)+"-lambdaIL6Th-lambdaIL6C-Assumption Scale-"+string(0.2)+".csv");
    T3(:,i) = csvread("Cluster"+string(i)+"-lambdaIL6Th-lambdaIL6C-Assumption Scale-"+string(5)+".csv");
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Reading the parameter values acquired by scaling lambdaIL6D-lambdaIL6C
%assumption by 1,0.2 and 5 respectively
for i=1:4
    N1(:,i) = csvread("Cluster"+string(i)+"-lambdaIL6D-lambdaIL6C-Assumption Scale-"+string(1)+".csv");
    N2(:,i) = csvread("Cluster"+string(i)+"-lambdaIL6D-lambdaIL6C-Assumption Scale-"+string(0.2)+".csv");
    N3(:,i) = csvread("Cluster"+string(i)+"-lambdaIL6D-lambdaIL6C-Assumption Scale-"+string(5)+".csv");
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Converting color codes to RGB
str = '#3F9B0B';
color1 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#FF796C';
color2 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#0343DF';
color3 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#000000';
color4 = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plotting the deviations from the original parameter values
X = 1:67;
figure
hold on
b=barh(X,M1(:,:)-M2(:,:),'stacked','BarWidth',0.5)
c=barh(X,M1(:,:)-M3(:,:),'stacked','BarWidth',0.5)
d=barh(X,T1(:,:)-T2(:,:),'stacked','BarWidth',0.5)
e=barh(X,T1(:,:)-T3(:,:),'stacked','BarWidth',0.5)
f=barh(X,N1(:,:)-N2(:,:),'stacked','BarWidth',0.5)
g=barh(X,N1(:,:)-N3(:,:),'stacked','BarWidth',0.5)

b(1).FaceColor = color1;b(2).FaceColor = color2;b(3).FaceColor = color3;b(4).FaceColor = color4;
c(1).FaceColor = color1;c(2).FaceColor = color2;c(3).FaceColor = color3;c(4).FaceColor = color4;
d(1).FaceColor = color1;d(2).FaceColor = color2;d(3).FaceColor = color3;d(4).FaceColor = color4;
e(1).FaceColor = color1;e(2).FaceColor = color2;e(3).FaceColor = color3;e(4).FaceColor = color4;
f(1).FaceColor = color1;f(2).FaceColor = color2;f(3).FaceColor = color3;f(4).FaceColor = color4;
g(1).FaceColor = color1;g(2).FaceColor = color2;g(3).FaceColor = color3;g(4).FaceColor = color4;

yticks([1:67])
yticklabels(Par_list)
legend('Cluster1','Cluster2','Cluster3','Cluster4')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

