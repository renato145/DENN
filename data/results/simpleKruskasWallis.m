clc;
clear;
close all;
MechName={'random'};%,'worst','mostclose'
aname={'no_nn',['nn-normal',MechName{1}],['nn-dropout', MechName{1}],['nn-distibution', MechName{1}]};
nnName={'normal','dropout','distribution'};

%mof

x(:,1)=csvread(['experiment4/no_nn_mof.csv'],1,0);
for i=1:length(nnName)
x(:,i+1)=csvread(['experiment4/nn-',nnName{i},'-',MechName{1},'_mof.csv'],1,0);
end

[p,tbl,stats] = kruskalwallis(x,aname,'off');
c=multcompare(stats,'ctype','bonferroni');%,'display','off'