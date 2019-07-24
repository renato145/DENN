clc;
clear;
close all;
aname={'no_nn','nn'};
x(:,1)=csvread('experiment1/nn_mof.csv',1,0);
x(:,2)=csvread('experiment1/no_nn_mof.csv',1,0);
[p,tbl,stats] = kruskalwallis(x,aname,'off');
c=multcompare(stats,'ctype','bonferroni');%,'display','off'