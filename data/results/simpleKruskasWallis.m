clc;
clear;
close all;
aname={'no_nn','nn-random','nn-mostclose','nn-worst', 'nn-distribution'};


%mof

x(:,1)=csvread('experiment1/no_nn_mof.csv',1,0);
x(:,2)=csvread('experiment1/nn-random_mof.csv',1,0);
x(:,3)=csvread('experiment1/nn-mostclose_mof.csv',1,0);
x(:,4)=csvread('experiment1/nn-worst_mof.csv',1,0);
x(:,5)=csvread('experiment1/nn-distribution_mof.csv',1,0);

%successrate
%{
x(:,1)=csvread('experiment1/no_nn_sr.csv',1,0);
x(:,2)=csvread('experiment1/nn-random_sr.csv',1,0);
x(:,3)=csvread('experiment1/nn-mostclose_sr.csv',1,0);
x(:,4)=csvread('experiment1/nn-worst_sr.csv',1,0);
x(:,5)=csvread('experiment1/nn-distribution_sr.csv',1,0);
%}
[p,tbl,stats] = kruskalwallis(x,aname,'off');
c=multcompare(stats,'ctype','bonferroni');%,'display','off'