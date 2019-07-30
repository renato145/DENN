clc;
clear;
close all;
%take masterstrMerr for the first table
%take masterstrStat for second table
% FitnessSharing 
filename='DCMA-ES resultsv2.xlsx';
aname={'CLS','Crowding','Fitnessdiv','No-div','Opp', 'RI'};


%%%%i added
sevFName={'small','medium','large'};

fname={'G24\_u','G24\_1','G24\_f','G24\_uf','G24\_2','G24\_2u','G24\_3','G24\_3b','G24\_3f','G24\_4',...
       'G24\_5','G24\_6a','G24\_6b','G24\_6c','G24\_6d','G24\_7','G24\_8a','G24\_8b',...
       'G24v\_3','G24v\_3b','G24w\_3','G24w\_3b'};
conInd=[2,3,5,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22];
set=6;   



%%%%%changes i made to creade data
Mof=[];
%c=[se,'/', alName{1},'offerrors.csv'];
for j=1:length(aname)   
    Moft=csvread([sevFName{2},'/', aname{j},'trackerr.csv']);%offerrors, arr, feval,trackerr    
    Moft=Moft(:,1:30);
    data(j,:,:)=Moft;
    %Mof(j)=[Mof,Moftt];
   %Mof(j,:,:)=[Mof,Moft(:,1)];
end

%%%%%%%%%end of changes

for i=1:length(fname)%NguyenBenchmark + Bu functions
    for k=1:length(aname)
        Function(i).y(k,:)=data(k,i,:);
    end
end
masterstrStat='\hline\textbf{Functions}& \textbf{Statistical Test}\\\hline';
masterstrMerr='\hline\multirow{2}{*}{\textbf{Algorithms}}&\multicolumn{6}{c}{\textbf{Functions}}\\ ';
for numF=1:length(fname)
%for j=1:10
str=strcat('\textbf{',fname{numF},'}&');
x=Function(numF).y;
% fr=Function(numF).frate;
%figure(j)

%hold o
%subplot(10,4,j)
[p,tbl,stats] = kruskalwallis(x',aname,'off'); %Los algoritmos comparados

c=multcompare(stats,'ctype','bonferroni','display','off');
% title('Functions G24\_1 Frequency 1000 evals, k=0.5 and S=20','FontSize',18)

ind=find(c(:,end)<=0.05);
ad=false(length(c(:,end)));ad(ind)=true;
for k=1:length(ad)
    if ~ad(k)
        p=ranksum(x(c(k,1),:),x(c(k,2),:));
        if p<=0.05
            ad(k)=true;
        end
    end
end
%ind=find(ad==true);
for k=1:length(ind)
    if c(ind(k),4)>0
        str=strcat(str,'',num2str(c(ind(k),1)),'$>$',num2str(c(ind(k),2)),',');
    else
        str=strcat(str,'',num2str(c(ind(k),2)),'$>$',num2str(c(ind(k),1)),',');
    end
end
if length(ind)>0
    masterstrStat=[masterstrStat,char(13),strcat(str(1:end-1),'\\\hline')];
else
    masterstrStat=[masterstrStat,char(13),strcat(str,'\\\hline')];
end
%end
end
rows=ceil(length(fname)/set);
count=1;
for r=1:rows
    masterstrMerr=[masterstrMerr,'&'];
    cols=min([count+set-1,length(fname)]);substr='';
    for fnum=count:cols
        substr=[substr,'\textbf{',fname{fnum},'}&'];
    end
    masterstrMerr=[masterstrMerr,newline,substr(1:end-1),'\\\hline '];substr='';
    for subr=1:length(aname)
        substr=[substr,aname{subr},'&'];
        for fnum=count:cols
            m=mean((Function(fnum).y)');
            s=std((Function(fnum).y)');
            [~,ind]=min(m);
            if subr==ind
                substr=[substr,'\textbf{',num2str(m(subr),'%.4f'),'($\pm$',num2str(s(subr),'%.3f'),')}&'];
            else
                substr=[substr,num2str(m(subr),'%.4f'),'($\pm$',num2str(s(subr),'%.3f'),')&'];
            end
        end
        substr=[substr(1:end-1),newline,'\\'];
    end
    count=fnum+1;
    masterstrMerr=[masterstrMerr,newline,substr,'\hline '];
end

