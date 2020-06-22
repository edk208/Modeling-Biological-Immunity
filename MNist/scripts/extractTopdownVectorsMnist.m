function extractTopdownVectors()
%EXTRACTHALLEBERRYVECTORS Summary of this function goes here
%   Detailed explanation goes here

addpath('../../../mlab/util/');

workspace_path = '../../../';

[inputData, inputHdr] = readpvpfile(['../outputOne/P1.pvp']);
[vData, vHdr] = readpvpfile(['../outputOne/V1.pvp']);

Zeronum = zeros(1,128);
Onenum = zeros(1,128);
allzero=[];
allone=[];

Zerov = zeros(1,4096);
Onev = zeros(1,4096);
allZerov=[];
allOnev=[];

fid = fopen('train.txt');
labels = textscan(fid,'%s %d');
countzero =0;
countone = 0;

for i=1:60000
    if labels{2}(i) == 4
    X(sub2ind([1,128],inputData{i,1}.values(:,1)+1)) = inputData{i,1}.values(:,2);
    X(numel(Zeronum)) = 0;
    Zeronum = Zeronum + X;
    allzero = [allzero; X];
    countzero = countzero + 1;
    end
    
    if labels{2}(i) == 5
    Y(sub2ind([1,128],inputData{i,1}.values(:,1)+1)) = inputData{i,1}.values(:,2);
    Y(numel(Onenum)) = 0;
    Onenum = Onenum + Y;
    allone = [allone; Y];
    countone = countone + 1;
    end
   
    if labels{2}(i) == 4
    X2(sub2ind([1,4096],vData{i,1}.values(:,1)+1)) = vData{i,1}.values(:,2);
    X2(numel(Zerov)) = 0;
    Zerov = Zerov + X2;
    allZerov = [allZerov; X2];
    end

    
    if labels{2}(i) == 5
    Y2(sub2ind([1,4096],vData{i,1}.values(:,1)+1)) = vData{i,1}.values(:,2);
    Y2(numel(Onev)) = 0;
    Onev = Onev + Y2;
    allOnev = [allOnev; Y2];
    end
end
figure;
Zeronum= Zeronum/countzero;
Onenum = Onenum/countone;

Zerov= Zerov/countzero;
Onev = Onev/countone;
%Xnew(Xnew<0.04) =0;
%Notnew(Notnew<0.04) = 0;
set(gcf,'color','w');
%polarplot(B, 'r');
bar([Zeronum',Onenum']);
figure;


set(gcf,'color','w');
bar([Zerov',Onev']);

Fournum = Zeronum;
Fivenum = Onenum;

Fourv = Zerov;
Fivev = Onev;

save('four.mat','Fournum');
save('five.mat','Fivenum');

save('fourv.mat','Fourv');
save('fivev.mat','Fivev');


end

