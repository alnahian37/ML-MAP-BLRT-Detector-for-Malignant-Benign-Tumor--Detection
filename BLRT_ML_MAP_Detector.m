clc;
clear all;
close all;
load('HW3_data_benign_bash','data_benign')
load('HW3_data_malignant_bash','data_malignant')
ben=data_benign';
mal=data_malignant';
benY=zeros(100,1);
malY=ones(100,1);

figure(1)
scatter3(ben(1,:),ben(2,:),ben(3,:),'b+')
hold on
scatter3(mal(1,:),mal(2,:),mal(3,:),'r.')
legend('Benign', 'Malignant')
xlabel('x1')
ylabel('x2')
zlabel('x3')
C1 = [0.4 0.3 0.5;
    0.3 0.9 0.7;
    0.5 0.7 1.6];
mu1 = [2.7;2.8;3.6];
mu2=[5.1;5.9;5.3];
C2=[6.2 2.3 3.1;
    2.3 2.2 1.7;
    3.1 1.7 6.4];
bl1=0.88;
bl2=0.36;
map1=0.88;
map2=0.12;

W1=-.5*inv(C1);
w1=inv(C1)*mu1;
w10=-0.5*mu1'*inv(C1)*mu1-0.5*log(det(C1));
bias1=[log(bl1) log(map1) 0];

W2=-.5*inv(C2);
w2=inv(C2)*mu2;
w20=-0.5*mu2'*inv(C2)*mu2-0.5*log(det(C2));
bias2=[log(bl2) log(map2) 0];
s1="For classifier: ";
s2=["BLRT", "MAP", "ML"];

%%% For dataset 1 where each data is benign%%%%
yben=zeros(100,1);
g=zeros(100,1);
for cl=1:3
    disp(strcat(s1,s2(cl)))
    yben=zeros(100,1);
    g=zeros(100,1);
for i=1:length(ben)
    x=ben(:,i);
    g1=x'*W1*x+w1'*x+w10+bias1(cl);
    g2=x'*W2*x+w2'*x+w20+bias2(cl);
    g(i)=g1-g2;
    
    if g(i)<0
        yben(i)=1;
    else
        yben(i)=0;
    end
end
disp("Probability of detection when true class in Benign=")
disp(1-nnz(yben)/length(yben))
disp("Probability of False Alarm when true class in Benign=")
disp(nnz(yben)/length(yben))
end

%%% For dataset 2 where each data is malignant
for cl=1:3
    
    disp(strcat(s1,s2(cl)))
    ymal=zeros(100,1);
    g=zeros(100,1);
for i=1:length(mal)
    x=mal(:,i);
    g1=x'*W1*x+w1'*x+w10+bias1(cl);
    g2=x'*W2*x+w2'*x+w20+bias2(cl);
    g(i)=g1-g2;
    
    
    if g(i)>0
        ymal(i)=0;
    else
       ymal(i)=1;
    end
end
disp("Probability of Detection when true class in Malignant=")
disp(nnz(ymal)/length(ymal))
disp("Probability of False alarm when true class in Malignant=")
disp(1-nnz(ymal)/length(ymal))
end

poin=10000;

x1=(6+3).*rand(poin,1) -3;
x2=(10+8).*rand(poin,1) -8;

x31=zeros(poin,1);
x32=zeros(poin,1);

for cl=1:3
[x,y,z ]= meshgrid([-10:.1:15]); 
    
V=-0.5028.*z.*z + (-0.4217+1.016.*x+0.5.*y).* z - 2.0045.*x.*x + .3436.*y.*x-0.4931.*y.*y +6.4151.*x-1.6425.*y+1.2065+bias1(cl)-bias2(cl);
    
    
figure(1+cl);

isosurface(x,y,z,V,0)
hold on

scatter3(ben(1,:),ben(2,:),ben(3,:),'b+')
hold on
scatter3(mal(1,:),mal(2,:),mal(3,:),'r.')
legend('', 'Benign', 'Malignant')
xlabel('x1')
ylabel('x2')
zlabel('x3')
title(strcat(s1,s2(cl)))
end