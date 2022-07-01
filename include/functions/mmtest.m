%% Create function for Modified Mann-Kendall test
function[H,p_value, Z, pz]=mmtest(V,alpha)
V=reshape(V,length(V),1); 
alpha = alpha/2; %
n=length(V);
i=0; j=0; S=0;
for i=1:n-1
    for j= i+1:n
        S= S + sign(V(j)-V(i));
    end
end
VarSo=(n*(n-1)*(2*n+5))/18;
%%%%%%%%%%%%%%%%%%%%%%%%
ANSW = 3;  %%% It depends on computational time
switch ANSW
    case 1
        xx=1:n;
        aa=polyfit(xx,V,1);
        yy=aa(1,1)*xx+aa(1,2);
        V=V-yy';
    case 2
        [b]=Sen_Slope(V);
        xx=1:n;
        yy=b*xx+ (mean(V) -(b*n)/2);
        V=V-yy';
    case 3
        V=detrend(V);
end
%%%%%%%%%%%%%%%%%%%%%%%%%
I = tiedrank(V); %% I = ranks
[Acx,lags,Bounds]=autocorr(I,n-1);
%[Acx,lags]=xcov(I,I,n-1,'coeff'); %%
%Acx=Acx(n:end);
ros=Acx(2:end); %% Autocorrelation Ranks
i=0; sni=0;
for i=1:n-2
    if ros(i)<= Bounds(1) && ros(i) >= Bounds(2)
        sni=sni;
    else
        sni=sni+(n-i)*(n-i-1)*(n-i-2)*ros(i);
    end
end
nns=1+(2/(n*(n-1)*(n-2)))*sni;
VarS=VarSo*(nns);
StdS=sqrt(VarS);
if S >= 0
   Z=((S-1)/StdS)*(S~=0);
else
   Z=(S+1)/StdS;
end
p_value=2*(1-normcdf(abs(Z),0,1)); 
pz=norminv(1-alpha,0,1); 
H=abs(Z)>pz; %
end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Trend Magnitude ---> Sen (1968)%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[b]=Sen_Slope(X)
i=0; 
n=length(X);
V= zeros(1,(n^2-n)/2);
for j=2:n
    for l=1:j-1
        i=i+1;
        V(i)=(X(j)-X(l))/(j-l);
    end
end
b=median(V);
end