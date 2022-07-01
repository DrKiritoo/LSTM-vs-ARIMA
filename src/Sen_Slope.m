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