function [U_newBase]=transfrom_Adj2Trig(U_in,L,Ntrig)


%linearly independent patterns
nBasis=L-1;


% Build adjacent current pattern matrix
Aad=diag(ones(L,1));
Aad(2:L,1:L-1)=Aad(2:L,1:L-1)+diag(ones(L-1,1)*(-1));
Aad(1,L)=-1;


%Normalization
Aad=Aad/sqrt(2);

%Normalization for trigonometric functions
normTrig=sqrt(2/L);

Atrig=zeros(L,nBasis);
ll=(0:L-1)';
%Angles of electrode start
th=2*pi*ll/L;

%Cosine
for k=1:L/2-1
    Atrig(:,k)=normTrig*cos(Ntrig(k)*th);
%Alternating in the middle
end
    k=L/2;
    Atrig(:,k)=1/sqrt(L)*cos(Ntrig(k)*th);
%Sine
for k=(L/2+1):nBasis
    Atrig(:,k)=normTrig*sin(Ntrig(k)*th);
end


%Compute coefficients for change of basis 
coeff = zeros(nBasis);

for ii = 1:L
    for j=1:nBasis
        coeff(j,ii) = Atrig(:,j).'*Aad(:,ii);
    end
end


%Check that Voltages sum to zero
meanVol=sum(U_in);

for iii=1:size(U_in,2)
    if meanVol(iii)>eps  %If sum greater than machine precision
        U_in(:,iii)=U_in(:,iii)-meanVol(iii); 
    end
end

%Solve matrix equation to obtain representation in trig basis
U_newBase=U_in/(coeff);

end