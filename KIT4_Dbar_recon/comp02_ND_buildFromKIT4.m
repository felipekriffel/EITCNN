% Building ND-matrix from KIT4 measurements.
% First we need to perform a change of basis, then we can compute the inner
% products with respect to the chosen cosine/sine basis.
%
% Andreas Hauptmann, 2017


%Choose if you want to plot the current patterns
plotPattern=true;

%load saved data from KIT4
load data/KIT4_measurement U_ad10 U_ad0

% Number of electrodes
L=16;

load data/elecMeas_adj Ntrig
load data/mesh

%First with target
%Scaling of the electrodes by 
U_in=(U_ad0)*(2*pi/(2*L)); %Length of electrode

%Reordering to conform with our geometry
U_in=[U_in(9:L,:);U_in(1:8,:)];
Uel=transfrom_Adj2Trig(U_in,L,Ntrig);


%First with target
%Scaling of the electrodes by 
U_in=(U_ad10)*(2*pi/(2*L)); %Length of electrode

%Reordering to conform with our geometry
U_in=[U_in(9:L,:);U_in(1:8,:)];
Uel1=transfrom_Adj2Trig(U_in,L,Ntrig);



 
%%

% load angular values for grid points
load data/angPars_adj fii Dfii
fiiBase=fii; %For ND map
fii=fii-pi; %For traces

Nfii=length(fii);

%Get angular values for electrode positions
theta=electrodeBuilder(L);

%Middle points of electrodes
thetaMid=(theta(:,1)+theta(:,2))/2;
%Middle between electrodes
gammaMid=thetaMid+theta(1,2);

%Periodicity
gammaMid=[gammaMid;gammaMid(1)+2*pi];
theta=[theta;theta(1,:)];
Uel=[Uel;Uel(1,:)];
Uel1=[Uel1;Uel1(1,:)];

%Init
u_trace=zeros(length(fii),L-1);
u1_trace=zeros(length(fii),L-1);
u_diffC=zeros(length(fii),L-1);


%Order of basis functions
nBasis=length(Ntrig);


NtoD=zeros(L-1,L-1);
NtoD1=zeros(L-1,L-1);



for N=1:length(Ntrig)
        
    nn=1;
    for iii=1:Nfii

           
       if fii(iii)<=gammaMid(nn)
           u_trace(iii,N)=Uel(nn,N);
           u1_trace(iii,N)=Uel1(nn,N);

       else
 
           nn=nn+1;
           u_trace(iii,N)=Uel(nn,N);
           u1_trace(iii,N)=Uel1(nn,N);

           
       end
    end
    
    %Again to make sure we are mean free
    u_trace(:,N)=u_trace(:,N)-mean(u_trace(:,N));
    u1_trace(:,N)=u1_trace(:,N)-mean(u1_trace(:,N));


    if(plotFlag)
    figure(1)
    clf
    plot(fiiBase,u_trace(:,N))
    hold on
    plot(fiiBase,u1_trace(:,N))
     drawnow,drawnow,

    end
    
    
     for jjj = 1:length(Ntrig)
        if jjj<=(nBasis+1)/2
            NtoD(jjj,N)  = (Dfii.*(cos(Ntrig(jjj)*fiiBase)))'*u_trace(:,N);
            NtoD1(jjj,N)  = (Dfii.*(cos(Ntrig(jjj)*fiiBase)))'*u1_trace(:,N);
        else
            NtoD(jjj,N)  = (Dfii.*(sin(Ntrig(jjj)*fiiBase)))'*u_trace(:,N);
            NtoD1(jjj,N)  = (Dfii.*(sin(Ntrig(jjj)*fiiBase)))'*u1_trace(:,N);
        end
     end

    
     
     
end

% 
%save data/ND_by_KIT_full NtoD_diff Nvec u_trace u1_trace Ntrig
save data/ND NtoD NtoD1 Ntrig


