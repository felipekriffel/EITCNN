%% Building some electrodes
function theta=electrodeBuilder(L)

theta=zeros(2,L);

anglVar=2*pi/(2*L);

theta(:)=[0:2*L-1]*anglVar;

theta=theta';


end