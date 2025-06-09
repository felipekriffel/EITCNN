ex=2;
% Choose data set in experiment, the maximum number of individual 
% measurements for each experiment is: [4 6 6 4 2 7 2 6]
ver=3;

plotFlag=true
printFlag = true

% plot results
if(plotFlag)
    load data/reconstruction p e t recon
    figure(2)
    clf
    x = p(1, :)';
    y = p(2, :)';
    tri = t(1:3, :)';  % conectividade dos triângulos
    u = recon(:);      % garante formato coluna
    
    figure('Position', [100, 100, 800, 700]);
    trisurf(tri, x, y, u);
    shading interp;
    colormap jet;
    colorbar;
    view(2);           % vista 2D, como em pdeplot
    xlabel('x'); ylabel('y');      
%     colorbar off, 
    axis equal, axis off, 
end        

%printing results
if(printFlag)
    if ~exist('KIT4_recons', 'dir')
        mkdir('KIT4_recons');
    end
    filename = ['KIT4_recons/phantom_' num2str(ex) '_' num2str(ver) '.jpeg'];
    disp(filename)
    print(filename, '-djpeg');
end

