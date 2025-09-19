function tri_index = get_tri_index(nodes, tri, mx,my)
  nx = size(mx, 1);
  ny = size(mx, 2);
  
  tri_index = zeros(nx,ny);

  for j = 1:nx
    for k = 1:ny
      pk = [mx(j,k) my(j,k)];
      %display(['checking' num2str(pk)]);
      % Procurar o triângulo que contém o ponto se ta dentro do circulo
      if pk(1)**2 + pk(2)**2<=1+1e-12
        for t = 1:size(tri,2)
          % Coordenadas dos vértices do triângulo
          v = nodes(:,tri(1:3,t))';
          % Matrizes para coordenadas barycêntricas
          A = [v(1,:) - v(3,:); v(2,:) - v(3,:)]';
          b = (pk - v(3,:))';

          % Resolver para coordenadas barycêntricas
          lambda = A \ b;
          lambda(3) = 1 - sum(lambda);

          % Verificar se o ponto está dentro do triângulo (todos lambdas entre 0 e 1)
          if all(lambda >= -1e-12) && all(lambda <= 1+1e-12)
            tri_index(j,k) = t;
            break;
          endif
        endfor
       else
         tri_index(j,k) = -1;
       endif
    endfor
  endfor
endfunction
