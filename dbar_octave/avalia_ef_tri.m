function vals = avalia_ef_tri(nodes, tri, u, mx,my,tri_index)
  nx = size(mx, 1);
  ny = size(mx, 2);
  
  vals = NaN(nx,ny);
  ones = [1;1;1];
  for j=1:nx
    for k = 1:ny
      pk = [mx(j,k) my(j,k)];
      % Procurar o triângulo que contém o ponto
      tk = tri_index(j,k);
      if tk==-1||tk==0
        vals(j,k) = 0;
      else
        % Coordenadas dos vértices do triângulo
        v = nodes(:,tri(1:3,tk))';
        % Matrizes para coeficientes
        A = [v(:,1) v(:,2) ones];
        b = u(tri(1:3,tk));
        % Resolver para obter os coeficientes
        coef = A \ b;
        vals(j,k) = coef(1)*pk(1) + coef(2)*pk(2) + coef(3);
      endif      
    endfor
  endfor  
endfunction
