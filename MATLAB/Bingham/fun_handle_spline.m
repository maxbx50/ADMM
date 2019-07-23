function val = fun_handle_spline(p)
%Function handle f for a reference solution u

syms x y;
epsi = 0.125;
shift = 0.125;
beta = 0.8;
mu = 1.0;

%u = piecewise(epsi<sqrt((x-0.5)^2+(y-0.5)^2)<epsi+shift,1024*sqrt((x-0.5)^2+(y-0.5)^2)^3-576*((x-0.5)^2+(y-0.5)^2)+96*sqrt((x-0.5)^2+(y-0.5)^2)-4,(sqrt((x-0.5)^2+(y-0.5)^2)<=epsi),1.0,0.0);

u = 1024*sqrt((x-0.5)^2+(y-0.5)^2)^3-576*((x-0.5)^2+(y-0.5)^2)+96*sqrt((x-0.5)^2+(y-0.5)^2)-4;

d = gradient(u);
normd = norm(d);

divsd = piecewise(epsi<sqrt((x-0.5)^2+(y-0.5)^2)<epsi+shift & normd>beta,divergence(d/normd),0.0);
L = piecewise(epsi<sqrt((x-0.5)^2+(y-0.5)^2)<epsi+shift,divergence(d),0.0);


%sdm = matlabFunction(divsd);
%Lm = matlabFunction(L);

x = p(:,1);
y = p(:,2);


%val = @(x,y) -0.8*(sqrt((x-0.5).^2+(y-0.5).^2)>0.125).*(sqrt((x-0.5).^2+(y-0.5).^2)<0.25).*sdm(x,y) - (sqrt((x-0.5).^2+(y-0.5).^2)>0.125).*(sqrt((x-0.5).^2+(y-0.5).^2)<0.25).*Lm(x,y);
val = -beta*subs(divsd) - mu*subs(L);

end