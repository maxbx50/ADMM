function val = fun( p )
%FUN
%   Function handle for the source term
%
%   INPUT:   p      vector defining the points
%
%   OUTPUT:  val    f(p)
%

val = 10.*sin(p(:,1)*3*3.1416).*cos(p(:,2)*2*3.1416).*(2-0.5*p(:,1).*p(:,2));

%val = - 2.*p(:,1).*(p(:,1) - 1) - 2.*p(:,2).*(p(:,2) - 1);

%val = ones(size(p,1),1) + 0*p(:,1)+0*p(:,2);





