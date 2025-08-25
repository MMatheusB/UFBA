classdef simulator_duct
    
    properties
        
        
    end
    
    methods
        
        function s = simulator_duct(eos,gas,solo,E)
            
            
            
            
        end
        
        
        function [dydx] = edo_gasoduto(s,x,y,info,yo)
            
            m = info.m; AX = info.AX; Tinf = info.Tinf;
            
            theta = y(1); phi = y(2); nu = y(3);
            
            To1 = yo(1); Vo1 = yo(2); wo1 = yo(3);
            
            T = y(1)*yo(3); V = y(2)*yo(2); m = y(3)*yo(3);
            
            pm = s.gas.PMt;

            G = equacao_estado(V,T,'icog',s.gas,s.eos);
            
            kappa = s.gas.coef_con_ter(Go1);
            
            rho = pm/Vo1; R = 8.314472;  %kJ/(kmol.K)
            
            mu = s.gas.sub_vis(To1,Go1.P);
            
            [Cpt,Cvt] = G.ci_real(obj.gas);
            
            kappa = s.gas.coef_con_ter(G); alpha = kappa/(1000*Cpt*rho);
            
            Re = w*rho*Din/mu; 
            
            ff = 4*(-4*log10(E/Din/3.7 - 5.02/Re*log10(E/Din/3.7 - 5.02/Re*log10(E/Din/3.7 + 13/Re))))^-2;
            
            ft = (1.82*log10(Re) - 1.64)^-2;
            
            Nu = (ft/8)*(Re - 1000)*Pr/(1.07 + 12.7*((ft/8)^0.5)*(Pr^(2/3) - 1)); 
            
            h = Nu*kappa/Din;
            
            Up = pi/(1/(h*obj.Din) + acosh(uza)/(obj.solo.kappa*2));
            
            [dPdT,dPdV] = der_eos(G,gas);
            
            g = 1000/pm; epss = V/R*dPdT; a = -V/T*dPdV/dPdT;
            
            rg = g*R*To1/wo1^2; Gamma = R/Cvt; fm = ff/2;
            
            rE = fm*Gamma/rg; Cal = 
            
            
        end
        
        
        function [] = edo_diff_finitas(s,t,y,f,y0,yend,ord,x)
            
            if (-1)^ord < 0
                ord = ord + 1;
            end
            
            n = length(x);
            
            T = y(1:n); V = (n+1:2*n); w = y(2*n+1:end);
            
            y = [T V w]; dzdt = [];
            
            for i = 1:n
                
                if i-1 < ord/2
                    as = der_yvar(x(i),x(1:ord+1));
                    mult = y(1:ord+1,:)'.*(ones(3,1)*as);
                    dydx = sum(mult')';
                elseif n-i < ord/2
                    as = der_yvar(x(i),x(n-ord:n));
                    mult = y(n-ord:n,:)'.*(ones(3,1)*as);
                    dydx = sum(mult')';
                else
                    as = der_yvar(x(i),x(i-ord/2:i+ord/2));
                    mult = y(i-ord/2:i+ord/2,:)'.*(ones(3,1)*as);
                    dydx = sum(mult')';
                end
                
                [~,A,B] = f(y(i,:));
                dydt(1) = (A(1,:)*dydx + B(1));
                dydt(2) = (A(2,:)*dydx + B(2));
                dydt(3) = (A(3,:)*dydx + B(3));
                
                dzdt = [dzdt dydt'];
                
            end
            
            dzdt = [dzdt(1,:) dzdt(2,:) dzdt(3,:)]';
            
            pos0 = find(y0 == 0); posend = find(yend == 0);
            
            if isempty(find(y0+yend == 0))
                
            end
            
            for i = pos0
                dzdt((i-1)*n + 1) = y(1,i) - y0(i);
            end
            
            for i = posend
                dzdt((i-1)*n) = y(n,i) - yend(i);
            end
            
        end
        
        function [as] = der_yvar(s,x0,xe)
            X = x0 - xe;
            n = length(xe);
            Xe = (xe'*ones(1,n))' - xe'*ones(1,n) + eye(n);
            Pe = prod(Xe');
            aza = ones(n,1)*X - diag(X) + eye(n);
            pos = find(aza == 0);
            if isempty(pos)
                azul = prod(aza');
                az = azul'*ones(1,n);
                co = az./aza; co = co - diag(diag(co));
                zap = co;
            else
                nl = pos(end)/n;
                nl = round(nl); 
                azaaux = aza; azaaux(pos) = 1;
                azul = prod(azaaux');
                az = azul'*ones(1,n);
                co = az./azaaux; co = co - diag(diag(co));
                zap = zeros(n); zap(pos) = co(pos); zap(nl,:) = co(nl,:);
            end
            as = sum(zap)./Pe;
        end
        
    end
    
end