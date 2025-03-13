classdef Soave_class
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess = private,Hidden = true) 
        ac
        bm % par�mtro de volume da equa��o de SRK para uma mistura
        am % par�metro de for�as intermoleculares da equa��o de SRK para uma mistura
        alfa
        m % Matriz de coeficientes bin�rios para a equa��o de SRK
        k % 
    end
    properties (SetAccess = private) 
        P % Press�o em kPa
        V % Volume molar em m^3/kmol
        Z % Fator de compressibilidade
        T % Temperatura em K
        x % composi��o do g�s
    end
    
    
    methods
        
        function obj = Soave_class(V,T,P,gas)
            
            obj.x = gas.x;
            
            if strcmp(T,'icog') % Teste para o caso da Temperatura ser desconhecida
                
                obj.V = V;
            
                obj.P = P;
                
                obj.T = fzero(@(T) sub_eosT(obj,T,gas),P*V/8.314472/0.95);
                                
                obj = eos_para(obj,gas);
                
            elseif strcmp(V,'icog') % Teste para o caso do Volume ser desconhecido
                
                obj.T = T;
                
                obj.P = P;
                
                obj = eos_para(obj,gas);
                
                Z = fzero(@(Z) sub_eosV(obj,Z),0.85);
                
                obj.V = 8.314472*Z*T/P;
                
            elseif strcmp(P,'icog') % Teste para o caso da Press�o ser desconhecida
                
                obj.T = T;
                
                obj.V = V;
                
                obj = eos_para(obj,gas);
                
                obj.P = eos(obj);
            
            else % Teste para o caso de V,T,P serem conhecidos
                
                obj.T = T;
                
                obj.V = V;
                
                obj = eos_para(obj,gas);
                
                if abs(P - eos(obj)) > 10^-6

                    error('ErrorTests:convertTest','\nOs valores de V,T,P n�o batem com a equa��o de SRK \n')
                    
                else
                    
                    obj.P = eos(obj);
                    
                end
                
            end
            obj.Z = obj.P*obj.V/8.314472/obj.T;
        end
        
        function [Pv] = eos_vector(obj,Vv,Tv,gas)
            
            obj.ac = 0.42747.*((R.*gas.Tc).^2)./gas.Pc;
            Tr = Tv./gas.Tc;
            obj.alfa = (1 + obj.k.*(1 - (Tr.^0.5))).^2;
            
            
        end
        
        function [saida] = ref_change(obj,V1,T1,P1,gas1)
            
           saida = Soave_class(V1,T1,P1,gas1);
            
        end
        
        function [obj] = eos_para(obj,gas) % Rotina para calcular os par�metros da equa��o de SRK            
            ke = [0.250227992676200   0.249915852480494   0.254141735830166];

            k14 = ke(1);
            k15 = ke(2);
            k45 = ke(3);

            R = 8.314472;  %kJ/(kmol.K)

            obj.ac = 0.42747.*((R.*gas.Tc).^2)./gas.Pc;

            b = 0.08664.*R.*gas.Tc./gas.Pc;

            obj.k = 0.48508 + 1.55171.*gas.w - 0.15613.*(gas.w.^2);

            Tr = obj.T./gas.Tc;

            obj.alfa = (1 + obj.k.*(1 - (Tr.^0.5))).^2;
                    1           2       3   4       5       6       7      8    9       10     11   12      13      14    15     16     17      18     19   20      21      22
            % Ordem: [CH4,     H2,    N2,   CO2,    CO,    O2,    NO,   NO2,    SO2,   Arg,   He,   H2S,   C2H2,  C2H4,  C2H6,  C3H8,  C4H10, C5H12, C3H6, iC4H10, C4H8 , H20]
            m_bank = 
                   1 [0.0000 0.0000 0.0319 0.0937 0.0300 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0780 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   2   0.0000 0.0000 -0.000   k14    k15  0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   3   0.0319 0.0000 0.0000 -0.022 0.0460 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.1400 0.0000 0.0000 0.0388 0.0807 0.1007 0.0000 0.0000 0.0000 0.0000 0.0000;
                   4   0.0973   k14 -0.0220 0.0000   k45  0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.1020 0.0000 0.0000 0.1310 0.1410 0.1420 0.1320 0.0000 0.1320 0.0000 0.0000;
                   5   0.0300   k15  0.0460   k45  0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0200 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   6   0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   7   0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   8   0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   9   0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   10   0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   11   0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   12   0.0780 0.0000 0.1400 0.1020 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0860 0.0940 0.0870 0.0700 0.0914 0.0810 0.0000 0.0000;
                   13   0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   14   0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   15   0.0000 0.0000 0.0388 0.1310 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0860 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   16   0.0000 0.0000 0.0807 0.1410 0.0200 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0940 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   17   0.0000 0.0000 0.1007 0.1420 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0870 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   18   0.0000 0.0000 0.0000 0.1320 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0700 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   19   0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0914 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   20   0.0000 0.0000 0.0000 0.1320 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0810 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   21   0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000;
                   22   0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000];
                     
            for i=1:length(gas.x)
                for j=1:length(gas.x)
                    obj.m(i,j) = m_bank(gas.posicoes(i),gas.posicoes(j));
                end
            end

            if isfield(gas.dic_componentes,'H2')
                key = gas.dic_componentes.('H2');
                if Tr(key) > 2.5
                    obj.alfa(key) = 1.202*exp(-0.30288*Tr(key));
                end
            end

            obj.bm = sum(b.*obj.x);

            a = (((obj.ac'*obj.ac).*(obj.alfa'*obj.alfa)).^0.5);

            obj.am = obj.x*(a.*(1-obj.m))*obj.x';
                       
        end
        
        function dadT = eos_dadT(obj,gas)

            dadTs4 = -0.5*(obj.T^-0.5)*((obj.ac'*obj.ac).^0.5).*((obj.k.*(gas.Tc.^-0.5))'*(obj.alfa.^0.5)+(obj.alfa.^0.5)'*(obj.k.*(gas.Tc.^-0.5)));
            
            Tr = obj.T./gas.Tc;

            if isfield(gas.dic_componentes,'H2')
                key = gas.dic_componentes.('H2');
                if Tr(key) > 2.5

                    tr = -((obj.ac.*obj.alfa).^0.5)*(obj.ac(key)^0.5)*1.202^0.5*0.5*0.30288/gas.Tc(key)*exp(-0.30288*0.5*Tr(key)) - ((obj.ac(key)*obj.alfa(key))^0.5)*(obj.ac.^0.5).*obj.k.*0.5.*((gas.Tc*obj.T).^-0.5);

                    dadTs4(key,:) = tr;

                    dadTs4(:,key) = tr';

                    dadTs4(key,key) = -obj.ac(key)*1.202*0.30288/gas.Tc(key)*exp(-0.30288*Tr(key));

                end
            end

            dadT = obj.x*(dadTs4.*(1-obj.m))*obj.x';
            
        end
        
        function [d2adT2] = eos_d2adT2(obj,gas)

            da2dTs4 = 0.25*((obj.T^3)^-0.5)*((obj.ac'*obj.ac).^0.5).*((1 + obj.k)'*(obj.k./gas.Tc.^0.5) + (obj.k./gas.Tc.^0.5)'*(1 + obj.k));
            
            Tr = obj.T./gas.Tc;

            if isfield(gas.dic_componentes,'H2')
                key = gas.dic_componentes.('H2');
                if Tr(key) > 2.5

                    part1 = ((obj.ac.*obj.alfa).^0.5)*((obj.ac(key)*1.202)^0.5)*((0.30288/gas.Tc(key))^2)*exp(-0.30288*0.5*Tr(key))*0.25;
                    part2 = (obj.T^-0.5)*(obj.ac.^0.5).*obj.k.*(gas.Tc.^-0.5)*((obj.ac(key)^0.5)*1.202^0.5*0.5*0.30288/gas.Tc(key)*exp(-0.30288*0.5*Tr(key)));
                    part3 = 0.25*((obj.T^3)^-0.5)*((obj.ac(key)*obj.alfa(key))^0.5)*(obj.ac.^0.5).*obj.k.*(gas.Tc.^-0.5);
                    tr = part1 + part2 + part3;

                    da2dTs4(key,:) = tr;

                    da2dTs4(:,key) = tr';

                    da2dTs4(key,key) = obj.ac(key)*1.202*((0.30288/gas.Tc(key))^2)*exp(-0.30288*Tr(key));

                end
            end

            d2adT2 = obj.x*(da2dTs4.*(1-obj.m))*obj.x';
        
        end
        
        function [g] = sub_eosT(obj,T,gas) % Rotina para calcular a Temeperatura via fzero
            
            % Ordem: CH4 H2 N2 CO2 CO O2

            R = 8.314472;  %kJ/(kmol.K)

            aci = 0.42747.*((R.*gas.Tc).^2)./gas.Pc;

            b = 0.08664.*R.*gas.Tc./gas.Pc;

            ki = 0.48508 + 1.55171.*gas.w - 0.15613.*(gas.w.^2);

            Tr = T./gas.Tc;

            alfai = (1 + ki.*(1 - (Tr.^0.5))).^2;
            
            if isfield(gas.dic_componentes,'H2')
                key = gas.dic_componentes.('H2');
                if Tr(key) > 2.5
                    alfai(key) = 1.202*exp(-0.30288*Tr(key));
                end
            end

            bmi = sum(b.*gas.x);

            a = (((aci'*aci).*(alfai'*alfai)).^0.5);

            ami = gas.x*(a.*(1-obj.m))*gas.x';
            
            g = obj.P - R*T/(obj.V - bmi) + ami/(obj.V^2 + obj.V*bmi);
                
        end
        
        function [g] = sub_eosV(obj,Z) % Rotina para calcular o Volume via fzero
            
            R = 8.314472;  %kJ/(kmol.K)
            
            Vs = Z*obj.T*R/obj.P;
            
            g = Z - Vs./(Vs - obj.bm) + obj.am/obj.T/R./(Vs + obj.bm);
            
        end
        
        function [Pl] = eos(obj) % Rotina para calcular a Press�o
            
            R = 8.314472;  %kJ/(kmol.K)
            
            Pl = R*obj.T/(obj.V - obj.bm) - obj.am/(obj.V^2 + obj.V*obj.bm);
             
        end
        
        function [h,s] = hs_gas(obj,gas)
            
            viT = [obj.T obj.T^2/2 obj.T^3/3 obj.T^4/4 obj.T^5/5 -1/obj.T]';
            
            vi_T = [log(obj.T) obj.T obj.T^2/2 obj.T^3/3 obj.T^4/4 -1/2/obj.T^2]';
            
            Cp = gas.ci_ideal();    CpdT = Cp*viT;  Cp_TdT = Cp*vi_T;
            
            dadT = eos_dadT(obj,gas); R = 8.314472;  %kJ/(kmol.K)
            
            hr = obj.P*obj.V - R*obj.T - (-dadT*obj.T + obj.am)*(1/obj.bm)*(log(1 + obj.bm/obj.V));

            sr = R*log((obj.V - obj.bm)*obj.P/(R*obj.T)) - (dadT)*(1/obj.bm)*(log(1 + obj.bm/obj.V));
            
            h = hr + CpdT; s = sr + Cp_TdT - R*log(obj.P);
            
        end
        
        function [h] = h_gas(obj,gas)
            
            viT = [obj.T obj.T^2/2 obj.T^3/3 obj.T^4/4 obj.T^5/5 -1/obj.T]';
            
            Cp = gas.ci_ideal();    CpdT = Cp*viT;
            
            dadT = eos_dadT(obj,gas); R = 8.314472;  %kJ/(kmol.K)
            
            hr = obj.P*obj.V - R*obj.T - (-dadT*obj.T + obj.am)*(1/obj.bm)*(log(1 + obj.bm/obj.V));
           
            h = hr + CpdT;
            
        end
        
        function [s] = s_gas(obj,gas)
            
            vi_T = [log(obj.T) obj.T obj.T^2/2 obj.T^3/3 obj.T^4/4 -1/2/obj.T^2]';
            
            Cp = gas.ci_ideal();   Cp_TdT = Cp*vi_T;
            
            dadT = eos_dadT(obj,gas); R = 8.314472;  %kJ/(kmol.K)

            sr = R*log((obj.V-obj.bm)*obj.P/(R*obj.T)) - (dadT)*(1/obj.bm)*log(1 + obj.bm/obj.V);
            
            s = sr + Cp_TdT - R*log(obj.P);
            
        end
        
        function [c] = velosom(obj,gas)
           
            R = 8.314472;  %kJ/(kmol.K)

            [~,Cv] = gas.ci_ideal();
            
            [d2adT2] = eos_d2adT2(obj,gas);
            
            dadT = eos_dadT(obj,gas);
            
            ro = 1/obj.V;
            
            vT = [1 obj.T obj.T^2 obj.T^3 obj.T^4 1/obj.T^2]';

            Cvreal = Cv*vT - obj.T*d2adT2/obj.bm*log(1 + obj.bm/obj.V);

            c2 = R*obj.T/((1 - ro*obj.bm)^2) - obj.am*ro*(2 + ro*obj.bm)/((1 + ro*obj.bm)^2) + (obj.T/(Cvreal*(ro^2)))*((R*ro/(1 - ro*obj.bm) - (ro^2)*dadT/(1 + ro*obj.bm))^2);

            c = (c2*1000/gas.PMt)^0.5;
 
        end
        
        function [Cpt,Cvt] = ci_real(obj,gas)
            
            [d2adT2] = eos_d2adT2(obj,gas);
            
            [dPdT,dPdV] = der_eos(obj,gas);
            
            [~,Cv] = gas.ci_ideal();
            
            vT = [1 obj.T obj.T^2 obj.T^3 obj.T^4 obj.T^-2]';
            
            % Cv_real = Cv_ideal + T*\int(d2PdT2)dV
            
            Cvt = Cv*vT + obj.T*d2adT2*(1/(2*2^0.5*obj.bm))*log((obj.V - obj.bm*(1 + 2^0.5))/(obj.V - obj.bm*(1 - 2^0.5)));
            
            % Cp_real = Cv_real - T*(dPdT)^2/dPdV
            
            Cpt = Cvt - obj.T*(dPdT^2)*(dPdV^-1);
            
        end
        
        function [rodP] = int_rodP(obj,g01)
            
            R = 8.314472;  %kJ/(kmol.K)
            
            part1 = R*obj.T/(obj.bm^2)*(log((1 - obj.bm/obj.V)/(1 - obj.bm/g01.V)) + obj.bm*((obj.V - obj.bm)^-1 - (g01.V - obj.bm)^-1));
            part2 = obj.am/(2*obj.bm^3)*(obj.bm^2*(obj.V^-2 - g01.V^-2) + 2*obj.bm*((obj.V + obj.bm)^-1 - (g01.V + obj.bm)^-1) - 2*log((1 + obj.bm/obj.V)/(1 + obj.bm/g01.V)));
            rodP = part1 - part2 + (obj.P - R*g01.T/(obj.V - obj.bm) + g01.am/(obj.V^2 + obj.V*obj.bm))/obj.V;
            
        end
        
        function [dPdT,dPdV,dadT] = der_eos(obj,gas)
           
            dadT = eos_dadT(obj,gas);
            
            R = 8.314472;  %kJ/(kmol.K)
            
            dPdT = R/(obj.V - obj.bm) - dadT/(obj.V^2 + obj.V*obj.bm);
            
            dPdV = -R*obj.T/(obj.V - obj.bm)^2 + obj.am*(2*obj.V + obj.bm)/(obj.V^2 + obj.V*obj.bm)^2;
            
        end
        
        function [dhdT,dhdV,dsdT,dsdV,dPdT,dPdV] = der_hs_gas(obj,gas)
           
            [dPdT,dPdV,dadT] = der_eos(obj,gas);
            
            [d2adT2] = eos_d2adT2(obj,gas);
            
            vT = [1 obj.T obj.T^2 obj.T^3 obj.T^4 1/obj.T^2]';
            
            v_T = [1/obj.T 1 obj.T obj.T^2 obj.T^3 1/obj.T^3]';
            
            Cp = gas.ci_ideal(); CpT = Cp*vT;  Cp_T = Cp*v_T;
            
            R = 8.314472;  %kJ/(kmol.K)
            
            dhrdT = dPdT*obj.V - R - (-d2adT2*obj.T)*(1/obj.bm)*(log(1 + obj.bm/obj.V));
            
            dhrdV = obj.P + obj.V*dPdV + (-dadT*obj.T + obj.am)/(obj.V^2 + obj.V*obj.bm);
            
            dsrdT = R*(1/obj.P*dPdT - 1/obj.T) - d2adT2*(1/obj.bm)*(log(1 + obj.bm/obj.V));
            
            dsrdV = R*(1/obj.V + 1/obj.P*dPdV) + dadT/(obj.V^2 + obj.V*obj.bm);
            
            dhdT = dhrdT + CpT; dhdV = dhrdV;
            
            dsdT = dsrdT + Cp_T - R*1/obj.P*dPdT; dsdV = dsrdV - R*1/obj.P*dPdV;
            
        end
        
        function [dhdT,dhdV,dPdT,dPdV] = der_h_gas(obj,gas)
           
            [dPdT,dPdV,dadT] = der_eos(obj,gas);
            
            [d2adT2] = eos_d2adT2(obj,gas);
            
            vT = [1 obj.T obj.T^2 obj.T^3 obj.T^4 1/obj.T^2]';
            
            Cp = gas.ci_ideal();    CpT = Cp*vT;
            
            R = 8.314472;  %kJ/(kmol.K)
            
            dhrdT = dPdT*obj.V - R - (-d2adT2*obj.T)*(1/obj.bm)*(log(1 + obj.bm/obj.V));
            
            dhrdV = obj.P + obj.V*dPdV + (-dadT*obj.T + obj.am)/(obj.V^2 + obj.V*obj.bm);
            
            dhdT = dhrdT + CpT; dhdV = dhrdV;
                       
        end
        
        
        function [dhdT,dhdV,dPdT,dPdV,d2PdT2,d2PdTdV,d2PdV2,dCvdT,dCvdV] = der_gas(obj,gas)
           
            [dPdT,dPdV,dadT] = der_eos(obj,gas);
            
            [d2adT2] = eos_d2adT2(obj,gas);
            
            vT = [1 obj.T obj.T^2 obj.T^3 obj.T^4 1/obj.T^2]';
            
            [Cp,Cv] = gas.ci_ideal();    CpT = Cp*vT;
            
            R = 8.314472;  %kJ/(kmol.K)
            
            d2PdT2 = - d2adT2/(obj.V^2 + obj.V*obj.bm);
            
            d2PdTdV = d2adT2*(2*obj.V + obj.bm)/(obj.V^2 + obj.V*obj.bm)^2;
            
            d2PdV2 = 2*R*obj.T/(obj.V - obj.bm)^3 + obj.am*(2/(obj.V^2 + obj.V*obj.bm)^2+2*(2*obj.V + obj.bm)^2/(obj.V^2 + obj.V*obj.bm)^3);
            
            dhrdT = dPdT*obj.V - R - (-d2adT2*obj.T)*(1/obj.bm)*(log(1 + obj.bm/obj.V));
            
            dhrdV = obj.P + obj.V*dPdV + (-dadT*obj.T + obj.am)/(obj.V^2 + obj.V*obj.bm);
            
            dhdT = dhrdT + CpT; dhdV = dhrdV;
            
            dvT = [0 1 2*obj.T 3*obj.T^2 4*obj.T^3 -2/obj.T^3]';
            
            dCvdT = Cv*dvT;
            
            dCvdV = obj.T*d2PdT2;
            
        end
        
        
        function [VdP] = int_VdP(obj,g01)
            
            R = 8.314472;  %kJ/(kmol.K)
            
            part1 = R*obj.T*(log((obj.V - obj.bm)/(g01.V - obj.bm)) - obj.bm*(1/(obj.V - obj.bm) - 1/(g01.V - obj.bm)));
            part2 = obj.am/(obj.bm)*(-log(1 + obj.bm/obj.V) + log(1 + obj.bm/g01.V) - obj.bm*(1/(obj.V + obj.bm) - 1/(g01.V + obj.bm)));
            VdP = - part1 + part2 + (obj.P - R*g01.T/(obj.V - obj.bm) + g01.am/(obj.V^2 + obj.V*obj.bm))*obj.V;
            
        end
        
        function [drodPdT drodPdV drodPdTo1 drodPdVo1] = der_int_rodP(obj,g01,gas)
            
            gto1 = eos_class(obj.V,g01.T,'icog',gas);
            
            gvo1 = eos_class(g01.V,obj.T,'icog',gas);
            
            [dPdT,dPdV,dadT] = der_eos(obj,gas);
            
            [dPdTo1] = gto1.der_eos(gas);
            
            [~,dPdVo1] = gvo1.der_eos(gas);
            
            R = 8.314472;  %kJ/(kmol.K)
            
            part1 = R/(obj.bm^2)*(log((1 - obj.bm/obj.V)/(1 - obj.bm/g01.V)) + obj.bm*((obj.V - obj.bm)^-1 - (g01.V - obj.bm)^-1));
            part2 = dadT/(2*obj.bm^3)*(obj.bm^2*(obj.V^-2 - g01.V^-2) + 2*obj.bm*((obj.V + obj.bm)^-1 - (g01.V + obj.bm)^-1) - 2*log((1 + obj.bm/obj.V)/(1 + obj.bm/g01.V)));
            
            drodPdT = part1 - part2 + dPdT;
            
            drodPdV = 1/obj.V*dPdV - 1/obj.V^2*(obj.P - R*g01.T/(obj.V - obj.bm) + (dPdV + R*g01.T/(obj.V - obj.bm)^2 - g01.am/(obj.V^2 + obj.V*obj.bm)^2*(obj.V*2 + obj.bm))/obj.V);
            
            drodPdTo1 = - 1/obj.V*dPdTo1;
            
            drodPdVo1 = - 1/g01.V*dPdVo1;
            
        end
        
        function [dhdT,dhdV,dhdP] = der_h_gastvp(obj,gas)
            
            [d2adT2] = eos_d2adT2(obj,gas);
            
            dadT = eos_dadT(obj,gas);
            
            vT = [1 obj.T obj.T^2 obj.T^3 obj.T^4 1/obj.T^2]';
            
            Cp = gas.ci_ideal();    CpT = Cp*vT;
            
            R = 8.314472;  %kJ/(kmol.K)
            
            dhrdT = - R - (-d2adT2*obj.T)*(1/obj.bm)*(log(1 + obj.bm/obj.V));
            
            dhrdV = obj.P + (-dadT*obj.T + obj.am)/(obj.V^2 + obj.V*obj.bm);
            
            dhrdP = obj.V;
            
            dhdT = dhrdT + CpT; dhdV = dhrdV; dhdP = dhrdP;
                       
        end
        
        function [dhdT,dhdP] = der_h_gasTP(obj,gas,dVdT,dVdP)
            
            [d2adT2] = eos_d2adT2(obj,gas);
            
            dadT = eos_dadT(obj,gas);
            
            vT = [1 obj.T obj.T^2 obj.T^3 obj.T^4 1/obj.T^2]';
            
            Cp = gas.ci_ideal();    CpT = Cp*vT;
            
            R = 8.314472;  %kJ/(kmol.K)
            
            dhrdT = obj.P*dVdT - R - (-d2adT2*obj.T)*(1/obj.bm)*(log(1 + obj.bm/obj.V)) + (-dadT*obj.T + obj.am)/(obj.V^2 + obj.V*obj.bm)*dVdT;
            
            dhrdP = obj.V + obj.P*dVdP + (-dadT*obj.T + obj.am)/(obj.V^2 + obj.V*obj.bm)*dVdP;
            
            dhdT = dhrdT + CpT; dhdP = dhrdP;
                       
        end
        
    end
    
end

