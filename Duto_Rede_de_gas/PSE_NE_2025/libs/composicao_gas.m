classdef composicao_gas
    
    properties (Hidden = true,SetAccess = private)
        
        % Valores em mol da mistura de g�s
        
        CH4 = 0 % OK
        H2 = 0% OK
        N2 = 0% OK
        CO2 = 0% OK
        CO = 0% OK
        O2 = 0% OK
        NO = 0 % OK
        NO2 = 0
        SO2 = 0
        Ar = 0% OK
        He = 0% OK
        H2S = 0% OK
        C2H2 = 0% OK
        C2H4 = 0% OK
        C2H6 = 0% OK
        C3H8 = 0% OK
        C4H10 = 0% OK
        C5H12 = 0% OK
        C3H6 = 0% OK
        iC4H10 = 0% OK
        C4H8 = 0
        H2O = 0
        
        % Constantes para o c�lculo de Cp e Cv dos componentes n�o nulos
        Ap
        B
        C
        D
        E
        F
        % Constantes de Antoine dos componentes n�o nulos
        Aa
        Ba
        Ca
        Da
        Ea
        Fa
        % Constantes do m�todo de Lucas dos componentes n�o nulos
        Q
        Fq
        
    end
    
    properties (SetAccess = private)
               
        % Ordem: [CH4, H2, N2, CO2, CO, O2, NO, NO2, SO2, Ar, He, H2S, C2H2, C2H4, C2H6, C3H8, C4H10, C5H12, C3H6, iC4H10, C4H8]
       
        x % Composi��o
        
        posicoes % Posi��es das composi��es n�o nulas 
        PMt % Peso molecular da mistura
        % Propriedades puras dos componentes n�o nulos
		Tc % Temperatura cr�tica
		Pc % Press�o cr�tica
        Zc % Fator de compressibilidade cr�tico
        Vc % Volume cr�tico
        dip % Momento dipolo
		w % Fator ac�ntrico
		PM % Peso molecular
        
        % Dicion�rio das novas posi��es dos componentes n�o nulos
        dic_componentes
    end
    
    properties (SetAccess = private, GetAccess = private)
        
        %  Banco de dados das constantes Design Institute for Physical Properties (DIPPR) of the American Institute of Chemical Engineers
        Aa_bank = [39,12.6900000000000,58,140.540000000000,46,51,73,10.3800000000000,47,42,12,86,39.6300000000000,54,52,59,66,79,44,78.0100000000000,52,73.649]';
        Ba_bank = [-1324.40000000000,-94.8960000000000,-1084.10000000000,-4735,-1076.60000000000,-1200.20000000000,-2650,-2730.22580200000,-4084.50000000000,-1093.10000000000,-8.99000000000000,-3839.90000000000,-2552.20000000000,-2443,-2598.70000000000,-3492.60000000000,-4363.20000000000,-5420.30000000000,-3097.80000000000,-4634.10000000000,-4019.20000000000,-7258.2]';
        Ca_bank = [0,0,0,0,0,0,0,-39.0100000000000,0,0,0,0,0,0,0,0,0,0,0,0,0,0]';
        Da_bank = [-3.43660000000000,1,-8.31440000000000,-21.2680000000000,-4.88140000000000,-6.43610000000000,-8.26100000000000,0,-3.64690000000000,-4.14250000000000,0.672400000000000,-11.1990000000000,-2.78000000000000,-5.56430000000000,-5.12830000000000,-6.06690000000000,-7.04600000000000,-8.82530000000000,-3.44250000000000,-8.95750000000000,-4.52290000000000,-7.3037]';
        Ea_bank = [3.10000000000000e-05,0.000329000000000000,0.0441000000000000,0.0409000000000000,7.57000000000000e-05,0.0284000000000000,9.70000000000000e-15,0,1.80000000000000e-17,5.73000000000000e-05,0.274000000000000,0.0188000000000000,2.39000000000000e-16,1.91000000000000e-05,1.49000000000000e-05,1.09000000000000e-05,9.45000000000000e-06,9.62000000000000e-06,1.00000000000000e-16,1.34000000000000e-05,4.88000000000000e-16,4.1653e-6]';
        Fa_bank = [2,2,1,1,2,1,6,0,6,2,1,1,6,2,2,2,2,2,6,2,6,2]';
        %  Banco de dados das constantes do m�todo de Lucas
        Q_bank = [0 0.76 0 0 0 0 0 0 0 0 1.38 0 0 0 0 0 0 0 0 0 0 0];
        Fq_bank = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
        %          [CH4   H2     N2    CO2   CO    O2     NO    NO2    SO2   Ar     He   H2S   C2H2   C2H4   C2H6   C3H8   C4H10  C5H12  C3H6   iC4H10 C4H8]
        Tc_bank  = [190.6 41.667 126.2 304.2 132.9 154.58 180.0 431.01 430.8 150.86 5.19 373.2 308.30 282.34 305.32 369.83 425.12 469.70 364.90 407.85 419.50 647.096]; % Temperatura cr�tica em K
        Pc_bank  = [45.99 21.03 34 73.83 34.99 50.43 64.80 101.0 78.8 48.98 2.27 89.4 61.14 50.41 48.72 42.49 37.96 33.70 46.0 36.40 40.20 22.064]*100; % Press�o cr�tica em kPa
        w_bank   = [0.012 -0.216 0.038 0.224 0.048 0.0222 0.582 0.834 0.256 0.001 -0.390 0.097 0.189 0.087 0.099 0.152 0.200 0.252 0.142 0.186 0.194 0.3449]; % Fator ac�ntrico
        PM_bank  = [16.04 2.02 28.01 44.01 28.01 32 30.006 46.006 64.063 39.948 4.003 34.080 26.038 28.054 30.070 44.097 58.123 72.150 42.081 58.123 56.108 18.105]; % Peso molecular;
        Zc_bank  = [0.286 0.292 0.274 0.303 0.289 0.2888 0.251 0.473 0.269 0.291 0.301 0.284 0.268 0.282 0.279 0.276 0.274 0.268 0.280 0.278 0.278 0.229];
        Vc_bank  = [68.6 64.2 90.1 94.07 93.1 73.37 58.00 167.8 122.00 74.57 57.30 98.6 112.20 131.10 145.50 200.00 255.00 311.00 184.6 262.70 240.80 55.9472];  % cm3/mol
        dip_bank = [0.0 0.0 0.0 0.0 0.1 0.0 0.2 0.4 1.6 0.0 0.0 0.9 0.0 0.0 0.0 0.0 0.0 0.0 0.4 0.1 0.3 1.8];
        
    end 
    
    methods
        
        function obj = composicao_gas(va)
            
            teste = teste_comp(obj,va);
            
            if teste == 1
                
                names = {'CH4','H2','N2','CO2','CO','O2','NO','NO2','SO2','Ar','He','H2S','C2H2','C2H4','C2H6','C3H8','C4H10','C5H12', 'C3H6', 'iC4H10', 'C4H8', 'H2O'};

                for i = 1:length(names)

                    for j = 1:2:length(va)

                        if strcmp(names{i},va{j})

                            if strcmp(names{i},'CO2')
                                obj.CO2 = va{j+1};
                            elseif strcmp(names{i},'CO')
                                obj.CO = va{j+1} ;           
                            elseif strcmp(names{i},'O2')
                                obj.O2 = va{j+1};
                            elseif strcmp(names{i},'N2')
                                obj.N2 = va{j+1};
                            elseif strcmp(names{i},'H2')
                                obj.H2 = va{j+1};
                            elseif strcmp(names{i},'CH4')
                                obj.CH4 = va{j+1};
                            elseif strcmp(names{i},'NO')
                                obj.NO = va{j+1} ;           
                            elseif strcmp(names{i},'NO2')
                                obj.NO2 = va{j+1};
                            elseif strcmp(names{i},'SO2')
                                obj.SO2 = va{j+1};
                            elseif strcmp(names{i},'Arg')
                                obj.Arg = va{j+1};
                            elseif strcmp(names{i},'He')
                                obj.He = va{j+1};
                            elseif strcmp(names{i},'H2S')
                                obj.H2S = va{j+1};
                            elseif strcmp(names{i},'C2H2')
                                obj.C2H2 = va{j+1} ;           
                            elseif strcmp(names{i},'C2H4')
                                obj.C2H4 = va{j+1};
                            elseif strcmp(names{i},'C2H6')
                                obj.C2H6 = va{j+1};
                            elseif strcmp(names{i},'C3H8')
                                obj.C3H8 = va{j+1};
                            elseif strcmp(names{i},'C4H10')
                                obj.C4H10 = va{j+1};
                            elseif strcmp(names{i},'C5H12')
                                obj.C5H12 = va{j+1};
                            elseif strcmp(names{i},'C3H6')
                                obj.C3H6 = va{j+1};
                            elseif strcmp(names{i},'iC4H10')
                                obj.iC4H10 = va{j+1};
                            elseif strcmp(names{i},'C4H8')
                                obj.C4H8 = va{j+1};
                            elseif strcmp(names{i},'H2O')
                                obj.H2O = va{j+1};
                            end

                        end

                    end

                end
                
                x_completo = [obj.CH4 obj.H2 obj.N2 obj.CO2 obj.CO obj.O2 obj.NO obj.NO2 obj.SO2 obj.Ar obj.He obj.H2S obj.C2H2 obj.C2H4 obj.C2H6 obj.C3H8 obj.C4H10 obj.C5H12 obj.C3H6 obj.iC4H10 obj.C4H8 obj.H2O]/...
                sum([obj.CH4 obj.H2 obj.N2 obj.CO2 obj.CO obj.O2 obj.NO obj.NO2 obj.SO2 obj.Ar obj.He obj.H2S obj.C2H2 obj.C2H4 obj.C2H6 obj.C3H8 obj.C4H10 obj.C5H12 obj.C3H6 obj.iC4H10 obj.C4H8 obj.H2O]);
                				
				obj.posicoes = find(x_completo ~= 0);
				
                
                R = 8.314472;  %kJ/(kmol.K)
                %  Banco de dados das constantes de Cp e Cv
                % Ordem:    [CH4,      H2,      N2,          CO2,                       CO,             O2,     NO,   NO2,   SO2,   Arg,   He,    H2S,   C2H2,  C2H4,  C2H6,  C3H8,  C4H10, C5H12, C3H6,  iC4H10,C4H8,H2O]
                Ap_bank = R*[19.25/R, 27.14/R, 3.280, 0.6181*obj.PM_bank(4)/R, 1.074*obj.PM_bank(5)/R, 3.630, 4.354, 3.374, 4.417, 2.500, 2.500,  3.931, 6.132, 1.424, 1.131, 1.213, 1.935, 2.464, 1.637, 1.677, 1.967,4.395];
                B_bank = R*[0.05213/R, 0.0093/R, 0.593E-3, 4.845*2E-4*obj.PM_bank(4)/R, -1.727*2E-4*obj.PM_bank(5)/R, -1.794E-3, -7.644E-3, 27.257E-3, -2.234E-3, 0, 0, 1.490E-3, 1.952E-3, 14.394E-3, 19.225E-3, 28.785E-3, 36.915E-3, 45.351E-3, 22.706E-3, 37.853E-3, 31.630E-3,-4.186e-3];
                C_bank = R*[1.197E-5/R, -1.381E-5/R, 0, 3*-1.494E-7*obj.PM_bank(4)/R, 3*3.022E-7*obj.PM_bank(5)/R, 0.658E-5, 2.066E-5, -1.917E-5, 2.344E-5, 0, 0, 0, 0, -4.392E-6, -5.561E-6, -8.824E-6, -11.402E-6, -14.111E-6, -6.915E-6, -11.945E-6, -9.873E-6,1.405e-5];
                D_bank = R*[-1.132E-8/R, 0.7645E-8/R, 0, 2.291E-11*4*obj.PM_bank(4)/R, -13.75E-11*4*obj.PM_bank(5)/R, -0.601E-8, -2.156E-8, -0.616E-8, -3.271E-8, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0,-1.564e-8];
                E_bank = R*[0, 0, 0, -1.370E-15*5*obj.PM_bank(4)/R, 20.04E-15*5*obj.PM_bank(5)/R, 0.179E-11, 0.806E-11, 0.859E-11, 1.393E-11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0.632e-11];
                F_bank = R*[0 0 100000*0.040 0 0 0 0 0 0 0 0 -0.232E+5 -1.299E+5 0 0 0 0 0 0 0 0 0];
                    
                
				for i = 1:length(obj.posicoes)
                    % Vetor composi��o dos componentes n�o nulos
					obj.x(i)  = x_completo(obj.posicoes(i));
                    % Propriedades puras dos componentes n�o nulos
                    obj.Tc(i) = obj.Tc_bank(obj.posicoes(i));
					obj.Pc(i) = obj.Pc_bank(obj.posicoes(i));
                    obj.Zc(i) = obj.Zc_bank(obj.posicoes(i));
					obj.Vc(i) = obj.Vc_bank(obj.posicoes(i));
                    obj.dip(i)= obj.dip_bank(obj.posicoes(i));
					obj.w(i)  = obj.w_bank(obj.posicoes(i));
					obj.PM(i) = obj.PM_bank(obj.posicoes(i));					
                    % Constantes para o c�lculo de Cp e Cv dos componentes n�o nulos
                    obj.Ap(i) = Ap_bank(obj.posicoes(i));
                    obj.B(i) = B_bank(obj.posicoes(i));
                    obj.C(i)  = C_bank(obj.posicoes(i));					
                    obj.D(i) = D_bank(obj.posicoes(i));
                    obj.E(i)  = E_bank(obj.posicoes(i));
                    obj.F(i)  = F_bank(obj.posicoes(i));
                    % Constantes de Antoine dos componentes n�o nulos
                    obj.Aa(i) = obj.Aa_bank(obj.posicoes(i));
                    obj.Ba(i) = obj.Ba_bank(obj.posicoes(i));
                    obj.Ca(i) = obj.Ca_bank(obj.posicoes(i));
                    obj.Da(i) = obj.Da_bank(obj.posicoes(i));
                    obj.Ea(i) = obj.Ea_bank(obj.posicoes(i));
                    obj.Fa(i) = obj.Fa_bank(obj.posicoes(i));
                    % Constantes do m�todo de Lucas dos componentes n�o nulos
                    obj.Fq(i) = obj.Fq_bank(obj.posicoes(i));
                    obj.Q(i) = obj.Q_bank(obj.posicoes(i));
                    % Dicion�rio das novas posi��es dos componentes n�o nulos
                    obj.dic_componentes.(names{obj.posicoes(i)}) = i;
				end
					
				obj.PMt = sum(obj.PM.*obj.x);

            else
               
                disp('N�o vou dizer outra vez')
                
            end
            
        end
        
        function [teste] = teste_comp(obj,cell)
            
            if (-1)^(length(cell)) > 0
                
                str_var = cell(1:2:end);
                
                num_var = cell(2:2:end);
                
                for i = 1:length(num_var)
                    
                    if (ischar(str_var{i}) && isnumeric(num_var{i}))
                        
                        teste = 1;
                        
                    else
                        
                        teste = 0;
                        error('ErrorTests:convertTest','\nSiga o formato: {char,number,char,number} \n')
                        
                    end
                    
                end
                
            else
                
                teste = 0;
                error('ErrorTests:convertTest','\nSiga o formato: {char,number,char,number}. O n�mero de entradas � par\n')
                
            end
            
        end
        
        function [Cp,Cv] = ci_ideal(obj)
%             Componente: Refer�ncia do dado // Refer�ncia Valida��o

%             CH4: Prausnitz 4th // Perry Handbook
%             H2: Prausnitz 4th // Perry Handbook
%             N2: Van Ness 7th // Perry Handbook
%             CO2: Hysys // Perry Handbook
%             CO: Hysys // Perry Handbook
%             O2: Prausnitz 5th // Perry Handbook
%             NO: Prausnitz 5th // Thermodynamic Properties of Gaseous Nitrous Oxide and Nitric Oxide From Speed-of-Sound Measurements
%             NO2: Prausnitz 5th //
%             SO2: Prausnitz 5th // Perry Handbook
%             Arg: Prausnitz 5th // Perry Handbook
%             He: Prausnitz 5th // Perry Handbook
%             H2S: Van Ness 7th // Perry Handbook
%             C2H2: Van Ness 7th //
%             C2H4: Van Ness 7th // Perry Handbook
%             C2H6: Van Ness 7th // Perry Handbook
%             C3H8: Van Ness 7th // Perry Handbook
%             C4H10: Van Ness 7th // Perry Handbook
%             C5H12: Van Ness 7th // Perry Handbook
%             C3H6: Van Ness 7th // Perry Handbook
%             iC4H10: Van Ness 7th // Vapor pressure and gaseous speed of sound measurements for isobutane (R600a)
%             C4H8: Van Ness 7th // Perry Handbook

            R = 8.314472;  %kJ/(kmol.K)

            Cp = [sum(obj.Ap.*obj.x) sum(obj.B.*obj.x) sum(obj.C.*obj.x) sum(obj.D.*obj.x) sum(obj.E.*obj.x) sum(obj.F.*obj.x)];
  
            Av = obj.Ap - R;

            Cv = [sum(Av.*obj.x) sum(obj.B.*obj.x) sum(obj.C.*obj.x) sum(obj.D.*obj.x) sum(obj.E.*obj.x) sum(obj.F.*obj.x)];

        end
        
        function [visco] = sub_vis(obj,T,P)
        
            %__________________________________________________________________________
            % M�todo de Lucas
            
            % Ordem: [CH4, H2, N2, CO2, CO, O2, NO, NO2, SO2, Arg, He, H2S, C2H2, C2H4, C2H6, C3H8, C4H10, C5H12, C3H6, iC4H10, C4H8]
            No = 6.023*10^26;
            R = 8314.472;
            kte = (R*(No^2))^(1/6);
            
            Tr = T./obj.Tc;
            
            Pr = P./obj.Pc;
            
            dipr = 52.46.*(obj.dip.^2).*obj.Pc./(obj.Tc.^2);

            n = length(dipr);

            for i = 1:n

                if dipr(i)>= 0 && dipr(i)<0.022
                    Fp(i) = 1;
                else 
                    if dipr(i)>= 0.022 && dipr(i)<0.075
                        Fp(i) = 1 + 30.55*((0.292 - obj.Zc(i))^1.72);
                    else
                        if dipr(i)>=0.075
                            Fp(i) = 1 + 30.55*((0.292 - obj.Zc(i))^1.72)*abs(0.96 + 0.1*Tr(i) - 0.7);
                        else
                            Fp(i) = 0;
                        end
                    end
                end
            end

            if isfield(obj.dic_componentes,'H2')
                key = obj.dic_componentes.('H2');
                if Tr(key) < 12
                    sing = -1;
                else
                    sing = 1;
                end
                obj.Fq(key) = 1.22.*(obj.Q(key)^0.15)*(1 + 0.00385*(((Tr(key) - 12)^2)^(1/obj.PM(key)))*sing);
            end
            
            if isfield(obj.dic_componentes,'He')
                key = obj.dic_componentes.('He');
                if Tr(key) < 12
                    sing = -1;
                else
                    sing = 1;
                end
                obj.Fq(key) = 1.22.*(obj.Q(key)^0.15)*(1 + 0.00385*(((Tr(key) - 12)^2)^(1/obj.PM(key)))*sing);
            end
            
            Tcm = sum(obj.x.*obj.Tc);
            Pcm = (sum(obj.x.*obj.Pc));
            PMm = sum(obj.x.*obj.PM);
            Trm = T/Tcm;
            Fpm = sum(obj.x.*Fp);
            Fqm = sum(obj.x.*obj.Fq);
            Prm = P/Pcm;
            
            lhn = (obj.Aa + obj.Ba./(T + obj.Ca) + obj.Da*log(T) + obj.Ea.*T.^obj.Fa)';
            Pvap = exp(lhn')*1000;
            
            Pvapm = (sum(obj.x.*Pvap));
            
            epm = kte*((Tcm/((PMm^3)*((Pcm*1000)^4)))^(1/6));
            
            Z1 = Fqm*Fpm*(0.807*(Trm^0.618) - 0.357*exp(-0.449^Trm) + 0.340*exp(-4.058*Trm) + 0.018);
            
            a1 = 1.245e-3;           a2 = 5.1726;
            b1 = 1.6553;             b2 = 1.2723;
            c1 = 0.4489;             c2 = 3.0578;
            d1 = 1.7368;             d2 = 2.2310;
            e = 1.3088;
            f1 = 0.9425;             f2 = -0.1853;
            gama = -0.3286;          epsilon = -7.6351;
            delta = -37.7332;        quissi = 0.4489;

            a = (a1./Trm).*exp(a2.*(Trm.^gama));
            b = a.*(b1.*Trm - b2);
            c = (c1./Trm).*exp(c2.*(Trm.^delta));
            d = (d1./Trm).*exp(d2.*(Trm.^epsilon));
            f = f1.*exp(f2.*(Trm.^quissi));
            
            alfa = 3.262 + 14.98.*(Prm.^5.508);
            beta = 1.390 + 5.746.*Prm;
            
            
            
            for i = 1:n
                if Trm <= 1 && Prm<(Pvapm/Pcm)

                    Z2 = 0.600 + 0.760*Prm^alfa + (6.99*Prm^beta - 0.6).*(1 - Trm);

                else
                    if Trm > 1 && Prm > 0;
                        if Trm < 40 && Prm <= 100;

                            Z2 = Z1*(1 + a*(Prm^e)/(b*(Prm^f) + (1 + c*(Prm^d))^-1));
                        else
                            Z2 = 1;
                        end
                    else
                        Z2 = 1;
                    end
                end
            end

            Y = Z2./Z1;
            
            Fpp = (1 + (Fpm - 1).*(Y.^-3))./Fpm;

            Fqp = (1 + (Fqm - 1).*((Y.^-1) - 0.007*((log(Y)).^4)))./Fqm;

            visco = Z2.*Fpp.*Fqp./epm;
            
        end
        
        function [eta_m,Tc_m,Vc_m,PM_m,y_m,G1_m,dip_rm,w_m,dip_m] = sub_vis_chung(obj,T,V)
            
            R = 8314.472;
            
            V = V*10^3;
            
            %__________________________________________________________________________
            % M�todo de Chung
            
            eps_k = obj.Tc./1.2593;
            
            sigma = 0.809*obj.Vc.^(1/3);
            
            ai = [6.324 1.210*10^-3 5.283 6.623 19.745 -1.9 24.275 0.7972 -0.2382 0.06863];
            
            bi = [50.412 -1.154*10-3 254.209 38.096 7.630 -12.537 3.45 1.117 0.0677 0.3479];
            
            ci = [-51.68 -6.257*10^-3 -168.48 -8.464 -14.354 4.985 -11.291 0.01235 -0.8163 0.5926];
            
            di = [1189 0.03728 3898.0 31.42 31.53 -18.15 69.35 -4.117 4.025 -0.727];
            
            sigma_ij = (sigma'*sigma).^0.5;
            
            eps_k_ij = (eps_k'*eps_k).^0.5;
            
            W = []; M = [];
            
            for i = 1:length(obj.w);
                
                W = [W;obj.w];
                
                M = [M;obj.PM];
                
            end
            
            w_ij = (W + W')/2;
            
            PM_ij = 2*M'.*M./(M' + M);
            
            sigma_m = (obj.x*sigma_ij.^3*obj.x')^(1/3);
            
            eps_k_m = obj.x*(eps_k_ij.*sigma_ij.^3)*obj.x'/sigma_m^3;
            
            Tea_m = T/eps_k_m;
            
            PM_m = (obj.x*(eps_k_ij.*sigma_ij.^2.*PM_ij.^0.5)*obj.x'/eps_k_m/sigma_m^2)^2;
            
            w_m = obj.x*(w_ij.*sigma_ij.^3)*obj.x'/sigma_m^3;
            
            dip_m = (sigma_m^3*(obj.x.*obj.dip.^2)*(sigma_ij.^-3)*(obj.x.*obj.dip.^2)')^0.25;
            
            Omega_m = 1.16145*(Tea_m).^-0.14874  + 0.52487*exp(-0.77320*Tea_m) + 2.16178*exp(-Tea_m*2.43787);
            
            Tc_m = 1.2593*eps_k_m;
            
            Vc_m = (sigma_m/0.809)^3;
            
            dip_rm = 131.3*dip_m/(Vc_m*Tc_m)^0.5;
            
            Fc_m = 1 - 0.2756*w_m + 0.059035*dip_rm^4;
            
            y_m = Vc_m/V/6;
            
            G1_m = (1 - 0.5*y_m)/(1 - y_m)^3;
            
            E1 = ai(1) + bi(1)*w_m + ci(1)*dip_rm^4;
            
            E2 = ai(2) + bi(2)*w_m + ci(2)*dip_rm^4;
            
            E3 = ai(3) + bi(3)*w_m + ci(3)*dip_rm^4;
            
            E4 = ai(4) + bi(4)*w_m + ci(4)*dip_rm^4;
            
            E5 = ai(5) + bi(5)*w_m + ci(5)*dip_rm^4;
            
            E6 = ai(6) + bi(6)*w_m + ci(6)*dip_rm^4;
            
            E7 = ai(7) + bi(7)*w_m + ci(7)*dip_rm^4;
            
            E8 = ai(8) + bi(8)*w_m + ci(8)*dip_rm^4;
            
            E9 = ai(9) + bi(9)*w_m + ci(9)*dip_rm^4;
            
            E10 = ai(10) + bi(10)*w_m + ci(10)*dip_rm^4;
            
            G2_m = (E1*((1 - exp(-E4*y_m))/y_m) + E2*G1_m*exp(E5*y_m) + E3*G1_m)/(E1*E4 + E2 + E3);
            
            eta_aa_m = E7*y_m^2*G2_m*exp(E8 + E9*Tea_m^-1 + E10*E10*Tea_m^-2);
            
            eta_a_m = Tea_m^0.5/Omega_m*(Fc_m*(G2_m^-1 + E6*y_m)) + eta_aa_m;
            
            eta_m  = 36.344*(PM_m*Tc_m)^0.5./Vc_m^(2/3)*eta_a_m*10^-7;
            
        end
        
        function [kappa] = coef_con_ter(obj,G)
            
            [~,Cvt] = G.ci_real(obj);
            
            R = 8.314472;  %kJ/(kmol.K)
            
            [eta_m,Tc_m,Vc_m,PM_m,y_m,G1_m,dip_rm,w_m,dip_m] = sub_vis_chung(obj,G.T,G.V);
            
            Tr = G.T./Tc_m;
            
            q = 3.586*0.001*(Tc_m/PM_m).^(0.5)./(Vc_m.^(2/3));
            
            abcd = [2.4166,0.74824,-0.91858,121.72;
                    -0.50924,-1.5094,-49.991,69.983;
                    6.6107,5.6207,64.760,27.039;
                    14.543,-8.9139,-5.6379,74.344;
                    0.79274,0.82019,-0.69369,6.3173;
                    -5.8634,12.801,9.5893,65.529;
                    91.089,128.11,-54.217,523.81];
            
            B1 = abcd(1,1) + abcd(1,2)*w_m + abcd(1,3)*dip_rm^4 + abcd(1,4)*0;
            B2 = abcd(2,1) + abcd(2,2)*w_m + abcd(2,3)*dip_rm^4 + abcd(2,4)*0;
            B3 = abcd(3,1) + abcd(3,2)*w_m + abcd(3,3)*dip_rm^4 + abcd(3,4)*0;
            B4 = abcd(4,1) + abcd(4,2)*w_m + abcd(4,3)*dip_rm^4 + abcd(4,4)*0;
            B5 = abcd(5,1) + abcd(5,2)*w_m + abcd(5,3)*dip_rm^4 + abcd(5,4)*0;
            B6 = abcd(6,1) + abcd(6,2)*w_m + abcd(6,3)*dip_rm^4 + abcd(6,4)*0;
            B7 = abcd(7,1) + abcd(7,2)*w_m + abcd(7,3)*dip_rm^4 + abcd(7,4)*0;
            
            G2_m = (B1/y_m*(1 - exp(-B4*y_m)) + B2*G1_m*exp(B5*y_m) + B3*G1_m)/(B1*B4 + B2 + B3);
            
            alpha = (Cvt/R)-1.5;
            
            beta = 0.7862 - 0.7109*w_m + 1.3168*w_m^2;
            
            zeta = 2 + 10.5*Tr^2;
            
            Phi = 1 + alpha*(0.215 + 0.28288*alpha + 1.061*beta + 0.26665*zeta)./(0.6366 + beta.*zeta + 1.061*alpha.*beta);
            
            kappa = 31.2*eta_m*Phi/(PM_m*0.001)*(G2_m^-1 + B6*y_m) + q*B7*(y_m^2)*(Tr^0.5)*G2_m;
            
        end
        
        function [Pvap,dPvapdT,d2PvapdT2,d3PvapdT3] = p_vap(obj,T)
            
            if length(obj.posicoes) == 1
                
                if T >= obj.Tc
                    
                    error('ErrorTests:convertTest','\nA temperatura requisitada � maior que a temperatura cr�tica!\n')
                    
                else
                    
                    lhn = (obj.Aa + obj.Ba./(T + obj.Ca) + obj.Da*log(T) + obj.Ea*T^obj.Fa)';
                    
                    Pvap = exp(lhn)/1000;
                
                    dPvapdT = (Pvap*(-obj.Ba./(T + obj.Ca)^2 + obj.Da*1/T + obj.Ea*obj.Fa*T^(obj.Fa-1)));
                
                    d2PvapdT2 = (dPvapdT*(-obj.Ba./(T + obj.Ca)^2 + obj.Da*1/T + obj.Ea*obj.Fa*T^(obj.Fa-1))...
                                + Pvap*(2*obj.Ba./(T + obj.Ca)^3 - obj.Da*1/T^2 + obj.Ea*obj.Fa*(obj.Fa-1)*T^(obj.Fa-2)));
                            
                    d3PvapdT3 = (d2PvapdT2*(-obj.Ba./(T + obj.Ca)^2 + obj.Da*1/T + obj.Ea*obj.Fa*T^(obj.Fa-1))...
                                + 2*dPvapdT*(2*obj.Ba./(T + obj.Ca)^3 - obj.Da*1/T^2 + obj.Ea*obj.Fa*(obj.Fa-1)*T^(obj.Fa-2))...
                                + Pvap*(-6*obj.Ba./(T + obj.Ca)^4 + 2*obj.Da*1/T + obj.Ea*obj.Fa*(obj.Fa-1)*(obj.Fa-2)*T^(obj.Fa-3)));
                
                end
                    
            else
                
                error('ErrorTests:convertTest','\n� s� para substancias puras, provisoriamente! \n')
                
            end
            
            
            
        end
        
        function [eta] = vis_CO2_liq(obj,T,P)
            
            var = [-16.9530850247808,973.404368884083,0.783808918869192,-9.70547839474971e-26;];
            
            eta_vap = exp(var(1) + var(2)./T + var(3).*log(T) + var(4).*T.^10);
            
            if obj.x == 1 
                if isfield(obj.dic_componentes,'CO2')
                    
                else
                    
                    error('ErrorTests:convertTest','\nEssa rotina vale apenas para o CO2 l�quido\n')
                    
                end
                
            else
                
                error('ErrorTests:convertTest','\nEssa rotina vale apenas para o CO2 l�quido\n')
                
            end
            
            Pvap = p_vap(obj,T);
            
            APr = (P - Pvap)/obj.Pc;
            
            Tr = T/obj.Tc;
            
            Aq = 0.9991 -  (0.0004674/(1.0523*Tr^-0.03877-1.0513));
            
            Dq = 0.3257/(1.0039 - Tr^2.573)^0.2906 - 0.2086;
            
            Cq = -0.07921 + 2.1616*Tr - 13.404*Tr^2 + 44.1706*Tr^3 - 84.8291*Tr^4 + 96.1209*Tr^5 - 59.8127*Tr^6 + 15.6719*Tr^7;
            
            eta = eta_vap*(1 + Dq*(APr/2.118)^Aq)/(1 + Cq*obj.w*APr);
            
        end
        
        function [Cp,Cpt] = Cp_CO2_liq(obj,T)
                        
            if obj.x == 1 
                
                if isfield(obj.dic_componentes,'CO2')
                    
                else
                    
                    error('ErrorTests:convertTest','\nEssa rotina vale apenas para o CO2 l�quido\n')
                    
                end
                
            else
                
                error('ErrorTests:convertTest','\nEssa rotina vale apenas para o CO2 l�quido\n')
                
            end
            
            CP = [-8304.3 104.37 -0.43333 0.00060052 0];
            
            vCp = [CP(1),CP(2),CP(3),CP(4),CP(5)];
                        
            vt = [1 T T^2 T^3 T^4]';
            
            vit = [T T^2/2 T^3/3 T^4/4 T^5/5]';
            
            Cp = vCp*vt;
            
            Cpt = vCp*vit;
            
        end
        
        function [kappa,Cvt] = coef_CO2_liq(obj,G)
            
            [Cvt] = Cp_CO2_liq(obj,G.T);
            
            kappa = 0.4406 - 0.0012175*G.T;
            
        end
        
    end
    
end
