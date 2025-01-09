# -*- coding: utf-8 -*-
"""
Created on Mon May 20 22:29:39 2024

@author: Rodrigo Meira
"""

from eos_database import *
from casadi import *
from numpy import exp, log, array, roots
from scipy.optimize import fsolve

R = 8.31446261815324

class Species:
    
    def __init__(self,name,MM,Tc,Pc,Vc,omega):
        self.name = name     # Species' name
        self.Tc = Tc         # Critical temperature    (K)
        self.Pc = Pc         # Critical pressure       (kPa)
        self.Vc = Vc         # Critical volumn         (m³/kmol)
        self.MM = MM         # Molecular mass          (kg/kmol)
        self.omega = omega   # Acentricity             adm
        self.rho_c = 1/Vc    # Critical specific mass  (kmol/m³)
        
        self.Zc = Pc*Vc/R/Tc # Critical compressibility factor
        self.x = 0
    
    def set_cp_ig_function(self,fun_cp,fun_enthalpy):
        self.cp_ig_fun = fun_cp                # A CasADi function for the Cp^IG specie
        self.enthalpy_ig_fun = fun_enthalpy    # A CasADi function for the h^IG specie
        
    def set_entropy_ig_function(self, fun_entropy):
        self.entropy_ig_fun = fun_entropy      # A CasADi function for the s^IG specie
        
    def set_p_vap_function(self,fun):
        self.pvap_fun = fun     # A CasADi function for the P_vap specie
        
    def set_composition(self,x):
        self.x = x
    
    def set_kappa(self,fun):
        self.kappa_pr = fun 
    
    def evaluate_kappa(self):
        self.kappa = float(self.kappa_pr(self.omega))
    
    def evaluate_cp_ig(self,T):
        return array(self.cp_ig_fun(T))[0][0]    

    def evaluate_enthalpy_ig(self,T):
        return array(self.enthalpy_ig_fun(T))[0][0]
    
    def evaluate_entropy_ig(self,T):
        return array(self.entropy_ig_fun(T))[0][0]
    
    def evaluate_pvap(self,T):
        return array(self.pvap_fun(T))[0][0]
    
    
class Mixture:
    
    def __init__(self,list_of_species,dict_composition):
        
        sum_x = sum(dict_composition.values())
        
        [species.set_composition(dict_composition[species.name]/sum_x) for species in list_of_species]
        self.dict_composition = dict_composition
        self.list_of_species = list_of_species
        self.list_of_names = list(dict_composition.keys())
        self.MM_m = sum([species.x*species.MM for species in list_of_species])
        self.list_Pc = [species.Pc for species in list_of_species]
        self.list_Tc = [species.Tc for species in list_of_species]
        self.list_Vc = [species.Vc for species in list_of_species]
        self.list_Zc = [species.Zc for species in list_of_species]
        self.list_kappa = [species.kappa for species in list_of_species]
        self.Cp_ig_fun = [species.cp_ig_fun for species in list_of_species]
        self.enthalpy_ig_fun = [species.enthalpy_ig_fun for species in list_of_species]
        self.pvap_ig_fun = [species.pvap_fun for species in list_of_species]
        self.x = [species.x for species in self.list_of_species]
        w_x = [species.x*species.MM for species in self.list_of_species]
        self.w = [i/sum(w_x) for i in w_x]
        
    def evaluate_cp_ig(self,T):
        return sum(array(self.x)*array([float(species.evaluate_cp_ig(T)) for species in self.list_of_species]))
        
    def evaluate_enthalpy_ig(self,T):
        return sum(array(self.x)*array([float(species.evaluate_enthalpy_ig(T)) for species in self.list_of_species]))
        
    def evaluate_pvap(self,T):
        return [float(species.evaluate_pvap(T)) for species in self.list_of_species]
    
    def copy(self):
        return Mixture(self.list_of_species,self.dict_composition)
    
    def setup_entropy_ig_fun(self):
        self.entropy_ig_fun = [species.entropy_ig_fun for species in self.list_of_species]
        
    def evaluate_entropy_ig(self,T):
        return sum(array(self.x)*array([float(species.evaluate_entropy_ig(T)) for species in self.list_of_species]))
    
    def copy_and_change_composition(self,x):
        names = list(self.dict_composition.keys())
        dict_composition_new = {names[i]: x[i] for i in range(len(names))}
        
        return Mixture(self.list_of_species,dict_composition_new)
    
