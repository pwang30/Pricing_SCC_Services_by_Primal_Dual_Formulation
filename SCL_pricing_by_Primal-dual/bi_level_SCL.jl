# Author: Peng Wang       from Technical University of Madrid (UPM)
# Supervisor: Luis Badesa

# Pricing SCL by primal-dual formulation
# 29.May.2025

import Pkg
using JuMP,Gurobi, CSV,DataFrames,LinearAlgebra, XLSX, IterTools, DelimitedFiles,Plots,MAT, CPLEX, Ipopt
include("dataset_gene.jl")
include("offline_trainning.jl")
include("admittance_matrix_calculation.jl") 
# SGs, buses:2,3,4,5,27,30    IBRs, buses:1,23,26



#-----------------------------------Define Parameters for Calculating SCL-----------------------------------
I_IBG=1      # pre-defined SCL contribution from IBG
Iₗᵢₘ= 5       # SCL limit
β=0.95       # percentage of nominal voltage, range from 0.95-1.1
v_n=1        # nominal voltage
v=0.1        # gap for classification 

I_SCC_all_buses_scenarios, matrix_ω =dataset_gene(I_IBG, β,v_n)                                                            # data set generation                      
K_g, K_c, K_m, N_type_1, N_type_2, err_type_1, err_type_2= offline_trainning(I_SCC_all_buses_scenarios, matrix_ω, Iₗᵢₘ, v)  # offline_trainning




#-----------------------------------Define Parameters for Optimization-----------------------------------  
      
Load_total=[18.42,17.95,18.29,18.51,18.13,17.88,19.46,21.97,23.17,23.87,
23.91,23.77,23.80,23.82,24.23,23.79,26.01,26.91,25.26,23.69,22.12,20.04,18.17,18.01]*10^3/3.5   # (MW)
T=length(Load_total)

IBG₁=250*ones(T,1)   
IBG₂₃=250*ones(T,1)
IBG₂₆=250*ones(T,1)


Pˢᴳₘₐₓ=[6.584, 5.760, 3.781, 3.335, 3.252, 2.880]*10^3/5            # Max generation of SGs                    SGs, buses:2,3,4,5,27,30
Pˢᴳₘᵢₙ=[3.292, 2.880, 1.512, 0.667, 0.650, 0.288]*10^3/5            #  Min generation of SGs                   SGs, buses:2,3,4,5,27,30
Rₘₐₓ=[1.317, 1.152, 1.512, 1.334, 1.951, 1.728]*10^3/5              #  Ramp limits of SGs                      SGs, buses:2,3,4,5,27,30
Kˢᵗ=[200, 125, 92.5, 72, 55, 31]*10^3 /10                           #  Startup cost of SGs                     SGs, buses:2,3,4,5,27,30
Kˢʰ=[50, 28.5, 18.5, 14.4, 12, 10]*10^3  /10                        #  Shutdown cost of SGs                    SGs, buses:2,3,4,5,27,30
Oᵐ₁=[6.20, 7.10, 10.47, 12.28, 13.53, 15.36]                        #  Marginal generation cost of SG 1  in    SGs, buses:2,3,4,5,27,30
Oᵐ₂=[7.07, 8.72, 11.49, 12.84, 14.60, 15.02]                        #  Marginal generation cost of SG 2  in    SGs, buses:2,3,4,5,27,30
Oⁿˡ=[17.431, 15.005, 13.755, 10.930, 9.900, 8.570]*10^3 /10         #  No-load cost of SGs                     SGs, buses:2,3,4,5,27,30
Oᴱ_c=[2.46 3.19 2.86]                                               #  Price offer of SGs                      IBRs, buses:1,23,26
P_g₀=[5.268 4.608 3.025 2.668 2.602 0]*10^3/5                       #  Initial generation (t=0) of SGs         SGs, buses:2,3,4,5,27,30
yˢᴳ₀=[1 1 1 1 1 0]  



#-----------------------------------Define Primal-Dual Model-----------------------------------
model= Model()
#-------Define Primal Variales
 
@variable(model, Pˢᴳ²_1[1:T])        # generation of SGs , buses:2,3,4,5,27,30.  
@variable(model, Pˢᴳ²_2[1:T])  
@variable(model, Pˢᴳ³_1[1:T])     
@variable(model, Pˢᴳ³_2[1:T])                
@variable(model, Pˢᴳ⁴_1[1:T])   
@variable(model, Pˢᴳ⁴_2[1:T])             
@variable(model, Pˢᴳ⁵_1[1:T])    
@variable(model, Pˢᴳ⁵_2[1:T])            
@variable(model, Pˢᴳ²⁷_1[1:T]) 
@variable(model, Pˢᴳ²⁷_2[1:T])                
@variable(model, Pˢᴳ³⁰_1[1:T])                
@variable(model, Pˢᴳ³⁰_2[1:T])  

@variable(model, Pᴵᴮᴳ¹[1:T]>=0)         # generation of IBRs (WT) , buses:1, 23, 26 , dual variables: ζᵐⁱⁿₜ  
@variable(model, Pᴵᴮᴳ²³[1:T]>=0)              
@variable(model, Pᴵᴮᴳ²⁶[1:T]>=0)  

@variable(model, yˢᴳ²_1[1:T],Bin)      # status of SGs, buses:2,3,4,5,27,30.  Include strategic and non-strategic players
@variable(model, yˢᴳ²_2[1:T],Bin)          
@variable(model, yˢᴳ³_1[1:T],Bin)  
@variable(model, yˢᴳ³_2[1:T],Bin)          
@variable(model, yˢᴳ⁴_1[1:T],Bin)      
@variable(model, yˢᴳ⁴_2[1:T],Bin)      
@variable(model, yˢᴳ⁵_1[1:T],Bin)      
@variable(model, yˢᴳ⁵_2[1:T],Bin)      
@variable(model, yˢᴳ²⁷_1[1:T],Bin)  
@variable(model, yˢᴳ²⁷_2[1:T],Bin)          
@variable(model, yˢᴳ³⁰_1[1:T],Bin)   
@variable(model, yˢᴳ³⁰_2[1:T],Bin)                       

@variable(model, α₁[1:T]>=0)                   # percentage of IBGs' online capacity , dual variables:   φᵐⁱⁿₜ
@variable(model, α₂₃[1:T]>=0)
@variable(model, α₂₆[1:T]>=0)

@variable(model, Cᵁ²_1[1:T]>=0)                 # startup costs and shutdown costs for SGs , dual variables: ρˢᵗₜ , ρˢʰₜ
@variable(model, Cᵁ²_2[1:T]>=0)                 
@variable(model, Cᴰ²_1[1:T]>=0)                 
@variable(model, Cᴰ²_2[1:T]>=0)                 
@variable(model, Cᵁ³_1[1:T]>=0)    
@variable(model, Cᵁ³_2[1:T]>=0)             
@variable(model, Cᴰ³_1[1:T]>=0)  
@variable(model, Cᴰ³_2[1:T]>=0)               
@variable(model, Cᵁ⁴_1[1:T]>=0)     
@variable(model, Cᵁ⁴_2[1:T]>=0)            
@variable(model, Cᴰ⁴_1[1:T]>=0)   
@variable(model, Cᴰ⁴_2[1:T]>=0)              
@variable(model, Cᵁ⁵_1[1:T]>=0)  
@variable(model, Cᵁ⁵_2[1:T]>=0)               
@variable(model, Cᴰ⁵_1[1:T]>=0) 
@variable(model, Cᴰ⁵_2[1:T]>=0)                
@variable(model, Cᵁ²⁷_1[1:T]>=0)    
@variable(model, Cᵁ²⁷_2[1:T]>=0)             
@variable(model, Cᴰ²⁷_1[1:T]>=0)    
@variable(model, Cᴰ²⁷_2[1:T]>=0)             
@variable(model, Cᵁ³⁰_1[1:T]>=0)      
@variable(model, Cᵁ³⁰_2[1:T]>=0)            
@variable(model, Cᴰ³⁰_1[1:T]>=0)    
@variable(model, Cᴰ³⁰_2[1:T]>=0)            

@variable(model, I_₁₁[1:T])             # define SCL on bus 26,29,30 , dual variables: λ_₂₆, λ_₂₉, λ_₃₀, λ_lim₂₆, λ_lim₂₉, λ_lim₃₀
@variable(model, I_₂₆[1:T])
@variable(model, I_₂₉[1:T])
@variable(model, I_₃₀[1:T])

@variable(model, η_m[1:66,1:T],Bin)         #McCormick for SCL constraints, i.e., products of binary variables



#-------Define Dual Variales
@variable(model, λᴱ[1:T]>=0)            # price for market clearing                
@variable(model, λ_F[1:4,1:T]>=0)       # price for SCL AS      n=1 for bus 26 ; n=2 for bus 29  ; n=3 for bus 30 

@variable(model, ζᵐᵃˣ¹[1:T]>=0) 
@variable(model, ζᵐᵃˣ²³[1:T]>=0)
@variable(model, ζᵐᵃˣ²⁶[1:T]>=0)

@variable(model, φᵐᵃˣ¹[1:T]>=0)
@variable(model, φᵐᵃˣ²³[1:T]>=0)
@variable(model, φᵐᵃˣ²⁶[1:T]>=0)

@variable(model, μᵐⁱⁿˢᴳ²_1[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ²_1[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ²_2[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ²_2[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ³_1[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ³_1[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ³_2[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ³_2[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ⁴_1[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ⁴_1[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ⁴_2[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ⁴_2[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ⁵_1[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ⁵_1[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ⁵_2[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ⁵_2[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ²⁷_1[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ²⁷_1[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ²⁷_2[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ²⁷_2[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ³⁰_1[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ³⁰_1[1:T]>=0)
@variable(model, μᵐⁱⁿˢᴳ³⁰_2[1:T]>=0)
@variable(model, μᵐᵃˣˢᴳ³⁰_2[1:T]>=0)

@variable(model, πʳᵈˢᴳ²_1[1:T]>=0)
@variable(model, πʳᵘˢᴳ²_1[1:T]>=0)
@variable(model, πʳᵈˢᴳ²_2[1:T]>=0)
@variable(model, πʳᵘˢᴳ²_2[1:T]>=0)
@variable(model, πʳᵈˢᴳ³_1[1:T]>=0)
@variable(model, πʳᵘˢᴳ³_1[1:T]>=0)
@variable(model, πʳᵈˢᴳ³_2[1:T]>=0)
@variable(model, πʳᵘˢᴳ³_2[1:T]>=0)
@variable(model, πʳᵈˢᴳ⁴_1[1:T]>=0)
@variable(model, πʳᵘˢᴳ⁴_1[1:T]>=0)
@variable(model, πʳᵈˢᴳ⁴_2[1:T]>=0)
@variable(model, πʳᵘˢᴳ⁴_2[1:T]>=0)
@variable(model, πʳᵈˢᴳ⁵_1[1:T]>=0)
@variable(model, πʳᵘˢᴳ⁵_1[1:T]>=0)
@variable(model, πʳᵈˢᴳ⁵_2[1:T]>=0)
@variable(model, πʳᵘˢᴳ⁵_2[1:T]>=0)
@variable(model, πʳᵈˢᴳ²⁷_1[1:T]>=0)
@variable(model, πʳᵘˢᴳ²⁷_1[1:T]>=0)
@variable(model, πʳᵈˢᴳ²⁷_2[1:T]>=0)
@variable(model, πʳᵘˢᴳ²⁷_2[1:T]>=0)
@variable(model, πʳᵈˢᴳ³⁰_1[1:T]>=0)
@variable(model, πʳᵘˢᴳ³⁰_1[1:T]>=0)
@variable(model, πʳᵈˢᴳ³⁰_2[1:T]>=0)
@variable(model, πʳᵘˢᴳ³⁰_2[1:T]>=0)

@variable(model, σˢᵗˢᴳ²_1[1:T]>=0)
@variable(model, σˢʰˢᴳ²_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ²_2[1:T]>=0)
@variable(model, σˢʰˢᴳ²_2[1:T]>=0)
@variable(model, σˢᵗˢᴳ³_1[1:T]>=0)
@variable(model, σˢʰˢᴳ³_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ³_2[1:T]>=0)
@variable(model, σˢʰˢᴳ³_2[1:T]>=0)
@variable(model, σˢᵗˢᴳ⁴_1[1:T]>=0)
@variable(model, σˢʰˢᴳ⁴_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ⁴_2[1:T]>=0)
@variable(model, σˢʰˢᴳ⁴_2[1:T]>=0)
@variable(model, σˢᵗˢᴳ⁵_1[1:T]>=0)
@variable(model, σˢʰˢᴳ⁵_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ⁵_2[1:T]>=0)
@variable(model, σˢʰˢᴳ⁵_2[1:T]>=0)
@variable(model, σˢᵗˢᴳ²⁷_1[1:T]>=0)
@variable(model, σˢʰˢᴳ²⁷_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ²⁷_2[1:T]>=0)
@variable(model, σˢʰˢᴳ²⁷_2[1:T]>=0)
@variable(model, σˢᵗˢᴳ³⁰_1[1:T]>=0)
@variable(model, σˢʰˢᴳ³⁰_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ³⁰_2[1:T]>=0)
@variable(model, σˢʰˢᴳ³⁰_2[1:T]>=0)

@variable(model, ψᵐᵃˣˢᴳ²_1[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ²_2[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ³_1[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ³_2[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ⁴_1[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ⁴_2[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ⁵_1[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ⁵_2[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ²⁷_1[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ²⁷_2[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ³⁰_1[1:T]>=0)
@variable(model, ψᵐᵃˣˢᴳ³⁰_2[1:T]>=0)

@variable(model, γ_max[1:66,1:2,1:T]>=0)
@variable(model, γ_min[1:66,1:2,1:T]>=0)



#-------Define Primal Constraints

@constraint(model, Pˢᴳ²_1+Pˢᴳ²_2+Pˢᴳ³_1+Pˢᴳ³_2+Pˢᴳ⁴_1+Pˢᴳ⁴_2+Pˢᴳ⁵_1+Pˢᴳ⁵_2+Pˢᴳ²⁷_1+Pˢᴳ²⁷_2+Pˢᴳ³⁰_1+Pˢᴳ³⁰_2+
                   Pᴵᴮᴳ¹+Pᴵᴮᴳ²³+Pᴵᴮᴳ²⁶==Load_total)     # power balance , dual variable: λᴱₜ

@constraint(model, Pˢᴳ²_1.<=yˢᴳ²_1*Pˢᴳₘₐₓ[1])           # bounds for the output of SGs with UC , dual variables: μᵐⁱⁿₜ , μᵐᵃˣₜ
@constraint(model, yˢᴳ²_1*Pˢᴳₘᵢₙ[1].<=Pˢᴳ²_1)
@constraint(model, Pˢᴳ²_2.<=yˢᴳ²_2*Pˢᴳₘₐₓ[1])       
@constraint(model, yˢᴳ²_2*Pˢᴳₘᵢₙ[1].<=Pˢᴳ²_2)             
@constraint(model, Pˢᴳ³_1.<=yˢᴳ³_1*Pˢᴳₘₐₓ[2])       
@constraint(model, yˢᴳ³_1*Pˢᴳₘᵢₙ[2].<=Pˢᴳ³_1)       
@constraint(model, Pˢᴳ³_2.<=yˢᴳ³_2*Pˢᴳₘₐₓ[2])       
@constraint(model, yˢᴳ³_2*Pˢᴳₘᵢₙ[2].<=Pˢᴳ³_2)     
@constraint(model, Pˢᴳ⁴_1.<=yˢᴳ⁴_1*Pˢᴳₘₐₓ[3])       
@constraint(model, yˢᴳ⁴_1*Pˢᴳₘᵢₙ[3].<=Pˢᴳ⁴_1)
@constraint(model, Pˢᴳ⁴_2.<=yˢᴳ⁴_2*Pˢᴳₘₐₓ[3])       
@constraint(model, yˢᴳ⁴_2*Pˢᴳₘᵢₙ[3].<=Pˢᴳ⁴_2)     
@constraint(model, Pˢᴳ⁵_1.<=yˢᴳ⁵_1*Pˢᴳₘₐₓ[4])       
@constraint(model, yˢᴳ⁵_1*Pˢᴳₘᵢₙ[4].<=Pˢᴳ⁵_1)
@constraint(model, Pˢᴳ⁵_2.<=yˢᴳ⁵_2*Pˢᴳₘₐₓ[4])       
@constraint(model, yˢᴳ⁵_2*Pˢᴳₘᵢₙ[4].<=Pˢᴳ⁵_2)
@constraint(model, Pˢᴳ²⁷_1.<=yˢᴳ²⁷_1*Pˢᴳₘₐₓ[5])       
@constraint(model, yˢᴳ²⁷_1*Pˢᴳₘᵢₙ[5].<=Pˢᴳ²⁷_1)
@constraint(model, Pˢᴳ²⁷_2.<=yˢᴳ²⁷_2*Pˢᴳₘₐₓ[5])       
@constraint(model, yˢᴳ²⁷_2*Pˢᴳₘᵢₙ[5].<=Pˢᴳ²⁷_2)
@constraint(model, Pˢᴳ³⁰_1.<=yˢᴳ³⁰_1*Pˢᴳₘₐₓ[6])       
@constraint(model, yˢᴳ³⁰_1*Pˢᴳₘᵢₙ[6].<=Pˢᴳ³⁰_1)
@constraint(model, Pˢᴳ³⁰_2.<=yˢᴳ³⁰_2*Pˢᴳₘₐₓ[6])       
@constraint(model, yˢᴳ³⁰_2*Pˢᴳₘᵢₙ[6].<=Pˢᴳ³⁰_2)

@constraint(model, Pˢᴳ²_1[1]-P_g₀[1]<=Rₘₐₓ[1])        # bounds for the ramp of SGs , dual variables: πʳᵈₜ , πʳᵘₜ
@constraint(model, -Rₘₐₓ[1]<=Pˢᴳ²_1[1]-P_g₀[1])  
@constraint(model, Pˢᴳ²_2[1]-P_g₀[1]<=Rₘₐₓ[1])        
@constraint(model, -Rₘₐₓ[1]<=Pˢᴳ²_2[1]-P_g₀[1]) 
@constraint(model, Pˢᴳ³_1[1]-P_g₀[2]<=Rₘₐₓ[2])
@constraint(model, -Rₘₐₓ[2]<=Pˢᴳ³_1[1]-P_g₀[2])
@constraint(model, Pˢᴳ³_2[1]-P_g₀[2]<=Rₘₐₓ[2])        
@constraint(model, -Rₘₐₓ[2]<=Pˢᴳ³_2[1]-P_g₀[2]) 
@constraint(model, Pˢᴳ⁴_1[1]-P_g₀[3]<=Rₘₐₓ[3])
@constraint(model, -Rₘₐₓ[3]<=Pˢᴳ⁴_1[1]-P_g₀[3])
@constraint(model, Pˢᴳ⁴_2[1]-P_g₀[3]<=Rₘₐₓ[3])
@constraint(model, -Rₘₐₓ[3]<=Pˢᴳ⁴_2[1]-P_g₀[3])
@constraint(model, Pˢᴳ⁵_1[1]-P_g₀[4]<=Rₘₐₓ[4])
@constraint(model, -Rₘₐₓ[4]<=Pˢᴳ⁵_1[1]-P_g₀[4])
@constraint(model, Pˢᴳ⁵_2[1]-P_g₀[4]<=Rₘₐₓ[4])
@constraint(model, -Rₘₐₓ[4]<=Pˢᴳ⁵_2[1]-P_g₀[4])
@constraint(model, Pˢᴳ²⁷_1[1]-P_g₀[5]<=Rₘₐₓ[5])
@constraint(model, -Rₘₐₓ[5]<=Pˢᴳ²⁷_1[1]-P_g₀[5])
@constraint(model, Pˢᴳ²⁷_2[1]-P_g₀[5]<=Rₘₐₓ[5])
@constraint(model, -Rₘₐₓ[5]<=Pˢᴳ²⁷_2[1]-P_g₀[5])
@constraint(model, Pˢᴳ³⁰_1[1]-P_g₀[6]<=Rₘₐₓ[6])
@constraint(model, -Rₘₐₓ[6]<=Pˢᴳ³⁰_1[1]-P_g₀[6])
@constraint(model, Pˢᴳ³⁰_2[1]-P_g₀[6]<=Rₘₐₓ[6])
@constraint(model, -Rₘₐₓ[6]<=Pˢᴳ³⁰_2[1]-P_g₀[6])
for t in 2:T                                                   
    @constraint(model, Pˢᴳ²_1[t]-Pˢᴳ²_1[t-1]<=Rₘₐₓ[1])        
    @constraint(model, -Rₘₐₓ[1]<=Pˢᴳ²_1[t]-Pˢᴳ²_1[t-1])  
    @constraint(model, Pˢᴳ²_2[t]-Pˢᴳ²_2[t-1]<=Rₘₐₓ[1])        
    @constraint(model, -Rₘₐₓ[1]<=Pˢᴳ²_2[t]-Pˢᴳ²_2[t-1]) 

    @constraint(model, Pˢᴳ³_1[t]-Pˢᴳ³_1[t-1]<=Rₘₐₓ[2])
    @constraint(model, -Rₘₐₓ[2]<=Pˢᴳ³_1[t]-Pˢᴳ³_1[t-1])
    @constraint(model, Pˢᴳ³_2[t]-Pˢᴳ³_2[t-1]<=Rₘₐₓ[2])
    @constraint(model, -Rₘₐₓ[2]<=Pˢᴳ³_2[t]-Pˢᴳ³_2[t-1])  
    
    @constraint(model, Pˢᴳ⁴_1[t]-Pˢᴳ⁴_1[t-1]<=Rₘₐₓ[3])
    @constraint(model, -Rₘₐₓ[3]<=Pˢᴳ⁴_1[t]-Pˢᴳ⁴_1[t-1])
    @constraint(model, Pˢᴳ⁴_2[t]-Pˢᴳ⁴_2[t-1]<=Rₘₐₓ[3])
    @constraint(model, -Rₘₐₓ[3]<=Pˢᴳ⁴_2[t]-Pˢᴳ⁴_2[t-1])

    @constraint(model, Pˢᴳ⁵_1[t]-Pˢᴳ⁵_1[t-1]<=Rₘₐₓ[4])
    @constraint(model, -Rₘₐₓ[4]<=Pˢᴳ⁵_1[t]-Pˢᴳ⁵_1[t-1])
    @constraint(model, Pˢᴳ⁵_2[t]-Pˢᴳ⁵_2[t-1]<=Rₘₐₓ[4])
    @constraint(model, -Rₘₐₓ[4]<=Pˢᴳ⁵_2[t]-Pˢᴳ⁵_2[t-1])

    @constraint(model, Pˢᴳ²⁷_1[t]-Pˢᴳ²⁷_1[t-1]<=Rₘₐₓ[5])
    @constraint(model, -Rₘₐₓ[5]<=Pˢᴳ²⁷_1[t]-Pˢᴳ²⁷_1[t-1])
    @constraint(model, Pˢᴳ²⁷_2[t]-Pˢᴳ²⁷_2[t-1]<=Rₘₐₓ[5])
    @constraint(model, -Rₘₐₓ[5]<=Pˢᴳ²⁷_2[t]-Pˢᴳ²⁷_2[t-1])

    @constraint(model, Pˢᴳ³⁰_1[t]-Pˢᴳ³⁰_1[t-1]<=Rₘₐₓ[6])
    @constraint(model, -Rₘₐₓ[6]<=Pˢᴳ³⁰_1[t]-Pˢᴳ³⁰_1[t-1])
    @constraint(model, Pˢᴳ³⁰_2[t]-Pˢᴳ³⁰_2[t-1]<=Rₘₐₓ[6])
    @constraint(model, -Rₘₐₓ[6]<=Pˢᴳ³⁰_2[t]-Pˢᴳ³⁰_2[t-1])
end

@constraint(model, Cᵁ²_1[1]>=(yˢᴳ²_1[1]-yˢᴳ₀[1])*Kˢᵗ[1])        # startup costs and shutdown costs for SGs , dual variables: σˢᵗₜ , σˢʰₜ
@constraint(model, Cᴰ²_1[1]>=(yˢᴳ₀[1]-yˢᴳ²_1[1])*Kˢʰ[1])  
@constraint(model, Cᵁ²_2[1]>=(yˢᴳ²_2[1]-yˢᴳ₀[1])*Kˢᵗ[1])        
@constraint(model, Cᴰ²_2[1]>=(yˢᴳ₀[1]-yˢᴳ²_2[1])*Kˢʰ[1]) 
@constraint(model, Cᵁ³_1[1]>=(yˢᴳ³_1[1]-yˢᴳ₀[2])*Kˢᵗ[2])
@constraint(model, Cᴰ³_1[1]>=(yˢᴳ₀[2]-yˢᴳ³_1[1])*Kˢʰ[2])
@constraint(model, Cᵁ³_2[1]>=(yˢᴳ³_2[1]-yˢᴳ₀[2])*Kˢᵗ[2])
@constraint(model, Cᴰ³_2[1]>=(yˢᴳ₀[2]-yˢᴳ³_2[1])*Kˢʰ[2])
@constraint(model, Cᵁ⁴_1[1]>=(yˢᴳ⁴_1[1]-yˢᴳ₀[3])*Kˢᵗ[3])
@constraint(model, Cᴰ⁴_1[1]>=(yˢᴳ₀[3]-yˢᴳ⁴_1[1])*Kˢʰ[3])
@constraint(model, Cᵁ⁴_2[1]>=(yˢᴳ⁴_2[1]-yˢᴳ₀[3])*Kˢᵗ[3])
@constraint(model, Cᴰ⁴_2[1]>=(yˢᴳ₀[3]-yˢᴳ⁴_2[1])*Kˢʰ[3])
@constraint(model, Cᵁ⁵_1[1]>=(yˢᴳ⁵_1[1]-yˢᴳ₀[4])*Kˢᵗ[4])
@constraint(model, Cᴰ⁵_1[1]>=(yˢᴳ₀[4]-yˢᴳ⁵_1[1])*Kˢʰ[4])
@constraint(model, Cᵁ⁵_2[1]>=(yˢᴳ⁵_2[1]-yˢᴳ₀[4])*Kˢᵗ[4])
@constraint(model, Cᴰ⁵_2[1]>=(yˢᴳ₀[4]-yˢᴳ⁵_2[1])*Kˢʰ[4])
@constraint(model, Cᵁ²⁷_1[1]>=(yˢᴳ²⁷_1[1]-yˢᴳ₀[5])*Kˢᵗ[5])
@constraint(model, Cᴰ²⁷_1[1]>=(yˢᴳ₀[5]-yˢᴳ²⁷_1[1])*Kˢʰ[5])
@constraint(model, Cᵁ²⁷_2[1]>=(yˢᴳ²⁷_2[1]-yˢᴳ₀[5])*Kˢᵗ[5])
@constraint(model, Cᴰ²⁷_2[1]>=(yˢᴳ₀[5]-yˢᴳ²⁷_2[1])*Kˢʰ[5])
@constraint(model, Cᵁ³⁰_1[1]>=(yˢᴳ³⁰_1[1]-yˢᴳ₀[6])*Kˢᵗ[6])
@constraint(model, Cᴰ³⁰_1[1]>=(yˢᴳ₀[6]-yˢᴳ³⁰_1[1])*Kˢʰ[6])
@constraint(model, Cᵁ³⁰_2[1]>=(yˢᴳ³⁰_2[1]-yˢᴳ₀[6])*Kˢᵗ[6])
@constraint(model, Cᴰ³⁰_2[1]>=(yˢᴳ₀[6]-yˢᴳ³⁰_2[1])*Kˢʰ[6]) 
for t in 2:T
    @constraint(model, Cᵁ²_1[t]>=(yˢᴳ²_1[t]-yˢᴳ²_1[t-1])*Kˢᵗ[1])        
    @constraint(model, Cᴰ²_1[t]>=(yˢᴳ²_1[t-1]-yˢᴳ²_1[t])*Kˢʰ[1])  
    @constraint(model, Cᵁ²_2[t]>=(yˢᴳ²_2[t]-yˢᴳ²_2[t-1])*Kˢᵗ[1])        
    @constraint(model, Cᴰ²_2[t]>=(yˢᴳ²_2[t-1]-yˢᴳ²_2[t])*Kˢʰ[1]) 

    @constraint(model, Cᵁ³_1[t]>=(yˢᴳ³_1[t]-yˢᴳ³_1[t-1])*Kˢᵗ[2]) 
    @constraint(model, Cᴰ³_1[t]>=(yˢᴳ³_1[t-1]-yˢᴳ³_1[t])*Kˢʰ[2])
    @constraint(model, Cᵁ³_2[t]>=(yˢᴳ³_2[t]-yˢᴳ³_2[t-1])*Kˢᵗ[2])
    @constraint(model, Cᴰ³_2[t]>=(yˢᴳ³_2[t-1]-yˢᴳ³_2[t])*Kˢʰ[2])

    @constraint(model, Cᵁ⁴_1[t]>=(yˢᴳ⁴_1[t]-yˢᴳ⁴_1[t-1])*Kˢᵗ[3])
    @constraint(model, Cᴰ⁴_1[t]>=(yˢᴳ⁴_1[t-1]-yˢᴳ⁴_1[t])*Kˢʰ[3])
    @constraint(model, Cᵁ⁴_2[t]>=(yˢᴳ⁴_2[t]-yˢᴳ⁴_2[t-1])*Kˢᵗ[3])
    @constraint(model, Cᴰ⁴_2[t]>=(yˢᴳ⁴_2[t-1]-yˢᴳ⁴_2[t])*Kˢʰ[3])

    @constraint(model, Cᵁ⁵_1[t]>=(yˢᴳ⁵_1[t]-yˢᴳ⁵_1[t-1])*Kˢᵗ[4])
    @constraint(model, Cᴰ⁵_1[t]>=(yˢᴳ⁵_1[t-1]-yˢᴳ⁵_1[t])*Kˢʰ[4])
    @constraint(model, Cᵁ⁵_2[t]>=(yˢᴳ⁵_2[t]-yˢᴳ⁵_2[t-1])*Kˢᵗ[4])
    @constraint(model, Cᴰ⁵_2[t]>=(yˢᴳ⁵_2[t-1]-yˢᴳ⁵_2[t])*Kˢʰ[4])

    @constraint(model, Cᵁ²⁷_1[t]>=(yˢᴳ²⁷_1[t]-yˢᴳ²⁷_1[t-1])*Kˢᵗ[5])
    @constraint(model, Cᴰ²⁷_1[t]>=(yˢᴳ²⁷_1[t-1]-yˢᴳ²⁷_1[t])*Kˢʰ[5])
    @constraint(model, Cᵁ²⁷_2[t]>=(yˢᴳ²⁷_2[t]-yˢᴳ²⁷_2[t-1])*Kˢᵗ[5])
    @constraint(model, Cᴰ²⁷_2[t]>=(yˢᴳ²⁷_2[t-1]-yˢᴳ²⁷_2[t])*Kˢʰ[5])

    @constraint(model, Cᵁ³⁰_1[t]>=(yˢᴳ³⁰_1[t]-yˢᴳ³⁰_1[t-1])*Kˢᵗ[6])
    @constraint(model, Cᴰ³⁰_1[t]>=(yˢᴳ³⁰_1[t-1]-yˢᴳ³⁰_1[t])*Kˢʰ[6])
    @constraint(model, Cᵁ³⁰_2[t]>=(yˢᴳ³⁰_2[t]-yˢᴳ³⁰_2[t-1])*Kˢᵗ[6])
    @constraint(model, Cᴰ³⁰_2[t]>=(yˢᴳ³⁰_2[t-1]-yˢᴳ³⁰_2[t])*Kˢʰ[6])
end
    
for t in 1:T
    @constraint(model, Pᴵᴮᴳ¹[t] <= IBG₁[t]*α₁[t])        # wind power limit  , dual variable: ζᵐᵃˣₜ
    @constraint(model, Pᴵᴮᴳ²³[t]<= IBG₂₃[t]*α₂₃[t])       
    @constraint(model, Pᴵᴮᴳ²⁶[t]<= IBG₂₆[t]*α₂₆[t])       

    @constraint(model, α₁[t]<=1)                         # IBR online capacity limit  , dual variable:      φᵐᵃˣₜ
    @constraint(model, α₂₃[t]<=1)                   
    @constraint(model, α₂₆[t]<=1)                   
end

#-------Define McCormick equations of products of two binaries
for t in 1:T
    # yˢᴳ²_1
    @constraint(model, η_m[1,t]<=yˢᴳ²_1[t])
    @constraint(model, η_m[1,t]<=yˢᴳ²_2[t])
    @constraint(model, η_m[1,t]>=yˢᴳ²_1[t]+yˢᴳ²_2[t]-1)
    @constraint(model, η_m[2,t]<=yˢᴳ²_1[t])
    @constraint(model, η_m[2,t]<=yˢᴳ³_1[t])
    @constraint(model, η_m[2,t]>=yˢᴳ²_1[t]+yˢᴳ³_1[t]-1)
    @constraint(model, η_m[3,t]<=yˢᴳ²_1[t])
    @constraint(model, η_m[3,t]<=yˢᴳ³_2[t])
    @constraint(model, η_m[3,t]>=yˢᴳ²_1[t]+yˢᴳ³_2[t]-1)
    @constraint(model, η_m[4,t]<=yˢᴳ²_1[t])
    @constraint(model, η_m[4,t]<=yˢᴳ⁴_1[t])
    @constraint(model, η_m[4,t]>=yˢᴳ²_1[t]+yˢᴳ⁴_1[t]-1)
    @constraint(model, η_m[5,t]<=yˢᴳ²_1[t])
    @constraint(model, η_m[5,t]<=yˢᴳ⁴_2[t])
    @constraint(model, η_m[5,t]>=yˢᴳ²_1[t]+yˢᴳ⁴_2[t]-1)
    @constraint(model, η_m[6,t]<=yˢᴳ²_1[t])
    @constraint(model, η_m[6,t]<=yˢᴳ⁵_1[t])
    @constraint(model, η_m[6,t]>=yˢᴳ²_1[t]+yˢᴳ⁵_1[t]-1)
    @constraint(model, η_m[7,t]<=yˢᴳ²_1[t])
    @constraint(model, η_m[7,t]<=yˢᴳ⁵_2[t])
    @constraint(model, η_m[7,t]>=yˢᴳ²_1[t]+yˢᴳ⁵_2[t]-1)
    @constraint(model, η_m[8,t]<=yˢᴳ²_1[t])
    @constraint(model, η_m[8,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, η_m[8,t]>=yˢᴳ²_1[t]+yˢᴳ²⁷_1[t]-1)
    @constraint(model, η_m[9,t]<=yˢᴳ²_1[t])
    @constraint(model, η_m[9,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, η_m[9,t]>=yˢᴳ²_1[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, η_m[10,t]<=yˢᴳ²_1[t])
    @constraint(model, η_m[10,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, η_m[10,t]>=yˢᴳ²_1[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, η_m[11,t]<=yˢᴳ²_1[t])
    @constraint(model, η_m[11,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, η_m[11,t]>=yˢᴳ²_1[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ²_2
    @constraint(model, η_m[12,t]<=yˢᴳ²_2[t])
    @constraint(model, η_m[12,t]<=yˢᴳ³_1[t])
    @constraint(model, η_m[12,t]>=yˢᴳ²_2[t]+yˢᴳ³_1[t]-1)
    @constraint(model, η_m[13,t]<=yˢᴳ²_2[t])
    @constraint(model, η_m[13,t]<=yˢᴳ³_2[t])
    @constraint(model, η_m[13,t]>=yˢᴳ²_2[t]+yˢᴳ³_2[t]-1)
    @constraint(model, η_m[14,t]<=yˢᴳ²_2[t])
    @constraint(model, η_m[14,t]<=yˢᴳ⁴_1[t])
    @constraint(model, η_m[14,t]>=yˢᴳ²_2[t]+yˢᴳ⁴_1[t]-1)
    @constraint(model, η_m[15,t]<=yˢᴳ²_2[t])
    @constraint(model, η_m[15,t]<=yˢᴳ⁴_2[t])
    @constraint(model, η_m[15,t]>=yˢᴳ²_2[t]+yˢᴳ⁴_2[t]-1)
    @constraint(model, η_m[16,t]<=yˢᴳ²_2[t])
    @constraint(model, η_m[16,t]<=yˢᴳ⁵_1[t])
    @constraint(model, η_m[16,t]>=yˢᴳ²_2[t]+yˢᴳ⁵_1[t]-1)
    @constraint(model, η_m[17,t]<=yˢᴳ²_2[t])
    @constraint(model, η_m[17,t]<=yˢᴳ⁵_2[t])
    @constraint(model, η_m[17,t]>=yˢᴳ²_2[t]+yˢᴳ⁵_2[t]-1)
    @constraint(model, η_m[18,t]<=yˢᴳ²_2[t])
    @constraint(model, η_m[18,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, η_m[18,t]>=yˢᴳ²_2[t]+yˢᴳ²⁷_1[t]-1)
    @constraint(model, η_m[19,t]<=yˢᴳ²_2[t])
    @constraint(model, η_m[19,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, η_m[19,t]>=yˢᴳ²_2[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, η_m[20,t]<=yˢᴳ²_2[t])
    @constraint(model, η_m[20,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, η_m[20,t]>=yˢᴳ²_2[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, η_m[21,t]<=yˢᴳ²_2[t])
    @constraint(model, η_m[21,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, η_m[21,t]>=yˢᴳ²_2[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ³_1
    @constraint(model, η_m[22,t]<=yˢᴳ³_1[t])
    @constraint(model, η_m[22,t]<=yˢᴳ³_2[t])
    @constraint(model, η_m[22,t]>=yˢᴳ³_1[t]+yˢᴳ³_2[t]-1)
    @constraint(model, η_m[23,t]<=yˢᴳ³_1[t])
    @constraint(model, η_m[23,t]<=yˢᴳ⁴_1[t])
    @constraint(model, η_m[23,t]>=yˢᴳ³_1[t]+yˢᴳ⁴_1[t]-1)
    @constraint(model, η_m[24,t]<=yˢᴳ³_1[t])
    @constraint(model, η_m[24,t]<=yˢᴳ⁴_2[t])
    @constraint(model, η_m[24,t]>=yˢᴳ³_1[t]+yˢᴳ⁴_2[t]-1)
    @constraint(model, η_m[25,t]<=yˢᴳ³_1[t])
    @constraint(model, η_m[25,t]<=yˢᴳ⁵_1[t])
    @constraint(model, η_m[25,t]>=yˢᴳ³_1[t]+yˢᴳ⁵_1[t]-1)
    @constraint(model, η_m[26,t]<=yˢᴳ³_1[t])
    @constraint(model, η_m[26,t]<=yˢᴳ⁵_2[t])
    @constraint(model, η_m[26,t]>=yˢᴳ³_1[t]+yˢᴳ⁵_2[t]-1)
    @constraint(model, η_m[27,t]<=yˢᴳ³_1[t])
    @constraint(model, η_m[27,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, η_m[27,t]>=yˢᴳ³_1[t]+yˢᴳ²⁷_1[t]-1)
    @constraint(model, η_m[28,t]<=yˢᴳ³_1[t])
    @constraint(model, η_m[28,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, η_m[28,t]>=yˢᴳ³_1[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, η_m[29,t]<=yˢᴳ³_1[t])
    @constraint(model, η_m[29,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, η_m[29,t]>=yˢᴳ³_1[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, η_m[30,t]<=yˢᴳ³_1[t])
    @constraint(model, η_m[30,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, η_m[30,t]>=yˢᴳ³_1[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ³_2
    @constraint(model, η_m[31,t]<=yˢᴳ³_2[t])
    @constraint(model, η_m[31,t]<=yˢᴳ⁴_1[t])
    @constraint(model, η_m[31,t]>=yˢᴳ³_2[t]+yˢᴳ⁴_1[t]-1)
    @constraint(model, η_m[32,t]<=yˢᴳ³_2[t])
    @constraint(model, η_m[32,t]<=yˢᴳ⁴_2[t])
    @constraint(model, η_m[32,t]>=yˢᴳ³_2[t]+yˢᴳ⁴_2[t]-1)
    @constraint(model, η_m[33,t]<=yˢᴳ³_2[t])
    @constraint(model, η_m[33,t]<=yˢᴳ⁵_1[t])
    @constraint(model, η_m[33,t]>=yˢᴳ³_2[t]+yˢᴳ⁵_1[t]-1)
    @constraint(model, η_m[34,t]<=yˢᴳ³_2[t])
    @constraint(model, η_m[34,t]<=yˢᴳ⁵_2[t])
    @constraint(model, η_m[34,t]>=yˢᴳ³_2[t]+yˢᴳ⁵_2[t]-1)
    @constraint(model, η_m[35,t]<=yˢᴳ³_2[t])
    @constraint(model, η_m[35,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, η_m[35,t]>=yˢᴳ³_2[t]+yˢᴳ²⁷_1[t]-1)
    @constraint(model, η_m[36,t]<=yˢᴳ³_2[t])
    @constraint(model, η_m[36,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, η_m[36,t]>=yˢᴳ³_2[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, η_m[37,t]<=yˢᴳ³_2[t])
    @constraint(model, η_m[37,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, η_m[37,t]>=yˢᴳ³_2[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, η_m[38,t]<=yˢᴳ³_2[t])
    @constraint(model, η_m[38,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, η_m[38,t]>=yˢᴳ³_2[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ⁴_1
    @constraint(model, η_m[39,t]<=yˢᴳ⁴_1[t])
    @constraint(model, η_m[39,t]<=yˢᴳ⁴_2[t])
    @constraint(model, η_m[39,t]>=yˢᴳ⁴_1[t]+yˢᴳ⁴_2[t]-1)
    @constraint(model, η_m[40,t]<=yˢᴳ⁴_1[t])
    @constraint(model, η_m[40,t]<=yˢᴳ⁵_1[t])
    @constraint(model, η_m[40,t]>=yˢᴳ⁴_1[t]+yˢᴳ⁵_1[t]-1)
    @constraint(model, η_m[41,t]<=yˢᴳ⁴_1[t])
    @constraint(model, η_m[41,t]<=yˢᴳ⁵_2[t])
    @constraint(model, η_m[41,t]>=yˢᴳ⁴_1[t]+yˢᴳ⁵_2[t]-1)
    @constraint(model, η_m[42,t]<=yˢᴳ⁴_1[t])
    @constraint(model, η_m[42,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, η_m[42,t]>=yˢᴳ⁴_1[t]+yˢᴳ²⁷_1[t]-1)
    @constraint(model, η_m[43,t]<=yˢᴳ⁴_1[t])
    @constraint(model, η_m[43,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, η_m[43,t]>=yˢᴳ⁴_1[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, η_m[44,t]<=yˢᴳ⁴_1[t])
    @constraint(model, η_m[44,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, η_m[44,t]>=yˢᴳ⁴_1[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, η_m[45,t]<=yˢᴳ⁴_1[t])
    @constraint(model, η_m[45,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, η_m[45,t]>=yˢᴳ⁴_1[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ⁴_2
    @constraint(model, η_m[46,t]<=yˢᴳ⁴_2[t])
    @constraint(model, η_m[46,t]<=yˢᴳ⁵_1[t])
    @constraint(model, η_m[46,t]>=yˢᴳ⁴_2[t]+yˢᴳ⁵_1[t]-1)
    @constraint(model, η_m[47,t]<=yˢᴳ⁴_2[t])
    @constraint(model, η_m[47,t]<=yˢᴳ⁵_2[t])
    @constraint(model, η_m[47,t]>=yˢᴳ⁴_2[t]+yˢᴳ⁵_2[t]-1)
    @constraint(model, η_m[48,t]<=yˢᴳ⁴_2[t])
    @constraint(model, η_m[48,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, η_m[48,t]>=yˢᴳ⁴_2[t]+yˢᴳ²⁷_1[t]-1)
    @constraint(model, η_m[49,t]<=yˢᴳ⁴_2[t])
    @constraint(model, η_m[49,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, η_m[49,t]>=yˢᴳ⁴_2[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, η_m[50,t]<=yˢᴳ⁴_2[t])
    @constraint(model, η_m[50,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, η_m[50,t]>=yˢᴳ⁴_2[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, η_m[51,t]<=yˢᴳ⁴_2[t])
    @constraint(model, η_m[51,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, η_m[51,t]>=yˢᴳ⁴_2[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ⁵_1
    @constraint(model, η_m[52,t]<=yˢᴳ⁵_1[t])
    @constraint(model, η_m[52,t]<=yˢᴳ⁵_2[t])
    @constraint(model, η_m[52,t]>=yˢᴳ⁵_1[t]+yˢᴳ⁵_2[t]-1)
    @constraint(model, η_m[53,t]<=yˢᴳ⁵_1[t])
    @constraint(model, η_m[53,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, η_m[53,t]>=yˢᴳ⁵_1[t]+yˢᴳ²⁷_1[t]-1)
    @constraint(model, η_m[54,t]<=yˢᴳ⁵_1[t])
    @constraint(model, η_m[54,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, η_m[54,t]>=yˢᴳ⁵_1[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, η_m[55,t]<=yˢᴳ⁵_1[t])
    @constraint(model, η_m[55,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, η_m[55,t]>=yˢᴳ⁵_1[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, η_m[56,t]<=yˢᴳ⁵_1[t])
    @constraint(model, η_m[56,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, η_m[56,t]>=yˢᴳ⁵_1[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ⁵_2
    @constraint(model, η_m[57,t]<=yˢᴳ⁵_2[t])
    @constraint(model, η_m[57,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, η_m[57,t]>=yˢᴳ⁵_2[t]+yˢᴳ²⁷_1[t]-1)
    @constraint(model, η_m[58,t]<=yˢᴳ⁵_2[t])
    @constraint(model, η_m[58,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, η_m[58,t]>=yˢᴳ⁵_2[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, η_m[59,t]<=yˢᴳ⁵_2[t])
    @constraint(model, η_m[59,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, η_m[59,t]>=yˢᴳ⁵_2[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, η_m[60,t]<=yˢᴳ⁵_2[t])
    @constraint(model, η_m[60,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, η_m[60,t]>=yˢᴳ⁵_2[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ²⁷_1
    @constraint(model, η_m[61,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, η_m[61,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, η_m[61,t]>=yˢᴳ²⁷_1[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, η_m[62,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, η_m[62,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, η_m[62,t]>=yˢᴳ²⁷_1[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, η_m[63,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, η_m[63,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, η_m[63,t]>=yˢᴳ²⁷_1[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ²⁷_2
    @constraint(model, η_m[64,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, η_m[64,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, η_m[64,t]>=yˢᴳ²⁷_2[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, η_m[65,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, η_m[65,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, η_m[65,t]>=yˢᴳ²⁷_2[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ³⁰_1
    @constraint(model, η_m[66,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, η_m[66,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, η_m[66,t]>=yˢᴳ³⁰_1[t]+yˢᴳ³⁰_2[t]-1)
end

k=11   
for t in 1:T                                              # bounds for the SCL of buses  I_₃₀
        @constraint(model, I_₁₁[t]==                     
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+
        K_c[1,k]*α₁[t]+ K_c[2,k]*α₂₃[t]+ K_c[3,k]*α₂₆[t]+
        
        K_m[1,k]*η_m[1,t] +K_m[2,k]*η_m[2,t] +K_m[3,k]*η_m[3,t] +K_m[4,k]*η_m[4,t]+
        K_m[5,k]*η_m[5,t] +K_m[6,k]*η_m[6,t] +K_m[7,k]*η_m[7,t] +K_m[8,k]*η_m[8,t]+
        K_m[9,k]*η_m[9,t] +K_m[10,k]*η_m[10,t] +K_m[11,k]*η_m[11,t]+

        K_m[12,k]*η_m[12,t] +K_m[13,k]*η_m[13,t] +K_m[14,k]*η_m[14,t]+
        K_m[15,k]*η_m[15,t] +K_m[16,k]*η_m[16,t] +K_m[17,k]*η_m[17,t] +K_m[18,k]*η_m[18,t]+
        K_m[19,k]*η_m[19,t] +K_m[20,k]*η_m[20,t] +K_m[21,k]*η_m[21,t]+

        K_m[22,k]*η_m[22,t] +K_m[23,k]*η_m[23,t]+
        K_m[24,k]*η_m[24,t] +K_m[25,k]*η_m[25,t] +K_m[26,k]*η_m[26,t] +K_m[27,k]*η_m[27,t]+
        K_m[28,k]*η_m[28,t] +K_m[29,k]*η_m[29,t] +K_m[30,k]*η_m[30,t]+

        K_m[31,k]*η_m[31,t]+
        K_m[32,k]*η_m[32,t] +K_m[33,k]*η_m[33,t] +K_m[34,k]*η_m[34,t] +K_m[35,k]*η_m[35,t]+
        K_m[36,k]*η_m[36,t] +K_m[37,k]*η_m[37,t] +K_m[38,k]*η_m[38,t]+

        K_m[39,k]*η_m[39,t] +K_m[40,k]*η_m[40,t] +K_m[41,k]*η_m[41,t] +K_m[42,k]*η_m[42,t]+
        K_m[43,k]*η_m[43,t] +K_m[44,k]*η_m[44,t] +K_m[45,k]*η_m[45,t]+

        K_m[46,k]*η_m[46,t] +K_m[47,k]*η_m[47,t] +K_m[48,k]*η_m[48,t]+
        K_m[49,k]*η_m[49,t] +K_m[50,k]*η_m[50,t] +K_m[51,k]*η_m[51,t]+

        K_m[52,k]*η_m[52,t] +K_m[53,k]*η_m[53,t]+
        K_m[54,k]*η_m[54,t] +K_m[55,k]*η_m[55,t] +K_m[56,k]*η_m[56,t]+

        K_m[57,k]*η_m[57,t]+
        K_m[58,k]*η_m[58,t] +K_m[59,k]*η_m[59,t] +K_m[60,k]*η_m[60,t]+

        K_m[61,k]*η_m[61,t] +K_m[62,k]*η_m[62,t] +K_m[63,k]*η_m[63,t]+

        K_m[64,k]*η_m[64,t] +K_m[65,k]*η_m[65,t]+

        K_m[66,k]*η_m[66,t])

        @constraint(model, I_₁₁[t]>=Iₗᵢₘ)                  # AS requirement for SCL on bus F=30  , dual variable: λ_F  
end

k=26    # for bus 26   
for t in 1:T                                             # bounds for the SCL of buses  I_₂₆   
        @constraint(model, I_₂₆[t]==                     # SCL on bus F=26
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+
        K_c[1,k]*α₁[t]+ K_c[2,k]*α₂₃[t]+ K_c[3,k]*α₂₆[t]+

        K_m[1,k]*η_m[1,t] +K_m[2,k]*η_m[2,t] +K_m[3,k]*η_m[3,t] +K_m[4,k]*η_m[4,t]+
        K_m[5,k]*η_m[5,t] +K_m[6,k]*η_m[6,t] +K_m[7,k]*η_m[7,t] +K_m[8,k]*η_m[8,t]+
        K_m[9,k]*η_m[9,t] +K_m[10,k]*η_m[10,t] +K_m[11,k]*η_m[11,t]+

        K_m[12,k]*η_m[12,t] +K_m[13,k]*η_m[13,t] +K_m[14,k]*η_m[14,t]+
        K_m[15,k]*η_m[15,t] +K_m[16,k]*η_m[16,t] +K_m[17,k]*η_m[17,t] +K_m[18,k]*η_m[18,t]+
        K_m[19,k]*η_m[19,t] +K_m[20,k]*η_m[20,t] +K_m[21,k]*η_m[21,t]+

        K_m[22,k]*η_m[22,t] +K_m[23,k]*η_m[23,t]+
        K_m[24,k]*η_m[24,t] +K_m[25,k]*η_m[25,t] +K_m[26,k]*η_m[26,t] +K_m[27,k]*η_m[27,t]+
        K_m[28,k]*η_m[28,t] +K_m[29,k]*η_m[29,t] +K_m[30,k]*η_m[30,t]+

        K_m[31,k]*η_m[31,t]+
        K_m[32,k]*η_m[32,t] +K_m[33,k]*η_m[33,t] +K_m[34,k]*η_m[34,t] +K_m[35,k]*η_m[35,t]+
        K_m[36,k]*η_m[36,t] +K_m[37,k]*η_m[37,t] +K_m[38,k]*η_m[38,t]+

        K_m[39,k]*η_m[39,t] +K_m[40,k]*η_m[40,t] +K_m[41,k]*η_m[41,t] +K_m[42,k]*η_m[42,t]+
        K_m[43,k]*η_m[43,t] +K_m[44,k]*η_m[44,t] +K_m[45,k]*η_m[45,t]+

        K_m[46,k]*η_m[46,t] +K_m[47,k]*η_m[47,t] +K_m[48,k]*η_m[48,t]+
        K_m[49,k]*η_m[49,t] +K_m[50,k]*η_m[50,t] +K_m[51,k]*η_m[51,t]+

        K_m[52,k]*η_m[52,t] +K_m[53,k]*η_m[53,t]+
        K_m[54,k]*η_m[54,t] +K_m[55,k]*η_m[55,t] +K_m[56,k]*η_m[56,t]+

        K_m[57,k]*η_m[57,t]+
        K_m[58,k]*η_m[58,t] +K_m[59,k]*η_m[59,t] + K_m[60,k]*η_m[60,t]+

        K_m[61,k]*η_m[61,t] +K_m[62,k]*η_m[62,t] +K_m[63,k]*η_m[63,t]+

        K_m[64,k]*η_m[64,t] +K_m[65,k]*η_m[65,t]+

        K_m[66,k]*η_m[66,t])

        @constraint(model, I_₂₆[t]>=Iₗᵢₘ)                  # AS requirement for SCL on bus F=26  , dual variable: λ_F  
end

k=29    
for t in 1:T                                              # bounds for the SCL of buses   I_₂₉  
        @constraint(model, I_₂₉[t]==                      # SCL on bus F=29  
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+
        K_c[1,k]*α₁[t]+ K_c[2,k]*α₂₃[t]+ K_c[3,k]*α₂₆[t]+
        
        K_m[1,k]*η_m[1,t] +K_m[2,k]*η_m[2,t] +K_m[3,k]*η_m[3,t] +K_m[4,k]*η_m[4,t]+
        K_m[5,k]*η_m[5,t] +K_m[6,k]*η_m[6,t] +K_m[7,k]*η_m[7,t] +K_m[8,k]*η_m[8,t]+
        K_m[9,k]*η_m[9,t] +K_m[10,k]*η_m[10,t] +K_m[11,k]*η_m[11,t]+

        K_m[12,k]*η_m[12,t] +K_m[13,k]*η_m[13,t] +K_m[14,k]*η_m[14,t]+
        K_m[15,k]*η_m[15,t] +K_m[16,k]*η_m[16,t] +K_m[17,k]*η_m[17,t] +K_m[18,k]*η_m[18,t]+
        K_m[19,k]*η_m[19,t] +K_m[20,k]*η_m[20,t] +K_m[21,k]*η_m[21,t]+

        K_m[22,k]*η_m[22,t] +K_m[23,k]*η_m[23,t]+
        K_m[24,k]*η_m[24,t] +K_m[25,k]*η_m[25,t] +K_m[26,k]*η_m[26,t] +K_m[27,k]*η_m[27,t]+
        K_m[28,k]*η_m[28,t] +K_m[29,k]*η_m[29,t] +K_m[30,k]*η_m[30,t]+

        K_m[31,k]*η_m[31,t]+
        K_m[32,k]*η_m[32,t] +K_m[33,k]*η_m[33,t] +K_m[34,k]*η_m[34,t] +K_m[35,k]*η_m[35,t]+
        K_m[36,k]*η_m[36,t] +K_m[37,k]*η_m[37,t] +K_m[38,k]*η_m[38,t]+

        K_m[39,k]*η_m[39,t] +K_m[40,k]*η_m[40,t] +K_m[41,k]*η_m[41,t] +K_m[42,k]*η_m[42,t]+
        K_m[43,k]*η_m[43,t] +K_m[44,k]*η_m[44,t] +K_m[45,k]*η_m[45,t]+

        K_m[46,k]*η_m[46,t] +K_m[47,k]*η_m[47,t] +K_m[48,k]*η_m[48,t]+
        K_m[49,k]*η_m[49,t] +K_m[50,k]*η_m[50,t] +K_m[51,k]*η_m[51,t]+

        K_m[52,k]*η_m[52,t] +K_m[53,k]*η_m[53,t]+
        K_m[54,k]*η_m[54,t] +K_m[55,k]*η_m[55,t] +K_m[56,k]*η_m[56,t]+

        K_m[57,k]*η_m[57,t]+
        K_m[58,k]*η_m[58,t] +K_m[59,k]*η_m[59,t] + K_m[60,k]*η_m[60,t]+

        K_m[61,k]*η_m[61,t] +K_m[62,k]*η_m[62,t] +K_m[63,k]*η_m[63,t]+

        K_m[64,k]*η_m[64,t] +K_m[65,k]*η_m[65,t]+

        K_m[66,k]*η_m[66,t])

        @constraint(model, I_₂₉[t]>=Iₗᵢₘ)                  # AS requirement for SCL on bus F=29  , dual variable: λ_F  
end

k=30    
for t in 1:T                                              # bounds for the SCL of buses  I_₃₀
        @constraint(model, I_₃₀[t]==                     # SCL on bus F=30  
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+
        K_c[1,k]*α₁[t]+ K_c[2,k]*α₂₃[t]+ K_c[3,k]*α₂₆[t]+
        
        K_m[1,k]*η_m[1,t] +K_m[2,k]*η_m[2,t] +K_m[3,k]*η_m[3,t] +K_m[4,k]*η_m[4,t]+
        K_m[5,k]*η_m[5,t] +K_m[6,k]*η_m[6,t] +K_m[7,k]*η_m[7,t] +K_m[8,k]*η_m[8,t]+
        K_m[9,k]*η_m[9,t] +K_m[10,k]*η_m[10,t] +K_m[11,k]*η_m[11,t]+

        K_m[12,k]*η_m[12,t] +K_m[13,k]*η_m[13,t] +K_m[14,k]*η_m[14,t]+
        K_m[15,k]*η_m[15,t] +K_m[16,k]*η_m[16,t] +K_m[17,k]*η_m[17,t] +K_m[18,k]*η_m[18,t]+
        K_m[19,k]*η_m[19,t] +K_m[20,k]*η_m[20,t] +K_m[21,k]*η_m[21,t]+

        K_m[22,k]*η_m[22,t] +K_m[23,k]*η_m[23,t]+
        K_m[24,k]*η_m[24,t] +K_m[25,k]*η_m[25,t] +K_m[26,k]*η_m[26,t] +K_m[27,k]*η_m[27,t]+
        K_m[28,k]*η_m[28,t] +K_m[29,k]*η_m[29,t] +K_m[30,k]*η_m[30,t]+

        K_m[31,k]*η_m[31,t]+
        K_m[32,k]*η_m[32,t] +K_m[33,k]*η_m[33,t] +K_m[34,k]*η_m[34,t] +K_m[35,k]*η_m[35,t]+
        K_m[36,k]*η_m[36,t] +K_m[37,k]*η_m[37,t] +K_m[38,k]*η_m[38,t]+

        K_m[39,k]*η_m[39,t] +K_m[40,k]*η_m[40,t] +K_m[41,k]*η_m[41,t] +K_m[42,k]*η_m[42,t]+
        K_m[43,k]*η_m[43,t] +K_m[44,k]*η_m[44,t] +K_m[45,k]*η_m[45,t]+

        K_m[46,k]*η_m[46,t] +K_m[47,k]*η_m[47,t] +K_m[48,k]*η_m[48,t]+
        K_m[49,k]*η_m[49,t] +K_m[50,k]*η_m[50,t] +K_m[51,k]*η_m[51,t]+

        K_m[52,k]*η_m[52,t] +K_m[53,k]*η_m[53,t]+
        K_m[54,k]*η_m[54,t] +K_m[55,k]*η_m[55,t] +K_m[56,k]*η_m[56,t]+

        K_m[57,k]*η_m[57,t]+
        K_m[58,k]*η_m[58,t] +K_m[59,k]*η_m[59,t] +K_m[60,k]*η_m[60,t]+

        K_m[61,k]*η_m[61,t] +K_m[62,k]*η_m[62,t] +K_m[63,k]*η_m[63,t]+

        K_m[64,k]*η_m[64,t] +K_m[65,k]*η_m[65,t]+

        K_m[66,k]*η_m[66,t])

        @constraint(model, I_₃₀[t]>=Iₗᵢₘ)                  # AS requirement for SCL on bus F=30  , dual variable: λ_F  
end



#-------Define Dual Constraints   
n=1    # for bus 11
k=11
@constraint(model, Oⁿˡ[1]  -K_g[1,k]*λ_F[n,T] -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[T] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[T] +Kˢᵗ[1]*σˢᵗˢᴳ²_1[T] -Kˢʰ[1]*σˢʰˢᴳ²_1[T]+ ψᵐᵃˣˢᴳ²_1[T]  -γ_max[1,1,T] +γ_min[1,1,T]
-γ_max[2,1,T] +γ_min[2,1,T] -γ_max[3,1,T] +γ_min[3,1,T] -γ_max[4,1,T] +γ_min[4,1,T] -γ_max[5,1,T] +γ_min[5,1,T]
-γ_max[6,1,T] +γ_min[6,1,T] -γ_max[7,1,T] +γ_min[7,1,T] -γ_max[8,1,T] +γ_min[8,1,T] -γ_max[9,1,T] +γ_min[9,1,T]
-γ_max[10,1,T] +γ_min[10,1,T] -γ_max[11,1,T] +γ_min[11,1,T]>=0)             # dual constraints for UC, when t==T

@constraint(model, Oⁿˡ[1]  -K_g[2,k]*λ_F[n,T] -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[T] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[T] +Kˢᵗ[1]*σˢᵗˢᴳ²_2[T] -Kˢʰ[1]*σˢʰˢᴳ²_2[T]+ ψᵐᵃˣˢᴳ²_2[T] -γ_max[1,2,T] +γ_min[1,1,T]
-γ_max[12,1,T] +γ_min[12,1,T] -γ_max[13,1,T] +γ_min[13,1,T] -γ_max[14,1,T] +γ_min[14,1,T] -γ_max[15,1,T] +γ_min[15,1,T]
-γ_max[16,1,T] +γ_min[16,1,T] -γ_max[17,1,T] +γ_min[17,1,T] -γ_max[18,1,T] +γ_min[18,1,T] -γ_max[19,1,T] +γ_min[19,1,T]
-γ_max[20,1,T] +γ_min[20,1,T] -γ_max[21,1,T] +γ_min[21,1,T]>=0)

@constraint(model, Oⁿˡ[2]  -K_g[3,k]*λ_F[n,T] -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[T] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[T] +Kˢᵗ[2]*σˢᵗˢᴳ³_1[T] -Kˢʰ[2]*σˢʰˢᴳ³_1[T]+ ψᵐᵃˣˢᴳ³_1[T] -γ_max[2,2,T] +γ_min[2,1,T]
-γ_max[12,2,T] +γ_min[12,1,T] -γ_max[22,1,T] +γ_min[22,1,T] -γ_max[23,1,T] +γ_min[23,1,T] -γ_max[24,1,T] +γ_min[24,1,T]
-γ_max[25,1,T] +γ_min[25,1,T] -γ_max[26,1,T] +γ_min[26,1,T] -γ_max[27,1,T] +γ_min[27,1,T] -γ_max[28,1,T] +γ_min[28,1,T]
-γ_max[29,1,T] +γ_min[29,1,T] -γ_max[30,1,T] +γ_min[30,1,T]>=0)

@constraint(model, Oⁿˡ[2]  -K_g[4,k]*λ_F[n,T] -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[T] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[T] +Kˢᵗ[2]*σˢᵗˢᴳ³_2[T] -Kˢʰ[2]*σˢʰˢᴳ³_2[T]+ ψᵐᵃˣˢᴳ³_2[T] -γ_max[3,2,T] +γ_min[3,1,T]
-γ_max[13,2,T] +γ_min[13,1,T] -γ_max[22,2,T] +γ_min[22,1,T] -γ_max[31,1,T] +γ_min[31,1,T] -γ_max[32,1,T] +γ_min[32,1,T]
-γ_max[33,1,T] +γ_min[33,1,T] -γ_max[34,1,T] +γ_min[34,1,T] -γ_max[35,1,T] +γ_min[35,1,T] -γ_max[36,1,T] +γ_min[36,1,T]
-γ_max[37,1,T] +γ_min[37,1,T] -γ_max[38,1,T] +γ_min[38,1,T]>=0)

@constraint(model, Oⁿˡ[3]  -K_g[5,k]*λ_F[n,T] -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[T] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[T] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_1[T] -Kˢʰ[3]*σˢʰˢᴳ⁴_1[T]+ ψᵐᵃˣˢᴳ⁴_1[T] -γ_max[4,2,T] +γ_min[4,1,T]
-γ_max[14,2,T] +γ_min[14,1,T] -γ_max[23,2,T] +γ_min[23,1,T] -γ_max[31,2,T] +γ_min[31,1,T] -γ_max[39,1,T] +γ_min[39,1,T]
-γ_max[40,1,T] +γ_min[40,1,T] -γ_max[41,1,T] +γ_min[41,1,T] -γ_max[42,1,T] +γ_min[42,1,T] -γ_max[43,1,T] +γ_min[43,1,T]
-γ_max[44,1,T] +γ_min[44,1,T] -γ_max[45,1,T] +γ_min[45,1,T]>=0)


@constraint(model, Oⁿˡ[3]  -K_g[6,k]*λ_F[n,T] -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[T] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[T] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_2[T] -Kˢʰ[3]*σˢʰˢᴳ⁴_2[T]+ ψᵐᵃˣˢᴳ⁴_2[T] -γ_max[5,2,T] +γ_min[5,1,T]
-γ_max[15,2,T] +γ_min[15,1,T] -γ_max[24,2,T] +γ_min[24,1,T] -γ_max[32,2,T] +γ_min[32,1,T] -γ_max[39,2,T] +γ_min[39,1,T]
-γ_max[46,1,T] +γ_min[46,1,T] -γ_max[47,1,T] +γ_min[47,1,T] -γ_max[48,1,T] +γ_min[48,1,T] -γ_max[49,1,T] +γ_min[49,1,T]
-γ_max[50,1,T] +γ_min[50,1,T] -γ_max[51,1,T] +γ_min[51,1,T]>=0)


@constraint(model, Oⁿˡ[4]  -K_g[7,k]*λ_F[n,T] -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[T] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[T] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_1[T] -Kˢʰ[4]*σˢʰˢᴳ⁵_1[T]+ ψᵐᵃˣˢᴳ⁵_1[T] -γ_max[6,2,T] +γ_min[6,1,T]
-γ_max[16,2,T] +γ_min[16,1,T] -γ_max[25,2,T] +γ_min[25,1,T] -γ_max[33,2,T] +γ_min[33,1,T] -γ_max[40,2,T] +γ_min[40,1,T]
-γ_max[46,2,T] +γ_min[46,1,T] -γ_max[52,1,T] +γ_min[52,1,T] -γ_max[53,1,T] +γ_min[53,1,T] -γ_max[54,1,T] +γ_min[54,1,T]
-γ_max[55,1,T] +γ_min[55,1,T] -γ_max[56,1,T] +γ_min[56,1,T]>=0)


@constraint(model, Oⁿˡ[4]  -K_g[8,k]*λ_F[n,T] -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[T] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[T] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_2[T] -Kˢʰ[4]*σˢʰˢᴳ⁵_2[T]+ ψᵐᵃˣˢᴳ⁵_2[T] -γ_max[7,2,T] +γ_min[7,1,T]
-γ_max[17,2,T] +γ_min[17,1,T] -γ_max[26,2,T] +γ_min[26,1,T] -γ_max[34,2,T] +γ_min[34,1,T] -γ_max[41,2,T] +γ_min[41,1,T]
-γ_max[47,2,T] +γ_min[47,1,T] -γ_max[52,2,T] +γ_min[52,1,T] -γ_max[57,1,T] +γ_min[57,1,T] -γ_max[58,1,T] +γ_min[58,1,T]
-γ_max[59,1,T] +γ_min[59,1,T] -γ_max[60,1,T] +γ_min[60,1,T]>=0)


@constraint(model, Oⁿˡ[5]  -K_g[9,k]*λ_F[n,T] -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[T] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[T] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_1[T] -Kˢʰ[5]*σˢʰˢᴳ²⁷_1[T]+ ψᵐᵃˣˢᴳ²⁷_1[T] -γ_max[8,2,T] +γ_min[8,1,T]
-γ_max[18,2,T] +γ_min[18,1,T] -γ_max[27,2,T] +γ_min[27,1,T] -γ_max[35,2,T] +γ_min[35,1,T] -γ_max[42,2,T] +γ_min[42,1,T]
-γ_max[48,2,T] +γ_min[48,1,T] -γ_max[53,2,T] +γ_min[53,1,T] -γ_max[57,2,T] +γ_min[57,1,T] -γ_max[61,1,T] +γ_min[61,1,T]
-γ_max[62,1,T] +γ_min[62,1,T] -γ_max[63,1,T] +γ_min[63,1,T]>=0)


@constraint(model, Oⁿˡ[5]  -K_g[10,k]*λ_F[n,T] -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[T] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[T] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_2[T] -Kˢʰ[5]*σˢʰˢᴳ²⁷_2[T]+ ψᵐᵃˣˢᴳ²⁷_2[T] -γ_max[9,2,T] +γ_min[9,1,T]
-γ_max[19,2,T] +γ_min[19,1,T] -γ_max[28,2,T] +γ_min[28,1,T] -γ_max[36,2,T] +γ_min[36,1,T] -γ_max[43,2,T] +γ_min[43,1,T]
-γ_max[49,2,T] +γ_min[49,1,T] -γ_max[54,2,T] +γ_min[54,1,T] -γ_max[58,2,T] +γ_min[58,1,T] -γ_max[61,2,T] +γ_min[61,1,T]
-γ_max[64,1,T] +γ_min[64,1,T] -γ_max[65,1,T] +γ_min[65,1,T]>=0)


@constraint(model, Oⁿˡ[6]  -K_g[11,k]*λ_F[n,T] -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[T] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[T] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_1[T] -Kˢʰ[6]*σˢʰˢᴳ³⁰_1[T]+ ψᵐᵃˣˢᴳ³⁰_1[T] -γ_max[10,2,T] +γ_min[10,1,T]
-γ_max[20,2,T] +γ_min[20,1,T] -γ_max[29,2,T] +γ_min[29,1,T] -γ_max[37,2,T] +γ_min[37,1,T] -γ_max[44,2,T] +γ_min[44,1,T]
-γ_max[50,2,T] +γ_min[50,1,T] -γ_max[55,2,T] +γ_min[55,1,T] -γ_max[59,2,T] +γ_min[59,1,T] -γ_max[62,2,T] +γ_min[62,1,T]
-γ_max[64,2,T] +γ_min[64,1,T] -γ_max[66,1,T] +γ_min[66,1,T]>=0)


@constraint(model, Oⁿˡ[6]  -K_g[12,k]*λ_F[n,T] -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[T] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[T] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_2[T] -Kˢʰ[6]*σˢʰˢᴳ³⁰_2[T]+ ψᵐᵃˣˢᴳ³⁰_2[T] -γ_max[11,2,T] +γ_min[11,1,T]
-γ_max[21,2,T] +γ_min[21,1,T] -γ_max[30,2,T] +γ_min[30,1,T] -γ_max[38,2,T] +γ_min[38,1,T] -γ_max[45,2,T] +γ_min[45,1,T]
-γ_max[51,2,T] +γ_min[51,1,T] -γ_max[56,2,T] +γ_min[56,1,T] -γ_max[60,2,T] +γ_min[60,1,T] -γ_max[63,2,T] +γ_min[63,1,T]
-γ_max[65,2,T] +γ_min[65,1,T] -γ_max[66,2,T] +γ_min[66,1,T]>=0)


for t in 1:T-1                                                                                                                                                       # dual constraints for UC, when t<=T-1
    @constraint(model, Oⁿˡ[1]  -K_g[1,k]*λ_F[n,t] - Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[t]+ Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[t]+ Kˢᵗ[1]*(σˢᵗˢᴳ²_1[t]-σˢᵗˢᴳ²_1[t+1])+ Kˢʰ[1]*(σˢᵗˢᴳ²_1[t+1]-σˢᵗˢᴳ²_1[t])+ ψᵐᵃˣˢᴳ²_1[t] -γ_max[1,1,t] +γ_min[1,1,t]
    -γ_max[2,1,t] +γ_min[2,1,t] -γ_max[3,1,t] +γ_min[3,1,t] -γ_max[4,1,t] +γ_min[4,1,t] -γ_max[5,1,t] +γ_min[5,1,t]
    -γ_max[6,1,t] +γ_min[6,1,t] -γ_max[7,1,t] +γ_min[7,1,t] -γ_max[8,1,t] +γ_min[8,1,t] -γ_max[9,1,t] +γ_min[9,1,t]
    -γ_max[10,1,t] +γ_min[10,1,t] -γ_max[11,1,t] +γ_min[11,1,t]>=0)

    @constraint(model, Oⁿˡ[1]  -K_g[2,k]*λ_F[n,t] - Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[t]+ Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[t]+ Kˢᵗ[1]*(σˢᵗˢᴳ²_2[t]-σˢᵗˢᴳ²_2[t+1])+ Kˢʰ[1]*(σˢᵗˢᴳ²_2[t+1]-σˢᵗˢᴳ²_2[t])+ ψᵐᵃˣˢᴳ²_2[t] -γ_max[1,2,t] +γ_min[1,1,t]
    -γ_max[12,1,t] +γ_min[12,1,t] -γ_max[13,1,t] +γ_min[13,1,t] -γ_max[14,1,t] +γ_min[14,1,t] -γ_max[15,1,t] +γ_min[15,1,t]
    -γ_max[16,1,t] +γ_min[16,1,t] -γ_max[17,1,t] +γ_min[17,1,t] -γ_max[18,1,t] +γ_min[18,1,t] -γ_max[19,1,t] +γ_min[19,1,t]
    -γ_max[20,1,t] +γ_min[20,1,t] -γ_max[21,1,t] +γ_min[21,1,t]>=0)

    @constraint(model, Oⁿˡ[2]  -K_g[3,k]*λ_F[n,t] - Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[t]+ Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[t]+ Kˢᵗ[2]*(σˢᵗˢᴳ³_1[t]-σˢᵗˢᴳ³_1[t+1])+ Kˢʰ[2]*(σˢᵗˢᴳ³_1[t+1]-σˢᵗˢᴳ³_1[t])+ ψᵐᵃˣˢᴳ³_1[t] -γ_max[2,2,t] +γ_min[2,1,t]
    -γ_max[12,2,t] +γ_min[12,1,t] -γ_max[22,1,t] +γ_min[22,1,t] -γ_max[23,1,t] +γ_min[23,1,t] -γ_max[24,1,t] +γ_min[24,1,t]
    -γ_max[25,1,t] +γ_min[25,1,t] -γ_max[26,1,t] +γ_min[26,1,t] -γ_max[27,1,t] +γ_min[27,1,t] -γ_max[28,1,t] +γ_min[28,1,t]
    -γ_max[29,1,t] +γ_min[29,1,t] -γ_max[30,1,t] +γ_min[30,1,t]>=0)

    @constraint(model, Oⁿˡ[2]  -K_g[4,k]*λ_F[n,t] - Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[t]+ Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[t]+ Kˢᵗ[2]*(σˢᵗˢᴳ³_2[t]-σˢᵗˢᴳ³_2[t+1])+ Kˢʰ[2]*(σˢᵗˢᴳ³_2[t+1]-σˢᵗˢᴳ³_2[t])+ ψᵐᵃˣˢᴳ³_2[t] -γ_max[3,2,t] +γ_min[3,1,t]
    -γ_max[13,2,t] +γ_min[13,1,t] -γ_max[22,2,t] +γ_min[22,1,t] -γ_max[31,1,t] +γ_min[31,1,t] -γ_max[32,1,t] +γ_min[32,1,t]
    -γ_max[33,1,t] +γ_min[33,1,t] -γ_max[34,1,t] +γ_min[34,1,t] -γ_max[35,1,t] +γ_min[35,1,t] -γ_max[36,1,t] +γ_min[36,1,t]
    -γ_max[37,1,t] +γ_min[37,1,t] -γ_max[38,1,t] +γ_min[38,1,t]>=0)

    @constraint(model, Oⁿˡ[3]  -K_g[5,k]*λ_F[n,t] - Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[t]+ Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[t]+ Kˢᵗ[3]*(σˢᵗˢᴳ⁴_1[t]-σˢᵗˢᴳ⁴_1[t+1])+ Kˢʰ[3]*(σˢᵗˢᴳ⁴_1[t+1]-σˢᵗˢᴳ⁴_1[t])+ ψᵐᵃˣˢᴳ⁴_1[t] -γ_max[4,2,t] +γ_min[4,1,t]
    -γ_max[14,2,t] +γ_min[14,1,t] -γ_max[23,2,t] +γ_min[23,1,t] -γ_max[31,2,t] +γ_min[31,1,t] -γ_max[39,1,t] +γ_min[39,1,t]
    -γ_max[40,1,t] +γ_min[40,1,t] -γ_max[41,1,t] +γ_min[41,1,t] -γ_max[42,1,t] +γ_min[42,1,t] -γ_max[43,1,t] +γ_min[43,1,t]
    -γ_max[44,1,t] +γ_min[44,1,t] -γ_max[45,1,t] +γ_min[45,1,t]>=0)

    @constraint(model, Oⁿˡ[3]  -K_g[6,k]*λ_F[n,t] - Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[t]+ Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[t]+ Kˢᵗ[3]*(σˢᵗˢᴳ⁴_2[t]-σˢᵗˢᴳ⁴_2[t+1])+ Kˢʰ[3]*(σˢᵗˢᴳ⁴_2[t+1]-σˢᵗˢᴳ⁴_2[t])+ ψᵐᵃˣˢᴳ⁴_2[t]-γ_max[5,2,t] +γ_min[5,1,t]
    -γ_max[15,2,t] +γ_min[15,1,t] -γ_max[24,2,t] +γ_min[24,1,t] -γ_max[32,2,t] +γ_min[32,1,t] -γ_max[39,2,t] +γ_min[39,1,t]
    -γ_max[46,1,t] +γ_min[46,1,t] -γ_max[47,1,t] +γ_min[47,1,t] -γ_max[48,1,t] +γ_min[48,1,t] -γ_max[49,1,t] +γ_min[49,1,t]
    -γ_max[50,1,t] +γ_min[50,1,t] -γ_max[51,1,t] +γ_min[51,1,t]>=0)
    
    @constraint(model, Oⁿˡ[4]  -K_g[7,k]*λ_F[n,t] - Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[t]+ Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[t]+ Kˢᵗ[4]*(σˢᵗˢᴳ⁵_1[t]-σˢᵗˢᴳ⁵_1[t+1])+ Kˢʰ[4]*(σˢᵗˢᴳ⁵_1[t+1]-σˢᵗˢᴳ⁵_1[t])+ ψᵐᵃˣˢᴳ⁵_1[t] -γ_max[6,2,t] +γ_min[6,1,t]
    -γ_max[16,2,t] +γ_min[16,1,t] -γ_max[25,2,t] +γ_min[25,1,t] -γ_max[33,2,t] +γ_min[33,1,t] -γ_max[40,2,t] +γ_min[40,1,t]
    -γ_max[46,2,t] +γ_min[46,1,t] -γ_max[52,1,t] +γ_min[52,1,t] -γ_max[53,1,t] +γ_min[53,1,t] -γ_max[54,1,t] +γ_min[54,1,t]
    -γ_max[55,1,t] +γ_min[55,1,t] -γ_max[56,1,t] +γ_min[56,1,t]>=0)

    @constraint(model, Oⁿˡ[4]  -K_g[8,k]*λ_F[n,t] - Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[t]+ Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[t]+ Kˢᵗ[4]*(σˢᵗˢᴳ⁵_2[t]-σˢᵗˢᴳ⁵_2[t+1])+ Kˢʰ[4]*(σˢᵗˢᴳ⁵_2[t+1]-σˢᵗˢᴳ⁵_2[t])+ ψᵐᵃˣˢᴳ⁵_2[t] -γ_max[7,2,t] +γ_min[7,1,t]
    -γ_max[17,2,t] +γ_min[17,1,t] -γ_max[26,2,t] +γ_min[26,1,t] -γ_max[34,2,t] +γ_min[34,1,t] -γ_max[41,2,t] +γ_min[41,1,t]
    -γ_max[47,2,t] +γ_min[47,1,t] -γ_max[52,2,t] +γ_min[52,1,t] -γ_max[57,1,t] +γ_min[57,1,t] -γ_max[58,1,t] +γ_min[58,1,t]
    -γ_max[59,1,t] +γ_min[59,1,t] -γ_max[60,1,t] +γ_min[60,1,t]>=0)
    
    @constraint(model, Oⁿˡ[5]  -K_g[9,k]*λ_F[n,t] - Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[t]+ Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[t]+ Kˢᵗ[5]*(σˢᵗˢᴳ²⁷_1[t]-σˢᵗˢᴳ²⁷_1[t+1])+ Kˢʰ[5]*(σˢᵗˢᴳ²⁷_1[t+1]-σˢᵗˢᴳ²⁷_1[t])+ ψᵐᵃˣˢᴳ²⁷_1[t] -γ_max[8,2,t] +γ_min[8,1,t]
    -γ_max[18,2,t] +γ_min[18,1,t] -γ_max[27,2,t] +γ_min[27,1,t] -γ_max[35,2,t] +γ_min[35,1,t] -γ_max[42,2,t] +γ_min[42,1,t]
    -γ_max[48,2,t] +γ_min[48,1,t] -γ_max[53,2,t] +γ_min[53,1,t] -γ_max[57,2,t] +γ_min[57,1,t] -γ_max[61,1,t] +γ_min[61,1,t]
    -γ_max[62,1,t] +γ_min[62,1,t] -γ_max[63,1,t] +γ_min[63,1,t]>=0)

    @constraint(model, Oⁿˡ[5]  -K_g[10,k]*λ_F[n,t] - Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[t]+ Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[t]+ Kˢᵗ[5]*(σˢᵗˢᴳ²⁷_2[t]-σˢᵗˢᴳ²⁷_2[t+1])+ Kˢʰ[5]*(σˢᵗˢᴳ²⁷_2[t+1]-σˢᵗˢᴳ²⁷_2[t])+ ψᵐᵃˣˢᴳ²⁷_2[t] -γ_max[9,2,t] +γ_min[9,1,t]
    -γ_max[19,2,t] +γ_min[19,1,t] -γ_max[28,2,t] +γ_min[28,1,t] -γ_max[36,2,t] +γ_min[36,1,t] -γ_max[43,2,t] +γ_min[43,1,t]
    -γ_max[49,2,t] +γ_min[49,1,t] -γ_max[54,2,t] +γ_min[54,1,t] -γ_max[58,2,t] +γ_min[58,1,t] -γ_max[61,2,t] +γ_min[61,1,t]
    -γ_max[64,1,t] +γ_min[64,1,t] -γ_max[65,1,t] +γ_min[65,1,t]>=0)
    
    @constraint(model, Oⁿˡ[6]  -K_g[11,k]*λ_F[n,t] - Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[t]+ Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[t]+ Kˢᵗ[6]*(σˢᵗˢᴳ³⁰_1[t]-σˢᵗˢᴳ³⁰_1[t+1])+ Kˢʰ[6]*(σˢᵗˢᴳ³⁰_1[t+1]-σˢᵗˢᴳ³⁰_1[t])+ ψᵐᵃˣˢᴳ³⁰_1[t] -γ_max[10,2,t] +γ_min[10,1,t]
    -γ_max[20,2,t] +γ_min[20,1,t] -γ_max[29,2,t] +γ_min[29,1,t] -γ_max[37,2,t] +γ_min[37,1,t] -γ_max[44,2,t] +γ_min[44,1,t]
    -γ_max[50,2,t] +γ_min[50,1,t] -γ_max[55,2,t] +γ_min[55,1,t] -γ_max[59,2,t] +γ_min[59,1,t] -γ_max[62,2,t] +γ_min[62,1,t]
    -γ_max[64,2,t] +γ_min[64,1,t] -γ_max[66,1,t] +γ_min[66,1,t]>=0)    

    @constraint(model, Oⁿˡ[6]  -K_g[12,k]*λ_F[n,t] - Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[t]+ Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[t]+ Kˢᵗ[6]*(σˢᵗˢᴳ³⁰_2[t]-σˢᵗˢᴳ³⁰_2[t+1])+ Kˢʰ[6]*(σˢᵗˢᴳ³⁰_2[t+1]-σˢᵗˢᴳ³⁰_2[t])+ ψᵐᵃˣˢᴳ³⁰_2[t] -γ_max[11,2,t] +γ_min[11,1,t]
    -γ_max[21,2,t] +γ_min[21,1,t] -γ_max[30,2,t] +γ_min[30,1,t] -γ_max[38,2,t] +γ_min[38,1,t] -γ_max[45,2,t] +γ_min[45,1,t]
    -γ_max[51,2,t] +γ_min[51,1,t] -γ_max[56,2,t] +γ_min[56,1,t] -γ_max[60,2,t] +γ_min[60,1,t] -γ_max[63,2,t] +γ_min[63,1,t]
    -γ_max[65,2,t] +γ_min[65,1,t] -γ_max[66,2,t] +γ_min[66,1,t]>=0)  

end

n=2    # for bus 26
k=26
@constraint(model, Oⁿˡ[1]  -K_g[1,k]*λ_F[n,T] -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[T] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[T] +Kˢᵗ[1]*σˢᵗˢᴳ²_1[T] -Kˢʰ[1]*σˢʰˢᴳ²_1[T]+ ψᵐᵃˣˢᴳ²_1[T]  -γ_max[1,1,T] +γ_min[1,1,T]
-γ_max[2,1,T] +γ_min[2,1,T] -γ_max[3,1,T] +γ_min[3,1,T] -γ_max[4,1,T] +γ_min[4,1,T] -γ_max[5,1,T] +γ_min[5,1,T]
-γ_max[6,1,T] +γ_min[6,1,T] -γ_max[7,1,T] +γ_min[7,1,T] -γ_max[8,1,T] +γ_min[8,1,T] -γ_max[9,1,T] +γ_min[9,1,T]
-γ_max[10,1,T] +γ_min[10,1,T] -γ_max[11,1,T] +γ_min[11,1,T]>=0)             # dual constraints for UC, when t==T

@constraint(model, Oⁿˡ[1]  -K_g[2,k]*λ_F[n,T] -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[T] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[T] +Kˢᵗ[1]*σˢᵗˢᴳ²_2[T] -Kˢʰ[1]*σˢʰˢᴳ²_2[T]+ ψᵐᵃˣˢᴳ²_2[T] -γ_max[1,2,T] +γ_min[1,1,T]
-γ_max[12,1,T] +γ_min[12,1,T] -γ_max[13,1,T] +γ_min[13,1,T] -γ_max[14,1,T] +γ_min[14,1,T] -γ_max[15,1,T] +γ_min[15,1,T]
-γ_max[16,1,T] +γ_min[16,1,T] -γ_max[17,1,T] +γ_min[17,1,T] -γ_max[18,1,T] +γ_min[18,1,T] -γ_max[19,1,T] +γ_min[19,1,T]
-γ_max[20,1,T] +γ_min[20,1,T] -γ_max[21,1,T] +γ_min[21,1,T]>=0)

@constraint(model, Oⁿˡ[2]  -K_g[3,k]*λ_F[n,T] -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[T] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[T] +Kˢᵗ[2]*σˢᵗˢᴳ³_1[T] -Kˢʰ[2]*σˢʰˢᴳ³_1[T]+ ψᵐᵃˣˢᴳ³_1[T] -γ_max[2,2,T] +γ_min[2,1,T]
-γ_max[12,2,T] +γ_min[12,1,T] -γ_max[22,1,T] +γ_min[22,1,T] -γ_max[23,1,T] +γ_min[23,1,T] -γ_max[24,1,T] +γ_min[24,1,T]
-γ_max[25,1,T] +γ_min[25,1,T] -γ_max[26,1,T] +γ_min[26,1,T] -γ_max[27,1,T] +γ_min[27,1,T] -γ_max[28,1,T] +γ_min[28,1,T]
-γ_max[29,1,T] +γ_min[29,1,T] -γ_max[30,1,T] +γ_min[30,1,T]>=0)

@constraint(model, Oⁿˡ[2]  -K_g[4,k]*λ_F[n,T] -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[T] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[T] +Kˢᵗ[2]*σˢᵗˢᴳ³_2[T] -Kˢʰ[2]*σˢʰˢᴳ³_2[T]+ ψᵐᵃˣˢᴳ³_2[T] -γ_max[3,2,T] +γ_min[3,1,T]
-γ_max[13,2,T] +γ_min[13,1,T] -γ_max[22,2,T] +γ_min[22,1,T] -γ_max[31,1,T] +γ_min[31,1,T] -γ_max[32,1,T] +γ_min[32,1,T]
-γ_max[33,1,T] +γ_min[33,1,T] -γ_max[34,1,T] +γ_min[34,1,T] -γ_max[35,1,T] +γ_min[35,1,T] -γ_max[36,1,T] +γ_min[36,1,T]
-γ_max[37,1,T] +γ_min[37,1,T] -γ_max[38,1,T] +γ_min[38,1,T]>=0)

@constraint(model, Oⁿˡ[3]  -K_g[5,k]*λ_F[n,T] -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[T] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[T] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_1[T] -Kˢʰ[3]*σˢʰˢᴳ⁴_1[T]+ ψᵐᵃˣˢᴳ⁴_1[T] -γ_max[4,2,T] +γ_min[4,1,T]
-γ_max[14,2,T] +γ_min[14,1,T] -γ_max[23,2,T] +γ_min[23,1,T] -γ_max[31,2,T] +γ_min[31,1,T] -γ_max[39,1,T] +γ_min[39,1,T]
-γ_max[40,1,T] +γ_min[40,1,T] -γ_max[41,1,T] +γ_min[41,1,T] -γ_max[42,1,T] +γ_min[42,1,T] -γ_max[43,1,T] +γ_min[43,1,T]
-γ_max[44,1,T] +γ_min[44,1,T] -γ_max[45,1,T] +γ_min[45,1,T]>=0)


@constraint(model, Oⁿˡ[3]  -K_g[6,k]*λ_F[n,T] -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[T] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[T] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_2[T] -Kˢʰ[3]*σˢʰˢᴳ⁴_2[T]+ ψᵐᵃˣˢᴳ⁴_2[T] -γ_max[5,2,T] +γ_min[5,1,T]
-γ_max[15,2,T] +γ_min[15,1,T] -γ_max[24,2,T] +γ_min[24,1,T] -γ_max[32,2,T] +γ_min[32,1,T] -γ_max[39,2,T] +γ_min[39,1,T]
-γ_max[46,1,T] +γ_min[46,1,T] -γ_max[47,1,T] +γ_min[47,1,T] -γ_max[48,1,T] +γ_min[48,1,T] -γ_max[49,1,T] +γ_min[49,1,T]
-γ_max[50,1,T] +γ_min[50,1,T] -γ_max[51,1,T] +γ_min[51,1,T]>=0)


@constraint(model, Oⁿˡ[4]  -K_g[7,k]*λ_F[n,T] -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[T] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[T] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_1[T] -Kˢʰ[4]*σˢʰˢᴳ⁵_1[T]+ ψᵐᵃˣˢᴳ⁵_1[T] -γ_max[6,2,T] +γ_min[6,1,T]
-γ_max[16,2,T] +γ_min[16,1,T] -γ_max[25,2,T] +γ_min[25,1,T] -γ_max[33,2,T] +γ_min[33,1,T] -γ_max[40,2,T] +γ_min[40,1,T]
-γ_max[46,2,T] +γ_min[46,1,T] -γ_max[52,1,T] +γ_min[52,1,T] -γ_max[53,1,T] +γ_min[53,1,T] -γ_max[54,1,T] +γ_min[54,1,T]
-γ_max[55,1,T] +γ_min[55,1,T] -γ_max[56,1,T] +γ_min[56,1,T]>=0)


@constraint(model, Oⁿˡ[4]  -K_g[8,k]*λ_F[n,T] -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[T] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[T] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_2[T] -Kˢʰ[4]*σˢʰˢᴳ⁵_2[T]+ ψᵐᵃˣˢᴳ⁵_2[T] -γ_max[7,2,T] +γ_min[7,1,T]
-γ_max[17,2,T] +γ_min[17,1,T] -γ_max[26,2,T] +γ_min[26,1,T] -γ_max[34,2,T] +γ_min[34,1,T] -γ_max[41,2,T] +γ_min[41,1,T]
-γ_max[47,2,T] +γ_min[47,1,T] -γ_max[52,2,T] +γ_min[52,1,T] -γ_max[57,1,T] +γ_min[57,1,T] -γ_max[58,1,T] +γ_min[58,1,T]
-γ_max[59,1,T] +γ_min[59,1,T] -γ_max[60,1,T] +γ_min[60,1,T]>=0)


@constraint(model, Oⁿˡ[5]  -K_g[9,k]*λ_F[n,T] -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[T] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[T] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_1[T] -Kˢʰ[5]*σˢʰˢᴳ²⁷_1[T]+ ψᵐᵃˣˢᴳ²⁷_1[T] -γ_max[8,2,T] +γ_min[8,1,T]
-γ_max[18,2,T] +γ_min[18,1,T] -γ_max[27,2,T] +γ_min[27,1,T] -γ_max[35,2,T] +γ_min[35,1,T] -γ_max[42,2,T] +γ_min[42,1,T]
-γ_max[48,2,T] +γ_min[48,1,T] -γ_max[53,2,T] +γ_min[53,1,T] -γ_max[57,2,T] +γ_min[57,1,T] -γ_max[61,1,T] +γ_min[61,1,T]
-γ_max[62,1,T] +γ_min[62,1,T] -γ_max[63,1,T] +γ_min[63,1,T]>=0)


@constraint(model, Oⁿˡ[5]  -K_g[10,k]*λ_F[n,T] -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[T] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[T] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_2[T] -Kˢʰ[5]*σˢʰˢᴳ²⁷_2[T]+ ψᵐᵃˣˢᴳ²⁷_2[T] -γ_max[9,2,T] +γ_min[9,1,T]
-γ_max[19,2,T] +γ_min[19,1,T] -γ_max[28,2,T] +γ_min[28,1,T] -γ_max[36,2,T] +γ_min[36,1,T] -γ_max[43,2,T] +γ_min[43,1,T]
-γ_max[49,2,T] +γ_min[49,1,T] -γ_max[54,2,T] +γ_min[54,1,T] -γ_max[58,2,T] +γ_min[58,1,T] -γ_max[61,2,T] +γ_min[61,1,T]
-γ_max[64,1,T] +γ_min[64,1,T] -γ_max[65,1,T] +γ_min[65,1,T]>=0)


@constraint(model, Oⁿˡ[6]  -K_g[11,k]*λ_F[n,T] -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[T] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[T] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_1[T] -Kˢʰ[6]*σˢʰˢᴳ³⁰_1[T]+ ψᵐᵃˣˢᴳ³⁰_1[T] -γ_max[10,2,T] +γ_min[10,1,T]
-γ_max[20,2,T] +γ_min[20,1,T] -γ_max[29,2,T] +γ_min[29,1,T] -γ_max[37,2,T] +γ_min[37,1,T] -γ_max[44,2,T] +γ_min[44,1,T]
-γ_max[50,2,T] +γ_min[50,1,T] -γ_max[55,2,T] +γ_min[55,1,T] -γ_max[59,2,T] +γ_min[59,1,T] -γ_max[62,2,T] +γ_min[62,1,T]
-γ_max[64,2,T] +γ_min[64,1,T] -γ_max[66,1,T] +γ_min[66,1,T]>=0)


@constraint(model, Oⁿˡ[6]  -K_g[12,k]*λ_F[n,T] -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[T] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[T] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_2[T] -Kˢʰ[6]*σˢʰˢᴳ³⁰_2[T]+ ψᵐᵃˣˢᴳ³⁰_2[T] -γ_max[11,2,T] +γ_min[11,1,T]
-γ_max[21,2,T] +γ_min[21,1,T] -γ_max[30,2,T] +γ_min[30,1,T] -γ_max[38,2,T] +γ_min[38,1,T] -γ_max[45,2,T] +γ_min[45,1,T]
-γ_max[51,2,T] +γ_min[51,1,T] -γ_max[56,2,T] +γ_min[56,1,T] -γ_max[60,2,T] +γ_min[60,1,T] -γ_max[63,2,T] +γ_min[63,1,T]
-γ_max[65,2,T] +γ_min[65,1,T] -γ_max[66,2,T] +γ_min[66,1,T]>=0)



for t in 1:T-1                                                                                                                                                       # dual constraints for UC, when t<=T-1
    @constraint(model, Oⁿˡ[1]  -K_g[1,k]*λ_F[n,t] - Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[t]+ Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[t]+ Kˢᵗ[1]*(σˢᵗˢᴳ²_1[t]-σˢᵗˢᴳ²_1[t+1])+ Kˢʰ[1]*(σˢᵗˢᴳ²_1[t+1]-σˢᵗˢᴳ²_1[t])+ ψᵐᵃˣˢᴳ²_1[t] -γ_max[1,1,t] +γ_min[1,1,t]
    -γ_max[2,1,t] +γ_min[2,1,t] -γ_max[3,1,t] +γ_min[3,1,t] -γ_max[4,1,t] +γ_min[4,1,t] -γ_max[5,1,t] +γ_min[5,1,t]
    -γ_max[6,1,t] +γ_min[6,1,t] -γ_max[7,1,t] +γ_min[7,1,t] -γ_max[8,1,t] +γ_min[8,1,t] -γ_max[9,1,t] +γ_min[9,1,t]
    -γ_max[10,1,t] +γ_min[10,1,t] -γ_max[11,1,t] +γ_min[11,1,t]>=0)

    @constraint(model, Oⁿˡ[1]  -K_g[2,k]*λ_F[n,t] - Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[t]+ Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[t]+ Kˢᵗ[1]*(σˢᵗˢᴳ²_2[t]-σˢᵗˢᴳ²_2[t+1])+ Kˢʰ[1]*(σˢᵗˢᴳ²_2[t+1]-σˢᵗˢᴳ²_2[t])+ ψᵐᵃˣˢᴳ²_2[t] -γ_max[1,2,t] +γ_min[1,1,t]
    -γ_max[12,1,t] +γ_min[12,1,t] -γ_max[13,1,t] +γ_min[13,1,t] -γ_max[14,1,t] +γ_min[14,1,t] -γ_max[15,1,t] +γ_min[15,1,t]
    -γ_max[16,1,t] +γ_min[16,1,t] -γ_max[17,1,t] +γ_min[17,1,t] -γ_max[18,1,t] +γ_min[18,1,t] -γ_max[19,1,t] +γ_min[19,1,t]
    -γ_max[20,1,t] +γ_min[20,1,t] -γ_max[21,1,t] +γ_min[21,1,t]>=0)

    @constraint(model, Oⁿˡ[2]  -K_g[3,k]*λ_F[n,t] - Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[t]+ Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[t]+ Kˢᵗ[2]*(σˢᵗˢᴳ³_1[t]-σˢᵗˢᴳ³_1[t+1])+ Kˢʰ[2]*(σˢᵗˢᴳ³_1[t+1]-σˢᵗˢᴳ³_1[t])+ ψᵐᵃˣˢᴳ³_1[t] -γ_max[2,2,t] +γ_min[2,1,t]
    -γ_max[12,2,t] +γ_min[12,1,t] -γ_max[22,1,t] +γ_min[22,1,t] -γ_max[23,1,t] +γ_min[23,1,t] -γ_max[24,1,t] +γ_min[24,1,t]
    -γ_max[25,1,t] +γ_min[25,1,t] -γ_max[26,1,t] +γ_min[26,1,t] -γ_max[27,1,t] +γ_min[27,1,t] -γ_max[28,1,t] +γ_min[28,1,t]
    -γ_max[29,1,t] +γ_min[29,1,t] -γ_max[30,1,t] +γ_min[30,1,t]>=0)

    @constraint(model, Oⁿˡ[2]  -K_g[4,k]*λ_F[n,t] - Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[t]+ Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[t]+ Kˢᵗ[2]*(σˢᵗˢᴳ³_2[t]-σˢᵗˢᴳ³_2[t+1])+ Kˢʰ[2]*(σˢᵗˢᴳ³_2[t+1]-σˢᵗˢᴳ³_2[t])+ ψᵐᵃˣˢᴳ³_2[t] -γ_max[3,2,t] +γ_min[3,1,t]
    -γ_max[13,2,t] +γ_min[13,1,t] -γ_max[22,2,t] +γ_min[22,1,t] -γ_max[31,1,t] +γ_min[31,1,t] -γ_max[32,1,t] +γ_min[32,1,t]
    -γ_max[33,1,t] +γ_min[33,1,t] -γ_max[34,1,t] +γ_min[34,1,t] -γ_max[35,1,t] +γ_min[35,1,t] -γ_max[36,1,t] +γ_min[36,1,t]
    -γ_max[37,1,t] +γ_min[37,1,t] -γ_max[38,1,t] +γ_min[38,1,t]>=0)

    @constraint(model, Oⁿˡ[3]  -K_g[5,k]*λ_F[n,t] - Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[t]+ Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[t]+ Kˢᵗ[3]*(σˢᵗˢᴳ⁴_1[t]-σˢᵗˢᴳ⁴_1[t+1])+ Kˢʰ[3]*(σˢᵗˢᴳ⁴_1[t+1]-σˢᵗˢᴳ⁴_1[t])+ ψᵐᵃˣˢᴳ⁴_1[t] -γ_max[4,2,t] +γ_min[4,1,t]
    -γ_max[14,2,t] +γ_min[14,1,t] -γ_max[23,2,t] +γ_min[23,1,t] -γ_max[31,2,t] +γ_min[31,1,t] -γ_max[39,1,t] +γ_min[39,1,t]
    -γ_max[40,1,t] +γ_min[40,1,t] -γ_max[41,1,t] +γ_min[41,1,t] -γ_max[42,1,t] +γ_min[42,1,t] -γ_max[43,1,t] +γ_min[43,1,t]
    -γ_max[44,1,t] +γ_min[44,1,t] -γ_max[45,1,t] +γ_min[45,1,t]>=0)

    @constraint(model, Oⁿˡ[3]  -K_g[6,k]*λ_F[n,t] - Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[t]+ Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[t]+ Kˢᵗ[3]*(σˢᵗˢᴳ⁴_2[t]-σˢᵗˢᴳ⁴_2[t+1])+ Kˢʰ[3]*(σˢᵗˢᴳ⁴_2[t+1]-σˢᵗˢᴳ⁴_2[t])+ ψᵐᵃˣˢᴳ⁴_2[t]-γ_max[5,2,t] +γ_min[5,1,t]
    -γ_max[15,2,t] +γ_min[15,1,t] -γ_max[24,2,t] +γ_min[24,1,t] -γ_max[32,2,t] +γ_min[32,1,t] -γ_max[39,2,t] +γ_min[39,1,t]
    -γ_max[46,1,t] +γ_min[46,1,t] -γ_max[47,1,t] +γ_min[47,1,t] -γ_max[48,1,t] +γ_min[48,1,t] -γ_max[49,1,t] +γ_min[49,1,t]
    -γ_max[50,1,t] +γ_min[50,1,t] -γ_max[51,1,t] +γ_min[51,1,t]>=0)
    
    @constraint(model, Oⁿˡ[4]  -K_g[7,k]*λ_F[n,t] - Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[t]+ Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[t]+ Kˢᵗ[4]*(σˢᵗˢᴳ⁵_1[t]-σˢᵗˢᴳ⁵_1[t+1])+ Kˢʰ[4]*(σˢᵗˢᴳ⁵_1[t+1]-σˢᵗˢᴳ⁵_1[t])+ ψᵐᵃˣˢᴳ⁵_1[t] -γ_max[6,2,t] +γ_min[6,1,t]
    -γ_max[16,2,t] +γ_min[16,1,t] -γ_max[25,2,t] +γ_min[25,1,t] -γ_max[33,2,t] +γ_min[33,1,t] -γ_max[40,2,t] +γ_min[40,1,t]
    -γ_max[46,2,t] +γ_min[46,1,t] -γ_max[52,1,t] +γ_min[52,1,t] -γ_max[53,1,t] +γ_min[53,1,t] -γ_max[54,1,t] +γ_min[54,1,t]
    -γ_max[55,1,t] +γ_min[55,1,t] -γ_max[56,1,t] +γ_min[56,1,t]>=0)

    @constraint(model, Oⁿˡ[4]  -K_g[8,k]*λ_F[n,t] - Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[t]+ Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[t]+ Kˢᵗ[4]*(σˢᵗˢᴳ⁵_2[t]-σˢᵗˢᴳ⁵_2[t+1])+ Kˢʰ[4]*(σˢᵗˢᴳ⁵_2[t+1]-σˢᵗˢᴳ⁵_2[t])+ ψᵐᵃˣˢᴳ⁵_2[t] -γ_max[7,2,t] +γ_min[7,1,t]
    -γ_max[17,2,t] +γ_min[17,1,t] -γ_max[26,2,t] +γ_min[26,1,t] -γ_max[34,2,t] +γ_min[34,1,t] -γ_max[41,2,t] +γ_min[41,1,t]
    -γ_max[47,2,t] +γ_min[47,1,t] -γ_max[52,2,t] +γ_min[52,1,t] -γ_max[57,1,t] +γ_min[57,1,t] -γ_max[58,1,t] +γ_min[58,1,t]
    -γ_max[59,1,t] +γ_min[59,1,t] -γ_max[60,1,t] +γ_min[60,1,t]>=0)
    
    @constraint(model, Oⁿˡ[5]  -K_g[9,k]*λ_F[n,t] - Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[t]+ Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[t]+ Kˢᵗ[5]*(σˢᵗˢᴳ²⁷_1[t]-σˢᵗˢᴳ²⁷_1[t+1])+ Kˢʰ[5]*(σˢᵗˢᴳ²⁷_1[t+1]-σˢᵗˢᴳ²⁷_1[t])+ ψᵐᵃˣˢᴳ²⁷_1[t] -γ_max[8,2,t] +γ_min[8,1,t]
    -γ_max[18,2,t] +γ_min[18,1,t] -γ_max[27,2,t] +γ_min[27,1,t] -γ_max[35,2,t] +γ_min[35,1,t] -γ_max[42,2,t] +γ_min[42,1,t]
    -γ_max[48,2,t] +γ_min[48,1,t] -γ_max[53,2,t] +γ_min[53,1,t] -γ_max[57,2,t] +γ_min[57,1,t] -γ_max[61,1,t] +γ_min[61,1,t]
    -γ_max[62,1,t] +γ_min[62,1,t] -γ_max[63,1,t] +γ_min[63,1,t]>=0)

    @constraint(model, Oⁿˡ[5]  -K_g[10,k]*λ_F[n,t] - Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[t]+ Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[t]+ Kˢᵗ[5]*(σˢᵗˢᴳ²⁷_2[t]-σˢᵗˢᴳ²⁷_2[t+1])+ Kˢʰ[5]*(σˢᵗˢᴳ²⁷_2[t+1]-σˢᵗˢᴳ²⁷_2[t])+ ψᵐᵃˣˢᴳ²⁷_2[t] -γ_max[9,2,t] +γ_min[9,1,t]
    -γ_max[19,2,t] +γ_min[19,1,t] -γ_max[28,2,t] +γ_min[28,1,t] -γ_max[36,2,t] +γ_min[36,1,t] -γ_max[43,2,t] +γ_min[43,1,t]
    -γ_max[49,2,t] +γ_min[49,1,t] -γ_max[54,2,t] +γ_min[54,1,t] -γ_max[58,2,t] +γ_min[58,1,t] -γ_max[61,2,t] +γ_min[61,1,t]
    -γ_max[64,1,t] +γ_min[64,1,t] -γ_max[65,1,t] +γ_min[65,1,t]>=0)
    
    @constraint(model, Oⁿˡ[6]  -K_g[11,k]*λ_F[n,t] - Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[t]+ Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[t]+ Kˢᵗ[6]*(σˢᵗˢᴳ³⁰_1[t]-σˢᵗˢᴳ³⁰_1[t+1])+ Kˢʰ[6]*(σˢᵗˢᴳ³⁰_1[t+1]-σˢᵗˢᴳ³⁰_1[t])+ ψᵐᵃˣˢᴳ³⁰_1[t] -γ_max[10,2,t] +γ_min[10,1,t]
    -γ_max[20,2,t] +γ_min[20,1,t] -γ_max[29,2,t] +γ_min[29,1,t] -γ_max[37,2,t] +γ_min[37,1,t] -γ_max[44,2,t] +γ_min[44,1,t]
    -γ_max[50,2,t] +γ_min[50,1,t] -γ_max[55,2,t] +γ_min[55,1,t] -γ_max[59,2,t] +γ_min[59,1,t] -γ_max[62,2,t] +γ_min[62,1,t]
    -γ_max[64,2,t] +γ_min[64,1,t] -γ_max[66,1,t] +γ_min[66,1,t]>=0)    

    @constraint(model, Oⁿˡ[6]  -K_g[12,k]*λ_F[n,t] - Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[t]+ Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[t]+ Kˢᵗ[6]*(σˢᵗˢᴳ³⁰_2[t]-σˢᵗˢᴳ³⁰_2[t+1])+ Kˢʰ[6]*(σˢᵗˢᴳ³⁰_2[t+1]-σˢᵗˢᴳ³⁰_2[t])+ ψᵐᵃˣˢᴳ³⁰_2[t] -γ_max[11,2,t] +γ_min[11,1,t]
    -γ_max[21,2,t] +γ_min[21,1,t] -γ_max[30,2,t] +γ_min[30,1,t] -γ_max[38,2,t] +γ_min[38,1,t] -γ_max[45,2,t] +γ_min[45,1,t]
    -γ_max[51,2,t] +γ_min[51,1,t] -γ_max[56,2,t] +γ_min[56,1,t] -γ_max[60,2,t] +γ_min[60,1,t] -γ_max[63,2,t] +γ_min[63,1,t]
    -γ_max[65,2,t] +γ_min[65,1,t] -γ_max[66,2,t] +γ_min[66,1,t]>=0)  

end
            
n=3    # for bus 29
k=29
@constraint(model, Oⁿˡ[1]  -K_g[1,k]*λ_F[n,T] -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[T] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[T] +Kˢᵗ[1]*σˢᵗˢᴳ²_1[T] -Kˢʰ[1]*σˢʰˢᴳ²_1[T]+ ψᵐᵃˣˢᴳ²_1[T]  -γ_max[1,1,T] +γ_min[1,1,T]
-γ_max[2,1,T] +γ_min[2,1,T] -γ_max[3,1,T] +γ_min[3,1,T] -γ_max[4,1,T] +γ_min[4,1,T] -γ_max[5,1,T] +γ_min[5,1,T]
-γ_max[6,1,T] +γ_min[6,1,T] -γ_max[7,1,T] +γ_min[7,1,T] -γ_max[8,1,T] +γ_min[8,1,T] -γ_max[9,1,T] +γ_min[9,1,T]
-γ_max[10,1,T] +γ_min[10,1,T] -γ_max[11,1,T] +γ_min[11,1,T]>=0)             # dual constraints for UC, when t==T

@constraint(model, Oⁿˡ[1]  -K_g[2,k]*λ_F[n,T] -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[T] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[T] +Kˢᵗ[1]*σˢᵗˢᴳ²_2[T] -Kˢʰ[1]*σˢʰˢᴳ²_2[T]+ ψᵐᵃˣˢᴳ²_2[T] -γ_max[1,2,T] +γ_min[1,1,T]
-γ_max[12,1,T] +γ_min[12,1,T] -γ_max[13,1,T] +γ_min[13,1,T] -γ_max[14,1,T] +γ_min[14,1,T] -γ_max[15,1,T] +γ_min[15,1,T]
-γ_max[16,1,T] +γ_min[16,1,T] -γ_max[17,1,T] +γ_min[17,1,T] -γ_max[18,1,T] +γ_min[18,1,T] -γ_max[19,1,T] +γ_min[19,1,T]
-γ_max[20,1,T] +γ_min[20,1,T] -γ_max[21,1,T] +γ_min[21,1,T]>=0)

@constraint(model, Oⁿˡ[2]  -K_g[3,k]*λ_F[n,T] -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[T] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[T] +Kˢᵗ[2]*σˢᵗˢᴳ³_1[T] -Kˢʰ[2]*σˢʰˢᴳ³_1[T]+ ψᵐᵃˣˢᴳ³_1[T] -γ_max[2,2,T] +γ_min[2,1,T]
-γ_max[12,2,T] +γ_min[12,1,T] -γ_max[22,1,T] +γ_min[22,1,T] -γ_max[23,1,T] +γ_min[23,1,T] -γ_max[24,1,T] +γ_min[24,1,T]
-γ_max[25,1,T] +γ_min[25,1,T] -γ_max[26,1,T] +γ_min[26,1,T] -γ_max[27,1,T] +γ_min[27,1,T] -γ_max[28,1,T] +γ_min[28,1,T]
-γ_max[29,1,T] +γ_min[29,1,T] -γ_max[30,1,T] +γ_min[30,1,T]>=0)

@constraint(model, Oⁿˡ[2]  -K_g[4,k]*λ_F[n,T] -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[T] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[T] +Kˢᵗ[2]*σˢᵗˢᴳ³_2[T] -Kˢʰ[2]*σˢʰˢᴳ³_2[T]+ ψᵐᵃˣˢᴳ³_2[T] -γ_max[3,2,T] +γ_min[3,1,T]
-γ_max[13,2,T] +γ_min[13,1,T] -γ_max[22,2,T] +γ_min[22,1,T] -γ_max[31,1,T] +γ_min[31,1,T] -γ_max[32,1,T] +γ_min[32,1,T]
-γ_max[33,1,T] +γ_min[33,1,T] -γ_max[34,1,T] +γ_min[34,1,T] -γ_max[35,1,T] +γ_min[35,1,T] -γ_max[36,1,T] +γ_min[36,1,T]
-γ_max[37,1,T] +γ_min[37,1,T] -γ_max[38,1,T] +γ_min[38,1,T]>=0)

@constraint(model, Oⁿˡ[3]  -K_g[5,k]*λ_F[n,T] -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[T] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[T] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_1[T] -Kˢʰ[3]*σˢʰˢᴳ⁴_1[T]+ ψᵐᵃˣˢᴳ⁴_1[T] -γ_max[4,2,T] +γ_min[4,1,T]
-γ_max[14,2,T] +γ_min[14,1,T] -γ_max[23,2,T] +γ_min[23,1,T] -γ_max[31,2,T] +γ_min[31,1,T] -γ_max[39,1,T] +γ_min[39,1,T]
-γ_max[40,1,T] +γ_min[40,1,T] -γ_max[41,1,T] +γ_min[41,1,T] -γ_max[42,1,T] +γ_min[42,1,T] -γ_max[43,1,T] +γ_min[43,1,T]
-γ_max[44,1,T] +γ_min[44,1,T] -γ_max[45,1,T] +γ_min[45,1,T]>=0)


@constraint(model, Oⁿˡ[3]  -K_g[6,k]*λ_F[n,T] -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[T] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[T] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_2[T] -Kˢʰ[3]*σˢʰˢᴳ⁴_2[T]+ ψᵐᵃˣˢᴳ⁴_2[T] -γ_max[5,2,T] +γ_min[5,1,T]
-γ_max[15,2,T] +γ_min[15,1,T] -γ_max[24,2,T] +γ_min[24,1,T] -γ_max[32,2,T] +γ_min[32,1,T] -γ_max[39,2,T] +γ_min[39,1,T]
-γ_max[46,1,T] +γ_min[46,1,T] -γ_max[47,1,T] +γ_min[47,1,T] -γ_max[48,1,T] +γ_min[48,1,T] -γ_max[49,1,T] +γ_min[49,1,T]
-γ_max[50,1,T] +γ_min[50,1,T] -γ_max[51,1,T] +γ_min[51,1,T]>=0)


@constraint(model, Oⁿˡ[4]  -K_g[7,k]*λ_F[n,T] -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[T] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[T] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_1[T] -Kˢʰ[4]*σˢʰˢᴳ⁵_1[T]+ ψᵐᵃˣˢᴳ⁵_1[T] -γ_max[6,2,T] +γ_min[6,1,T]
-γ_max[16,2,T] +γ_min[16,1,T] -γ_max[25,2,T] +γ_min[25,1,T] -γ_max[33,2,T] +γ_min[33,1,T] -γ_max[40,2,T] +γ_min[40,1,T]
-γ_max[46,2,T] +γ_min[46,1,T] -γ_max[52,1,T] +γ_min[52,1,T] -γ_max[53,1,T] +γ_min[53,1,T] -γ_max[54,1,T] +γ_min[54,1,T]
-γ_max[55,1,T] +γ_min[55,1,T] -γ_max[56,1,T] +γ_min[56,1,T]>=0)


@constraint(model, Oⁿˡ[4]  -K_g[8,k]*λ_F[n,T] -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[T] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[T] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_2[T] -Kˢʰ[4]*σˢʰˢᴳ⁵_2[T]+ ψᵐᵃˣˢᴳ⁵_2[T] -γ_max[7,2,T] +γ_min[7,1,T]
-γ_max[17,2,T] +γ_min[17,1,T] -γ_max[26,2,T] +γ_min[26,1,T] -γ_max[34,2,T] +γ_min[34,1,T] -γ_max[41,2,T] +γ_min[41,1,T]
-γ_max[47,2,T] +γ_min[47,1,T] -γ_max[52,2,T] +γ_min[52,1,T] -γ_max[57,1,T] +γ_min[57,1,T] -γ_max[58,1,T] +γ_min[58,1,T]
-γ_max[59,1,T] +γ_min[59,1,T] -γ_max[60,1,T] +γ_min[60,1,T]>=0)


@constraint(model, Oⁿˡ[5]  -K_g[9,k]*λ_F[n,T] -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[T] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[T] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_1[T] -Kˢʰ[5]*σˢʰˢᴳ²⁷_1[T]+ ψᵐᵃˣˢᴳ²⁷_1[T] -γ_max[8,2,T] +γ_min[8,1,T]
-γ_max[18,2,T] +γ_min[18,1,T] -γ_max[27,2,T] +γ_min[27,1,T] -γ_max[35,2,T] +γ_min[35,1,T] -γ_max[42,2,T] +γ_min[42,1,T]
-γ_max[48,2,T] +γ_min[48,1,T] -γ_max[53,2,T] +γ_min[53,1,T] -γ_max[57,2,T] +γ_min[57,1,T] -γ_max[61,1,T] +γ_min[61,1,T]
-γ_max[62,1,T] +γ_min[62,1,T] -γ_max[63,1,T] +γ_min[63,1,T]>=0)


@constraint(model, Oⁿˡ[5]  -K_g[10,k]*λ_F[n,T] -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[T] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[T] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_2[T] -Kˢʰ[5]*σˢʰˢᴳ²⁷_2[T]+ ψᵐᵃˣˢᴳ²⁷_2[T] -γ_max[9,2,T] +γ_min[9,1,T]
-γ_max[19,2,T] +γ_min[19,1,T] -γ_max[28,2,T] +γ_min[28,1,T] -γ_max[36,2,T] +γ_min[36,1,T] -γ_max[43,2,T] +γ_min[43,1,T]
-γ_max[49,2,T] +γ_min[49,1,T] -γ_max[54,2,T] +γ_min[54,1,T] -γ_max[58,2,T] +γ_min[58,1,T] -γ_max[61,2,T] +γ_min[61,1,T]
-γ_max[64,1,T] +γ_min[64,1,T] -γ_max[65,1,T] +γ_min[65,1,T]>=0)


@constraint(model, Oⁿˡ[6]  -K_g[11,k]*λ_F[n,T] -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[T] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[T] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_1[T] -Kˢʰ[6]*σˢʰˢᴳ³⁰_1[T]+ ψᵐᵃˣˢᴳ³⁰_1[T] -γ_max[10,2,T] +γ_min[10,1,T]
-γ_max[20,2,T] +γ_min[20,1,T] -γ_max[29,2,T] +γ_min[29,1,T] -γ_max[37,2,T] +γ_min[37,1,T] -γ_max[44,2,T] +γ_min[44,1,T]
-γ_max[50,2,T] +γ_min[50,1,T] -γ_max[55,2,T] +γ_min[55,1,T] -γ_max[59,2,T] +γ_min[59,1,T] -γ_max[62,2,T] +γ_min[62,1,T]
-γ_max[64,2,T] +γ_min[64,1,T] -γ_max[66,1,T] +γ_min[66,1,T]>=0)


@constraint(model, Oⁿˡ[6]  -K_g[12,k]*λ_F[n,T] -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[T] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[T] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_2[T] -Kˢʰ[6]*σˢʰˢᴳ³⁰_2[T]+ ψᵐᵃˣˢᴳ³⁰_2[T] -γ_max[11,2,T] +γ_min[11,1,T]
-γ_max[21,2,T] +γ_min[21,1,T] -γ_max[30,2,T] +γ_min[30,1,T] -γ_max[38,2,T] +γ_min[38,1,T] -γ_max[45,2,T] +γ_min[45,1,T]
-γ_max[51,2,T] +γ_min[51,1,T] -γ_max[56,2,T] +γ_min[56,1,T] -γ_max[60,2,T] +γ_min[60,1,T] -γ_max[63,2,T] +γ_min[63,1,T]
-γ_max[65,2,T] +γ_min[65,1,T] -γ_max[66,2,T] +γ_min[66,1,T]>=0)



for t in 1:T-1                                                                                                                                                       # dual constraints for UC, when t<=T-1
    @constraint(model, Oⁿˡ[1]  -K_g[1,k]*λ_F[n,t] - Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[t]+ Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[t]+ Kˢᵗ[1]*(σˢᵗˢᴳ²_1[t]-σˢᵗˢᴳ²_1[t+1])+ Kˢʰ[1]*(σˢᵗˢᴳ²_1[t+1]-σˢᵗˢᴳ²_1[t])+ ψᵐᵃˣˢᴳ²_1[t] -γ_max[1,1,t] +γ_min[1,1,t]
    -γ_max[2,1,t] +γ_min[2,1,t] -γ_max[3,1,t] +γ_min[3,1,t] -γ_max[4,1,t] +γ_min[4,1,t] -γ_max[5,1,t] +γ_min[5,1,t]
    -γ_max[6,1,t] +γ_min[6,1,t] -γ_max[7,1,t] +γ_min[7,1,t] -γ_max[8,1,t] +γ_min[8,1,t] -γ_max[9,1,t] +γ_min[9,1,t]
    -γ_max[10,1,t] +γ_min[10,1,t] -γ_max[11,1,t] +γ_min[11,1,t]>=0)

    @constraint(model, Oⁿˡ[1]  -K_g[2,k]*λ_F[n,t] - Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[t]+ Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[t]+ Kˢᵗ[1]*(σˢᵗˢᴳ²_2[t]-σˢᵗˢᴳ²_2[t+1])+ Kˢʰ[1]*(σˢᵗˢᴳ²_2[t+1]-σˢᵗˢᴳ²_2[t])+ ψᵐᵃˣˢᴳ²_2[t] -γ_max[1,2,t] +γ_min[1,1,t]
    -γ_max[12,1,t] +γ_min[12,1,t] -γ_max[13,1,t] +γ_min[13,1,t] -γ_max[14,1,t] +γ_min[14,1,t] -γ_max[15,1,t] +γ_min[15,1,t]
    -γ_max[16,1,t] +γ_min[16,1,t] -γ_max[17,1,t] +γ_min[17,1,t] -γ_max[18,1,t] +γ_min[18,1,t] -γ_max[19,1,t] +γ_min[19,1,t]
    -γ_max[20,1,t] +γ_min[20,1,t] -γ_max[21,1,t] +γ_min[21,1,t]>=0)

    @constraint(model, Oⁿˡ[2]  -K_g[3,k]*λ_F[n,t] - Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[t]+ Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[t]+ Kˢᵗ[2]*(σˢᵗˢᴳ³_1[t]-σˢᵗˢᴳ³_1[t+1])+ Kˢʰ[2]*(σˢᵗˢᴳ³_1[t+1]-σˢᵗˢᴳ³_1[t])+ ψᵐᵃˣˢᴳ³_1[t] -γ_max[2,2,t] +γ_min[2,1,t]
    -γ_max[12,2,t] +γ_min[12,1,t] -γ_max[22,1,t] +γ_min[22,1,t] -γ_max[23,1,t] +γ_min[23,1,t] -γ_max[24,1,t] +γ_min[24,1,t]
    -γ_max[25,1,t] +γ_min[25,1,t] -γ_max[26,1,t] +γ_min[26,1,t] -γ_max[27,1,t] +γ_min[27,1,t] -γ_max[28,1,t] +γ_min[28,1,t]
    -γ_max[29,1,t] +γ_min[29,1,t] -γ_max[30,1,t] +γ_min[30,1,t]>=0)

    @constraint(model, Oⁿˡ[2]  -K_g[4,k]*λ_F[n,t] - Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[t]+ Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[t]+ Kˢᵗ[2]*(σˢᵗˢᴳ³_2[t]-σˢᵗˢᴳ³_2[t+1])+ Kˢʰ[2]*(σˢᵗˢᴳ³_2[t+1]-σˢᵗˢᴳ³_2[t])+ ψᵐᵃˣˢᴳ³_2[t] -γ_max[3,2,t] +γ_min[3,1,t]
    -γ_max[13,2,t] +γ_min[13,1,t] -γ_max[22,2,t] +γ_min[22,1,t] -γ_max[31,1,t] +γ_min[31,1,t] -γ_max[32,1,t] +γ_min[32,1,t]
    -γ_max[33,1,t] +γ_min[33,1,t] -γ_max[34,1,t] +γ_min[34,1,t] -γ_max[35,1,t] +γ_min[35,1,t] -γ_max[36,1,t] +γ_min[36,1,t]
    -γ_max[37,1,t] +γ_min[37,1,t] -γ_max[38,1,t] +γ_min[38,1,t]>=0)

    @constraint(model, Oⁿˡ[3]  -K_g[5,k]*λ_F[n,t] - Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[t]+ Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[t]+ Kˢᵗ[3]*(σˢᵗˢᴳ⁴_1[t]-σˢᵗˢᴳ⁴_1[t+1])+ Kˢʰ[3]*(σˢᵗˢᴳ⁴_1[t+1]-σˢᵗˢᴳ⁴_1[t])+ ψᵐᵃˣˢᴳ⁴_1[t] -γ_max[4,2,t] +γ_min[4,1,t]
    -γ_max[14,2,t] +γ_min[14,1,t] -γ_max[23,2,t] +γ_min[23,1,t] -γ_max[31,2,t] +γ_min[31,1,t] -γ_max[39,1,t] +γ_min[39,1,t]
    -γ_max[40,1,t] +γ_min[40,1,t] -γ_max[41,1,t] +γ_min[41,1,t] -γ_max[42,1,t] +γ_min[42,1,t] -γ_max[43,1,t] +γ_min[43,1,t]
    -γ_max[44,1,t] +γ_min[44,1,t] -γ_max[45,1,t] +γ_min[45,1,t]>=0)

    @constraint(model, Oⁿˡ[3]  -K_g[6,k]*λ_F[n,t] - Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[t]+ Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[t]+ Kˢᵗ[3]*(σˢᵗˢᴳ⁴_2[t]-σˢᵗˢᴳ⁴_2[t+1])+ Kˢʰ[3]*(σˢᵗˢᴳ⁴_2[t+1]-σˢᵗˢᴳ⁴_2[t])+ ψᵐᵃˣˢᴳ⁴_2[t]-γ_max[5,2,t] +γ_min[5,1,t]
    -γ_max[15,2,t] +γ_min[15,1,t] -γ_max[24,2,t] +γ_min[24,1,t] -γ_max[32,2,t] +γ_min[32,1,t] -γ_max[39,2,t] +γ_min[39,1,t]
    -γ_max[46,1,t] +γ_min[46,1,t] -γ_max[47,1,t] +γ_min[47,1,t] -γ_max[48,1,t] +γ_min[48,1,t] -γ_max[49,1,t] +γ_min[49,1,t]
    -γ_max[50,1,t] +γ_min[50,1,t] -γ_max[51,1,t] +γ_min[51,1,t]>=0)
    
    @constraint(model, Oⁿˡ[4]  -K_g[7,k]*λ_F[n,t] - Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[t]+ Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[t]+ Kˢᵗ[4]*(σˢᵗˢᴳ⁵_1[t]-σˢᵗˢᴳ⁵_1[t+1])+ Kˢʰ[4]*(σˢᵗˢᴳ⁵_1[t+1]-σˢᵗˢᴳ⁵_1[t])+ ψᵐᵃˣˢᴳ⁵_1[t] -γ_max[6,2,t] +γ_min[6,1,t]
    -γ_max[16,2,t] +γ_min[16,1,t] -γ_max[25,2,t] +γ_min[25,1,t] -γ_max[33,2,t] +γ_min[33,1,t] -γ_max[40,2,t] +γ_min[40,1,t]
    -γ_max[46,2,t] +γ_min[46,1,t] -γ_max[52,1,t] +γ_min[52,1,t] -γ_max[53,1,t] +γ_min[53,1,t] -γ_max[54,1,t] +γ_min[54,1,t]
    -γ_max[55,1,t] +γ_min[55,1,t] -γ_max[56,1,t] +γ_min[56,1,t]>=0)

    @constraint(model, Oⁿˡ[4]  -K_g[8,k]*λ_F[n,t] - Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[t]+ Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[t]+ Kˢᵗ[4]*(σˢᵗˢᴳ⁵_2[t]-σˢᵗˢᴳ⁵_2[t+1])+ Kˢʰ[4]*(σˢᵗˢᴳ⁵_2[t+1]-σˢᵗˢᴳ⁵_2[t])+ ψᵐᵃˣˢᴳ⁵_2[t] -γ_max[7,2,t] +γ_min[7,1,t]
    -γ_max[17,2,t] +γ_min[17,1,t] -γ_max[26,2,t] +γ_min[26,1,t] -γ_max[34,2,t] +γ_min[34,1,t] -γ_max[41,2,t] +γ_min[41,1,t]
    -γ_max[47,2,t] +γ_min[47,1,t] -γ_max[52,2,t] +γ_min[52,1,t] -γ_max[57,1,t] +γ_min[57,1,t] -γ_max[58,1,t] +γ_min[58,1,t]
    -γ_max[59,1,t] +γ_min[59,1,t] -γ_max[60,1,t] +γ_min[60,1,t]>=0)
    
    @constraint(model, Oⁿˡ[5]  -K_g[9,k]*λ_F[n,t] - Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[t]+ Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[t]+ Kˢᵗ[5]*(σˢᵗˢᴳ²⁷_1[t]-σˢᵗˢᴳ²⁷_1[t+1])+ Kˢʰ[5]*(σˢᵗˢᴳ²⁷_1[t+1]-σˢᵗˢᴳ²⁷_1[t])+ ψᵐᵃˣˢᴳ²⁷_1[t] -γ_max[8,2,t] +γ_min[8,1,t]
    -γ_max[18,2,t] +γ_min[18,1,t] -γ_max[27,2,t] +γ_min[27,1,t] -γ_max[35,2,t] +γ_min[35,1,t] -γ_max[42,2,t] +γ_min[42,1,t]
    -γ_max[48,2,t] +γ_min[48,1,t] -γ_max[53,2,t] +γ_min[53,1,t] -γ_max[57,2,t] +γ_min[57,1,t] -γ_max[61,1,t] +γ_min[61,1,t]
    -γ_max[62,1,t] +γ_min[62,1,t] -γ_max[63,1,t] +γ_min[63,1,t]>=0)

    @constraint(model, Oⁿˡ[5]  -K_g[10,k]*λ_F[n,t] - Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[t]+ Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[t]+ Kˢᵗ[5]*(σˢᵗˢᴳ²⁷_2[t]-σˢᵗˢᴳ²⁷_2[t+1])+ Kˢʰ[5]*(σˢᵗˢᴳ²⁷_2[t+1]-σˢᵗˢᴳ²⁷_2[t])+ ψᵐᵃˣˢᴳ²⁷_2[t] -γ_max[9,2,t] +γ_min[9,1,t]
    -γ_max[19,2,t] +γ_min[19,1,t] -γ_max[28,2,t] +γ_min[28,1,t] -γ_max[36,2,t] +γ_min[36,1,t] -γ_max[43,2,t] +γ_min[43,1,t]
    -γ_max[49,2,t] +γ_min[49,1,t] -γ_max[54,2,t] +γ_min[54,1,t] -γ_max[58,2,t] +γ_min[58,1,t] -γ_max[61,2,t] +γ_min[61,1,t]
    -γ_max[64,1,t] +γ_min[64,1,t] -γ_max[65,1,t] +γ_min[65,1,t]>=0)
    
    @constraint(model, Oⁿˡ[6]  -K_g[11,k]*λ_F[n,t] - Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[t]+ Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[t]+ Kˢᵗ[6]*(σˢᵗˢᴳ³⁰_1[t]-σˢᵗˢᴳ³⁰_1[t+1])+ Kˢʰ[6]*(σˢᵗˢᴳ³⁰_1[t+1]-σˢᵗˢᴳ³⁰_1[t])+ ψᵐᵃˣˢᴳ³⁰_1[t] -γ_max[10,2,t] +γ_min[10,1,t]
    -γ_max[20,2,t] +γ_min[20,1,t] -γ_max[29,2,t] +γ_min[29,1,t] -γ_max[37,2,t] +γ_min[37,1,t] -γ_max[44,2,t] +γ_min[44,1,t]
    -γ_max[50,2,t] +γ_min[50,1,t] -γ_max[55,2,t] +γ_min[55,1,t] -γ_max[59,2,t] +γ_min[59,1,t] -γ_max[62,2,t] +γ_min[62,1,t]
    -γ_max[64,2,t] +γ_min[64,1,t] -γ_max[66,1,t] +γ_min[66,1,t]>=0)    

    @constraint(model, Oⁿˡ[6]  -K_g[12,k]*λ_F[n,t] - Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[t]+ Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[t]+ Kˢᵗ[6]*(σˢᵗˢᴳ³⁰_2[t]-σˢᵗˢᴳ³⁰_2[t+1])+ Kˢʰ[6]*(σˢᵗˢᴳ³⁰_2[t+1]-σˢᵗˢᴳ³⁰_2[t])+ ψᵐᵃˣˢᴳ³⁰_2[t] -γ_max[11,2,t] +γ_min[11,1,t]
    -γ_max[21,2,t] +γ_min[21,1,t] -γ_max[30,2,t] +γ_min[30,1,t] -γ_max[38,2,t] +γ_min[38,1,t] -γ_max[45,2,t] +γ_min[45,1,t]
    -γ_max[51,2,t] +γ_min[51,1,t] -γ_max[56,2,t] +γ_min[56,1,t] -γ_max[60,2,t] +γ_min[60,1,t] -γ_max[63,2,t] +γ_min[63,1,t]
    -γ_max[65,2,t] +γ_min[65,1,t] -γ_max[66,2,t] +γ_min[66,1,t]>=0)  

end

n=4    # for bus 30
k=30
@constraint(model, Oⁿˡ[1]  -K_g[1,k]*λ_F[n,T] -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[T] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[T] +Kˢᵗ[1]*σˢᵗˢᴳ²_1[T] -Kˢʰ[1]*σˢʰˢᴳ²_1[T]+ ψᵐᵃˣˢᴳ²_1[T]  -γ_max[1,1,T] +γ_min[1,1,T]
-γ_max[2,1,T] +γ_min[2,1,T] -γ_max[3,1,T] +γ_min[3,1,T] -γ_max[4,1,T] +γ_min[4,1,T] -γ_max[5,1,T] +γ_min[5,1,T]
-γ_max[6,1,T] +γ_min[6,1,T] -γ_max[7,1,T] +γ_min[7,1,T] -γ_max[8,1,T] +γ_min[8,1,T] -γ_max[9,1,T] +γ_min[9,1,T]
-γ_max[10,1,T] +γ_min[10,1,T] -γ_max[11,1,T] +γ_min[11,1,T]>=0)             # dual constraints for UC, when t==T

@constraint(model, Oⁿˡ[1]  -K_g[2,k]*λ_F[n,T] -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[T] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[T] +Kˢᵗ[1]*σˢᵗˢᴳ²_2[T] -Kˢʰ[1]*σˢʰˢᴳ²_2[T]+ ψᵐᵃˣˢᴳ²_2[T] -γ_max[1,2,T] +γ_min[1,1,T]
-γ_max[12,1,T] +γ_min[12,1,T] -γ_max[13,1,T] +γ_min[13,1,T] -γ_max[14,1,T] +γ_min[14,1,T] -γ_max[15,1,T] +γ_min[15,1,T]
-γ_max[16,1,T] +γ_min[16,1,T] -γ_max[17,1,T] +γ_min[17,1,T] -γ_max[18,1,T] +γ_min[18,1,T] -γ_max[19,1,T] +γ_min[19,1,T]
-γ_max[20,1,T] +γ_min[20,1,T] -γ_max[21,1,T] +γ_min[21,1,T]>=0)

@constraint(model, Oⁿˡ[2]  -K_g[3,k]*λ_F[n,T] -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[T] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[T] +Kˢᵗ[2]*σˢᵗˢᴳ³_1[T] -Kˢʰ[2]*σˢʰˢᴳ³_1[T]+ ψᵐᵃˣˢᴳ³_1[T] -γ_max[2,2,T] +γ_min[2,1,T]
-γ_max[12,2,T] +γ_min[12,1,T] -γ_max[22,1,T] +γ_min[22,1,T] -γ_max[23,1,T] +γ_min[23,1,T] -γ_max[24,1,T] +γ_min[24,1,T]
-γ_max[25,1,T] +γ_min[25,1,T] -γ_max[26,1,T] +γ_min[26,1,T] -γ_max[27,1,T] +γ_min[27,1,T] -γ_max[28,1,T] +γ_min[28,1,T]
-γ_max[29,1,T] +γ_min[29,1,T] -γ_max[30,1,T] +γ_min[30,1,T]>=0)

@constraint(model, Oⁿˡ[2]  -K_g[4,k]*λ_F[n,T] -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[T] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[T] +Kˢᵗ[2]*σˢᵗˢᴳ³_2[T] -Kˢʰ[2]*σˢʰˢᴳ³_2[T]+ ψᵐᵃˣˢᴳ³_2[T] -γ_max[3,2,T] +γ_min[3,1,T]
-γ_max[13,2,T] +γ_min[13,1,T] -γ_max[22,2,T] +γ_min[22,1,T] -γ_max[31,1,T] +γ_min[31,1,T] -γ_max[32,1,T] +γ_min[32,1,T]
-γ_max[33,1,T] +γ_min[33,1,T] -γ_max[34,1,T] +γ_min[34,1,T] -γ_max[35,1,T] +γ_min[35,1,T] -γ_max[36,1,T] +γ_min[36,1,T]
-γ_max[37,1,T] +γ_min[37,1,T] -γ_max[38,1,T] +γ_min[38,1,T]>=0)

@constraint(model, Oⁿˡ[3]  -K_g[5,k]*λ_F[n,T] -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[T] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[T] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_1[T] -Kˢʰ[3]*σˢʰˢᴳ⁴_1[T]+ ψᵐᵃˣˢᴳ⁴_1[T] -γ_max[4,2,T] +γ_min[4,1,T]
-γ_max[14,2,T] +γ_min[14,1,T] -γ_max[23,2,T] +γ_min[23,1,T] -γ_max[31,2,T] +γ_min[31,1,T] -γ_max[39,1,T] +γ_min[39,1,T]
-γ_max[40,1,T] +γ_min[40,1,T] -γ_max[41,1,T] +γ_min[41,1,T] -γ_max[42,1,T] +γ_min[42,1,T] -γ_max[43,1,T] +γ_min[43,1,T]
-γ_max[44,1,T] +γ_min[44,1,T] -γ_max[45,1,T] +γ_min[45,1,T]>=0)


@constraint(model, Oⁿˡ[3]  -K_g[6,k]*λ_F[n,T] -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[T] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[T] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_2[T] -Kˢʰ[3]*σˢʰˢᴳ⁴_2[T]+ ψᵐᵃˣˢᴳ⁴_2[T] -γ_max[5,2,T] +γ_min[5,1,T]
-γ_max[15,2,T] +γ_min[15,1,T] -γ_max[24,2,T] +γ_min[24,1,T] -γ_max[32,2,T] +γ_min[32,1,T] -γ_max[39,2,T] +γ_min[39,1,T]
-γ_max[46,1,T] +γ_min[46,1,T] -γ_max[47,1,T] +γ_min[47,1,T] -γ_max[48,1,T] +γ_min[48,1,T] -γ_max[49,1,T] +γ_min[49,1,T]
-γ_max[50,1,T] +γ_min[50,1,T] -γ_max[51,1,T] +γ_min[51,1,T]>=0)


@constraint(model, Oⁿˡ[4]  -K_g[7,k]*λ_F[n,T] -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[T] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[T] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_1[T] -Kˢʰ[4]*σˢʰˢᴳ⁵_1[T]+ ψᵐᵃˣˢᴳ⁵_1[T] -γ_max[6,2,T] +γ_min[6,1,T]
-γ_max[16,2,T] +γ_min[16,1,T] -γ_max[25,2,T] +γ_min[25,1,T] -γ_max[33,2,T] +γ_min[33,1,T] -γ_max[40,2,T] +γ_min[40,1,T]
-γ_max[46,2,T] +γ_min[46,1,T] -γ_max[52,1,T] +γ_min[52,1,T] -γ_max[53,1,T] +γ_min[53,1,T] -γ_max[54,1,T] +γ_min[54,1,T]
-γ_max[55,1,T] +γ_min[55,1,T] -γ_max[56,1,T] +γ_min[56,1,T]>=0)


@constraint(model, Oⁿˡ[4]  -K_g[8,k]*λ_F[n,T] -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[T] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[T] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_2[T] -Kˢʰ[4]*σˢʰˢᴳ⁵_2[T]+ ψᵐᵃˣˢᴳ⁵_2[T] -γ_max[7,2,T] +γ_min[7,1,T]
-γ_max[17,2,T] +γ_min[17,1,T] -γ_max[26,2,T] +γ_min[26,1,T] -γ_max[34,2,T] +γ_min[34,1,T] -γ_max[41,2,T] +γ_min[41,1,T]
-γ_max[47,2,T] +γ_min[47,1,T] -γ_max[52,2,T] +γ_min[52,1,T] -γ_max[57,1,T] +γ_min[57,1,T] -γ_max[58,1,T] +γ_min[58,1,T]
-γ_max[59,1,T] +γ_min[59,1,T] -γ_max[60,1,T] +γ_min[60,1,T]>=0)


@constraint(model, Oⁿˡ[5]  -K_g[9,k]*λ_F[n,T] -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[T] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[T] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_1[T] -Kˢʰ[5]*σˢʰˢᴳ²⁷_1[T]+ ψᵐᵃˣˢᴳ²⁷_1[T] -γ_max[8,2,T] +γ_min[8,1,T]
-γ_max[18,2,T] +γ_min[18,1,T] -γ_max[27,2,T] +γ_min[27,1,T] -γ_max[35,2,T] +γ_min[35,1,T] -γ_max[42,2,T] +γ_min[42,1,T]
-γ_max[48,2,T] +γ_min[48,1,T] -γ_max[53,2,T] +γ_min[53,1,T] -γ_max[57,2,T] +γ_min[57,1,T] -γ_max[61,1,T] +γ_min[61,1,T]
-γ_max[62,1,T] +γ_min[62,1,T] -γ_max[63,1,T] +γ_min[63,1,T]>=0)


@constraint(model, Oⁿˡ[5]  -K_g[10,k]*λ_F[n,T] -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[T] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[T] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_2[T] -Kˢʰ[5]*σˢʰˢᴳ²⁷_2[T]+ ψᵐᵃˣˢᴳ²⁷_2[T] -γ_max[9,2,T] +γ_min[9,1,T]
-γ_max[19,2,T] +γ_min[19,1,T] -γ_max[28,2,T] +γ_min[28,1,T] -γ_max[36,2,T] +γ_min[36,1,T] -γ_max[43,2,T] +γ_min[43,1,T]
-γ_max[49,2,T] +γ_min[49,1,T] -γ_max[54,2,T] +γ_min[54,1,T] -γ_max[58,2,T] +γ_min[58,1,T] -γ_max[61,2,T] +γ_min[61,1,T]
-γ_max[64,1,T] +γ_min[64,1,T] -γ_max[65,1,T] +γ_min[65,1,T]>=0)


@constraint(model, Oⁿˡ[6]  -K_g[11,k]*λ_F[n,T] -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[T] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[T] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_1[T] -Kˢʰ[6]*σˢʰˢᴳ³⁰_1[T]+ ψᵐᵃˣˢᴳ³⁰_1[T] -γ_max[10,2,T] +γ_min[10,1,T]
-γ_max[20,2,T] +γ_min[20,1,T] -γ_max[29,2,T] +γ_min[29,1,T] -γ_max[37,2,T] +γ_min[37,1,T] -γ_max[44,2,T] +γ_min[44,1,T]
-γ_max[50,2,T] +γ_min[50,1,T] -γ_max[55,2,T] +γ_min[55,1,T] -γ_max[59,2,T] +γ_min[59,1,T] -γ_max[62,2,T] +γ_min[62,1,T]
-γ_max[64,2,T] +γ_min[64,1,T] -γ_max[66,1,T] +γ_min[66,1,T]>=0)


@constraint(model, Oⁿˡ[6]  -K_g[12,k]*λ_F[n,T] -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[T] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[T] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_2[T] -Kˢʰ[6]*σˢʰˢᴳ³⁰_2[T]+ ψᵐᵃˣˢᴳ³⁰_2[T] -γ_max[11,2,T] +γ_min[11,1,T]
-γ_max[21,2,T] +γ_min[21,1,T] -γ_max[30,2,T] +γ_min[30,1,T] -γ_max[38,2,T] +γ_min[38,1,T] -γ_max[45,2,T] +γ_min[45,1,T]
-γ_max[51,2,T] +γ_min[51,1,T] -γ_max[56,2,T] +γ_min[56,1,T] -γ_max[60,2,T] +γ_min[60,1,T] -γ_max[63,2,T] +γ_min[63,1,T]
-γ_max[65,2,T] +γ_min[65,1,T] -γ_max[66,2,T] +γ_min[66,1,T]>=0)



for t in 1:T-1                                                                                                                                                       # dual constraints for UC, when t<=T-1
    @constraint(model, Oⁿˡ[1]  -K_g[1,k]*λ_F[n,t] - Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[t]+ Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[t]+ Kˢᵗ[1]*(σˢᵗˢᴳ²_1[t]-σˢᵗˢᴳ²_1[t+1])+ Kˢʰ[1]*(σˢᵗˢᴳ²_1[t+1]-σˢᵗˢᴳ²_1[t])+ ψᵐᵃˣˢᴳ²_1[t] -γ_max[1,1,t] +γ_min[1,1,t]
    -γ_max[2,1,t] +γ_min[2,1,t] -γ_max[3,1,t] +γ_min[3,1,t] -γ_max[4,1,t] +γ_min[4,1,t] -γ_max[5,1,t] +γ_min[5,1,t]
    -γ_max[6,1,t] +γ_min[6,1,t] -γ_max[7,1,t] +γ_min[7,1,t] -γ_max[8,1,t] +γ_min[8,1,t] -γ_max[9,1,t] +γ_min[9,1,t]
    -γ_max[10,1,t] +γ_min[10,1,t] -γ_max[11,1,t] +γ_min[11,1,t]>=0)

    @constraint(model, Oⁿˡ[1]  -K_g[2,k]*λ_F[n,t] - Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[t]+ Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[t]+ Kˢᵗ[1]*(σˢᵗˢᴳ²_2[t]-σˢᵗˢᴳ²_2[t+1])+ Kˢʰ[1]*(σˢᵗˢᴳ²_2[t+1]-σˢᵗˢᴳ²_2[t])+ ψᵐᵃˣˢᴳ²_2[t] -γ_max[1,2,t] +γ_min[1,1,t]
    -γ_max[12,1,t] +γ_min[12,1,t] -γ_max[13,1,t] +γ_min[13,1,t] -γ_max[14,1,t] +γ_min[14,1,t] -γ_max[15,1,t] +γ_min[15,1,t]
    -γ_max[16,1,t] +γ_min[16,1,t] -γ_max[17,1,t] +γ_min[17,1,t] -γ_max[18,1,t] +γ_min[18,1,t] -γ_max[19,1,t] +γ_min[19,1,t]
    -γ_max[20,1,t] +γ_min[20,1,t] -γ_max[21,1,t] +γ_min[21,1,t]>=0)

    @constraint(model, Oⁿˡ[2]  -K_g[3,k]*λ_F[n,t] - Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[t]+ Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[t]+ Kˢᵗ[2]*(σˢᵗˢᴳ³_1[t]-σˢᵗˢᴳ³_1[t+1])+ Kˢʰ[2]*(σˢᵗˢᴳ³_1[t+1]-σˢᵗˢᴳ³_1[t])+ ψᵐᵃˣˢᴳ³_1[t] -γ_max[2,2,t] +γ_min[2,1,t]
    -γ_max[12,2,t] +γ_min[12,1,t] -γ_max[22,1,t] +γ_min[22,1,t] -γ_max[23,1,t] +γ_min[23,1,t] -γ_max[24,1,t] +γ_min[24,1,t]
    -γ_max[25,1,t] +γ_min[25,1,t] -γ_max[26,1,t] +γ_min[26,1,t] -γ_max[27,1,t] +γ_min[27,1,t] -γ_max[28,1,t] +γ_min[28,1,t]
    -γ_max[29,1,t] +γ_min[29,1,t] -γ_max[30,1,t] +γ_min[30,1,t]>=0)

    @constraint(model, Oⁿˡ[2]  -K_g[4,k]*λ_F[n,t] - Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[t]+ Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[t]+ Kˢᵗ[2]*(σˢᵗˢᴳ³_2[t]-σˢᵗˢᴳ³_2[t+1])+ Kˢʰ[2]*(σˢᵗˢᴳ³_2[t+1]-σˢᵗˢᴳ³_2[t])+ ψᵐᵃˣˢᴳ³_2[t] -γ_max[3,2,t] +γ_min[3,1,t]
    -γ_max[13,2,t] +γ_min[13,1,t] -γ_max[22,2,t] +γ_min[22,1,t] -γ_max[31,1,t] +γ_min[31,1,t] -γ_max[32,1,t] +γ_min[32,1,t]
    -γ_max[33,1,t] +γ_min[33,1,t] -γ_max[34,1,t] +γ_min[34,1,t] -γ_max[35,1,t] +γ_min[35,1,t] -γ_max[36,1,t] +γ_min[36,1,t]
    -γ_max[37,1,t] +γ_min[37,1,t] -γ_max[38,1,t] +γ_min[38,1,t]>=0)

    @constraint(model, Oⁿˡ[3]  -K_g[5,k]*λ_F[n,t] - Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[t]+ Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[t]+ Kˢᵗ[3]*(σˢᵗˢᴳ⁴_1[t]-σˢᵗˢᴳ⁴_1[t+1])+ Kˢʰ[3]*(σˢᵗˢᴳ⁴_1[t+1]-σˢᵗˢᴳ⁴_1[t])+ ψᵐᵃˣˢᴳ⁴_1[t] -γ_max[4,2,t] +γ_min[4,1,t]
    -γ_max[14,2,t] +γ_min[14,1,t] -γ_max[23,2,t] +γ_min[23,1,t] -γ_max[31,2,t] +γ_min[31,1,t] -γ_max[39,1,t] +γ_min[39,1,t]
    -γ_max[40,1,t] +γ_min[40,1,t] -γ_max[41,1,t] +γ_min[41,1,t] -γ_max[42,1,t] +γ_min[42,1,t] -γ_max[43,1,t] +γ_min[43,1,t]
    -γ_max[44,1,t] +γ_min[44,1,t] -γ_max[45,1,t] +γ_min[45,1,t]>=0)

    @constraint(model, Oⁿˡ[3]  -K_g[6,k]*λ_F[n,t] - Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[t]+ Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[t]+ Kˢᵗ[3]*(σˢᵗˢᴳ⁴_2[t]-σˢᵗˢᴳ⁴_2[t+1])+ Kˢʰ[3]*(σˢᵗˢᴳ⁴_2[t+1]-σˢᵗˢᴳ⁴_2[t])+ ψᵐᵃˣˢᴳ⁴_2[t]-γ_max[5,2,t] +γ_min[5,1,t]
    -γ_max[15,2,t] +γ_min[15,1,t] -γ_max[24,2,t] +γ_min[24,1,t] -γ_max[32,2,t] +γ_min[32,1,t] -γ_max[39,2,t] +γ_min[39,1,t]
    -γ_max[46,1,t] +γ_min[46,1,t] -γ_max[47,1,t] +γ_min[47,1,t] -γ_max[48,1,t] +γ_min[48,1,t] -γ_max[49,1,t] +γ_min[49,1,t]
    -γ_max[50,1,t] +γ_min[50,1,t] -γ_max[51,1,t] +γ_min[51,1,t]>=0)
    
    @constraint(model, Oⁿˡ[4]  -K_g[7,k]*λ_F[n,t] - Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[t]+ Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[t]+ Kˢᵗ[4]*(σˢᵗˢᴳ⁵_1[t]-σˢᵗˢᴳ⁵_1[t+1])+ Kˢʰ[4]*(σˢᵗˢᴳ⁵_1[t+1]-σˢᵗˢᴳ⁵_1[t])+ ψᵐᵃˣˢᴳ⁵_1[t] -γ_max[6,2,t] +γ_min[6,1,t]
    -γ_max[16,2,t] +γ_min[16,1,t] -γ_max[25,2,t] +γ_min[25,1,t] -γ_max[33,2,t] +γ_min[33,1,t] -γ_max[40,2,t] +γ_min[40,1,t]
    -γ_max[46,2,t] +γ_min[46,1,t] -γ_max[52,1,t] +γ_min[52,1,t] -γ_max[53,1,t] +γ_min[53,1,t] -γ_max[54,1,t] +γ_min[54,1,t]
    -γ_max[55,1,t] +γ_min[55,1,t] -γ_max[56,1,t] +γ_min[56,1,t]>=0)

    @constraint(model, Oⁿˡ[4]  -K_g[8,k]*λ_F[n,t] - Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[t]+ Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[t]+ Kˢᵗ[4]*(σˢᵗˢᴳ⁵_2[t]-σˢᵗˢᴳ⁵_2[t+1])+ Kˢʰ[4]*(σˢᵗˢᴳ⁵_2[t+1]-σˢᵗˢᴳ⁵_2[t])+ ψᵐᵃˣˢᴳ⁵_2[t] -γ_max[7,2,t] +γ_min[7,1,t]
    -γ_max[17,2,t] +γ_min[17,1,t] -γ_max[26,2,t] +γ_min[26,1,t] -γ_max[34,2,t] +γ_min[34,1,t] -γ_max[41,2,t] +γ_min[41,1,t]
    -γ_max[47,2,t] +γ_min[47,1,t] -γ_max[52,2,t] +γ_min[52,1,t] -γ_max[57,1,t] +γ_min[57,1,t] -γ_max[58,1,t] +γ_min[58,1,t]
    -γ_max[59,1,t] +γ_min[59,1,t] -γ_max[60,1,t] +γ_min[60,1,t]>=0)
    
    @constraint(model, Oⁿˡ[5]  -K_g[9,k]*λ_F[n,t] - Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[t]+ Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[t]+ Kˢᵗ[5]*(σˢᵗˢᴳ²⁷_1[t]-σˢᵗˢᴳ²⁷_1[t+1])+ Kˢʰ[5]*(σˢᵗˢᴳ²⁷_1[t+1]-σˢᵗˢᴳ²⁷_1[t])+ ψᵐᵃˣˢᴳ²⁷_1[t] -γ_max[8,2,t] +γ_min[8,1,t]
    -γ_max[18,2,t] +γ_min[18,1,t] -γ_max[27,2,t] +γ_min[27,1,t] -γ_max[35,2,t] +γ_min[35,1,t] -γ_max[42,2,t] +γ_min[42,1,t]
    -γ_max[48,2,t] +γ_min[48,1,t] -γ_max[53,2,t] +γ_min[53,1,t] -γ_max[57,2,t] +γ_min[57,1,t] -γ_max[61,1,t] +γ_min[61,1,t]
    -γ_max[62,1,t] +γ_min[62,1,t] -γ_max[63,1,t] +γ_min[63,1,t]>=0)

    @constraint(model, Oⁿˡ[5]  -K_g[10,k]*λ_F[n,t] - Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[t]+ Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[t]+ Kˢᵗ[5]*(σˢᵗˢᴳ²⁷_2[t]-σˢᵗˢᴳ²⁷_2[t+1])+ Kˢʰ[5]*(σˢᵗˢᴳ²⁷_2[t+1]-σˢᵗˢᴳ²⁷_2[t])+ ψᵐᵃˣˢᴳ²⁷_2[t] -γ_max[9,2,t] +γ_min[9,1,t]
    -γ_max[19,2,t] +γ_min[19,1,t] -γ_max[28,2,t] +γ_min[28,1,t] -γ_max[36,2,t] +γ_min[36,1,t] -γ_max[43,2,t] +γ_min[43,1,t]
    -γ_max[49,2,t] +γ_min[49,1,t] -γ_max[54,2,t] +γ_min[54,1,t] -γ_max[58,2,t] +γ_min[58,1,t] -γ_max[61,2,t] +γ_min[61,1,t]
    -γ_max[64,1,t] +γ_min[64,1,t] -γ_max[65,1,t] +γ_min[65,1,t]>=0)
    
    @constraint(model, Oⁿˡ[6]  -K_g[11,k]*λ_F[n,t] - Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[t]+ Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[t]+ Kˢᵗ[6]*(σˢᵗˢᴳ³⁰_1[t]-σˢᵗˢᴳ³⁰_1[t+1])+ Kˢʰ[6]*(σˢᵗˢᴳ³⁰_1[t+1]-σˢᵗˢᴳ³⁰_1[t])+ ψᵐᵃˣˢᴳ³⁰_1[t] -γ_max[10,2,t] +γ_min[10,1,t]
    -γ_max[20,2,t] +γ_min[20,1,t] -γ_max[29,2,t] +γ_min[29,1,t] -γ_max[37,2,t] +γ_min[37,1,t] -γ_max[44,2,t] +γ_min[44,1,t]
    -γ_max[50,2,t] +γ_min[50,1,t] -γ_max[55,2,t] +γ_min[55,1,t] -γ_max[59,2,t] +γ_min[59,1,t] -γ_max[62,2,t] +γ_min[62,1,t]
    -γ_max[64,2,t] +γ_min[64,1,t] -γ_max[66,1,t] +γ_min[66,1,t]>=0)    

    @constraint(model, Oⁿˡ[6]  -K_g[12,k]*λ_F[n,t] - Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[t]+ Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[t]+ Kˢᵗ[6]*(σˢᵗˢᴳ³⁰_2[t]-σˢᵗˢᴳ³⁰_2[t+1])+ Kˢʰ[6]*(σˢᵗˢᴳ³⁰_2[t+1]-σˢᵗˢᴳ³⁰_2[t])+ ψᵐᵃˣˢᴳ³⁰_2[t] -γ_max[11,2,t] +γ_min[11,1,t]
    -γ_max[21,2,t] +γ_min[21,1,t] -γ_max[30,2,t] +γ_min[30,1,t] -γ_max[38,2,t] +γ_min[38,1,t] -γ_max[45,2,t] +γ_min[45,1,t]
    -γ_max[51,2,t] +γ_min[51,1,t] -γ_max[56,2,t] +γ_min[56,1,t] -γ_max[60,2,t] +γ_min[60,1,t] -γ_max[63,2,t] +γ_min[63,1,t]
    -γ_max[65,2,t] +γ_min[65,1,t] -γ_max[66,2,t] +γ_min[66,1,t]>=0)  

end


k=11
n=1
for t in 1:T
    for i in 1:66
    @constraint(model,  0>= K_m[i,k]*λ_F[n,t] -γ_max[i,1,t] -γ_max[i,2,t] +γ_min[i,1,t] )
    end
end


k=26
n=2
for t in 1:T
    for i in 1:66
    @constraint(model,  0>= K_m[i,k]*λ_F[n,t] -γ_max[i,1,t] -γ_max[i,2,t] +γ_min[i,1,t] )
    end
end

k=29
n=3
for t in 1:T
    for i in 1:66
    @constraint(model,  0>= K_m[i,k]*λ_F[n,t] -γ_max[i,1,t] -γ_max[i,2,t] +γ_min[i,1,t] )
    end
end

k=30
n=4
for t in 1:T
    for i in 1:66
    @constraint(model,  0>= K_m[i,k]*λ_F[n,t] -γ_max[i,1,t] -γ_max[i,2,t] +γ_min[i,1,t] )
    end
end

@constraint(model, Oᵐ₁[1] -λᴱ[T] +μᵐᵃˣˢᴳ²_1[T] -μᵐⁱⁿˢᴳ²_1[T] +πʳᵘˢᴳ²_1[T] -πʳᵈˢᴳ²_1[T]>=0)          # dual constraints for generation, when t==T, assume SGs in bus 2 are strategic
@constraint(model, Oᵐ₂[1] -λᴱ[T] +μᵐᵃˣˢᴳ²_2[T] -μᵐⁱⁿˢᴳ²_2[T] +πʳᵘˢᴳ²_2[T] -πʳᵈˢᴳ²_2[T]>=0)
@constraint(model, Oᵐ₁[2] -λᴱ[T] +μᵐᵃˣˢᴳ³_1[T] -μᵐⁱⁿˢᴳ³_1[T] +πʳᵘˢᴳ³_1[T] -πʳᵈˢᴳ³_1[T]>=0)
@constraint(model, Oᵐ₂[2] -λᴱ[T] +μᵐᵃˣˢᴳ³_2[T] -μᵐⁱⁿˢᴳ³_2[T] +πʳᵘˢᴳ³_2[T] -πʳᵈˢᴳ³_2[T]>=0)
@constraint(model, Oᵐ₁[3] -λᴱ[T] +μᵐᵃˣˢᴳ⁴_1[T] -μᵐⁱⁿˢᴳ⁴_1[T] +πʳᵘˢᴳ⁴_1[T] -πʳᵈˢᴳ⁴_1[T]>=0)
@constraint(model, Oᵐ₂[3] -λᴱ[T] +μᵐᵃˣˢᴳ⁴_2[T] -μᵐⁱⁿˢᴳ⁴_2[T] +πʳᵘˢᴳ⁴_2[T] -πʳᵈˢᴳ⁴_2[T]>=0)
@constraint(model, Oᵐ₁[4] -λᴱ[T] +μᵐᵃˣˢᴳ⁵_1[T] -μᵐⁱⁿˢᴳ⁵_1[T] +πʳᵘˢᴳ⁵_1[T] -πʳᵈˢᴳ⁵_1[T]>=0)
@constraint(model, Oᵐ₂[4] -λᴱ[T] +μᵐᵃˣˢᴳ⁵_2[T] -μᵐⁱⁿˢᴳ⁵_2[T] +πʳᵘˢᴳ⁵_2[T] -πʳᵈˢᴳ⁵_2[T]>=0)
@constraint(model, Oᵐ₁[5] -λᴱ[T] +μᵐᵃˣˢᴳ²⁷_1[T] -μᵐⁱⁿˢᴳ²⁷_1[T] +πʳᵘˢᴳ²⁷_1[T] -πʳᵈˢᴳ²⁷_1[T]>=0)
@constraint(model, Oᵐ₂[5] -λᴱ[T] +μᵐᵃˣˢᴳ²⁷_2[T] -μᵐⁱⁿˢᴳ²⁷_2[T] +πʳᵘˢᴳ²⁷_2[T] -πʳᵈˢᴳ²⁷_2[T]>=0)
@constraint(model, Oᵐ₁[6] -λᴱ[T] +μᵐᵃˣˢᴳ³⁰_1[T] -μᵐⁱⁿˢᴳ³⁰_1[T] +πʳᵘˢᴳ³⁰_1[T] -πʳᵈˢᴳ³⁰_1[T]>=0)
@constraint(model, Oᵐ₂[6] -λᴱ[T] +μᵐᵃˣˢᴳ³⁰_2[T] -μᵐⁱⁿˢᴳ³⁰_2[T] +πʳᵘˢᴳ³⁰_2[T] -πʳᵈˢᴳ³⁰_2[T]>=0)
for t in 1:T-1                                                                                                  # dual constraints for generation, when t<=T-1, assume SGs in bus 2 are strategic
    @constraint(model, Oᵐ₁[1] -λᴱ[t] +μᵐᵃˣˢᴳ²_1[t] -μᵐⁱⁿˢᴳ²_1[t] +πʳᵘˢᴳ²_1[t] -πʳᵘˢᴳ²_1[t+1] -πʳᵈˢᴳ²_1[t] +πʳᵈˢᴳ²_1[t+1]>=0)                
    @constraint(model, Oᵐ₂[1] -λᴱ[t] +μᵐᵃˣˢᴳ²_2[t] -μᵐⁱⁿˢᴳ²_2[t] +πʳᵘˢᴳ²_2[t] -πʳᵘˢᴳ²_2[t+1] -πʳᵈˢᴳ²_2[t] +πʳᵈˢᴳ²_2[t+1]>=0)
    @constraint(model, Oᵐ₁[2] -λᴱ[t] +μᵐᵃˣˢᴳ³_1[t] -μᵐⁱⁿˢᴳ³_1[t] +πʳᵘˢᴳ³_1[t] -πʳᵘˢᴳ³_1[t+1] -πʳᵈˢᴳ³_1[t] +πʳᵈˢᴳ³_1[t+1]>=0)                
    @constraint(model, Oᵐ₂[2] -λᴱ[t] +μᵐᵃˣˢᴳ³_2[t] -μᵐⁱⁿˢᴳ³_2[t] +πʳᵘˢᴳ³_2[t] -πʳᵘˢᴳ³_2[t+1] -πʳᵈˢᴳ³_2[t] +πʳᵈˢᴳ³_2[t+1]>=0)
    @constraint(model, Oᵐ₁[3] -λᴱ[t] +μᵐᵃˣˢᴳ⁴_1[t] -μᵐⁱⁿˢᴳ⁴_1[t] +πʳᵘˢᴳ⁴_1[t] -πʳᵘˢᴳ⁴_1[t+1] -πʳᵈˢᴳ⁴_1[t] +πʳᵈˢᴳ⁴_1[t+1]>=0)
    @constraint(model, Oᵐ₂[3] -λᴱ[t] +μᵐᵃˣˢᴳ⁴_2[t] -μᵐⁱⁿˢᴳ⁴_2[t] +πʳᵘˢᴳ⁴_2[t] -πʳᵘˢᴳ⁴_2[t+1] -πʳᵈˢᴳ⁴_2[t] +πʳᵈˢᴳ⁴_2[t+1]>=0)
    @constraint(model, Oᵐ₁[4] -λᴱ[t] +μᵐᵃˣˢᴳ⁵_1[t] -μᵐⁱⁿˢᴳ⁵_1[t] +πʳᵘˢᴳ⁵_1[t] -πʳᵘˢᴳ⁵_1[t+1] -πʳᵈˢᴳ⁵_1[t] +πʳᵈˢᴳ⁵_1[t+1]>=0)
    @constraint(model, Oᵐ₂[4] -λᴱ[t] +μᵐᵃˣˢᴳ⁵_2[t] -μᵐⁱⁿˢᴳ⁵_2[t] +πʳᵘˢᴳ⁵_2[t] -πʳᵘˢᴳ⁵_2[t+1] -πʳᵈˢᴳ⁵_2[t] +πʳᵈˢᴳ⁵_2[t+1]>=0)
    @constraint(model, Oᵐ₁[5] -λᴱ[t] +μᵐᵃˣˢᴳ²⁷_1[t] -μᵐⁱⁿˢᴳ²⁷_1[t] +πʳᵘˢᴳ²⁷_1[t] -πʳᵘˢᴳ²⁷_1[t+1] -πʳᵈˢᴳ²⁷_1[t] +πʳᵈˢᴳ²⁷_1[t+1]>=0)
    @constraint(model, Oᵐ₂[5] -λᴱ[t] +μᵐᵃˣˢᴳ²⁷_2[t] -μᵐⁱⁿˢᴳ²⁷_2[t] +πʳᵘˢᴳ²⁷_2[t] -πʳᵘˢᴳ²⁷_2[t+1] -πʳᵈˢᴳ²⁷_2[t] +πʳᵈˢᴳ²⁷_2[t+1]>=0)
    @constraint(model, Oᵐ₁[6] -λᴱ[t] +μᵐᵃˣˢᴳ³⁰_1[t] -μᵐⁱⁿˢᴳ³⁰_1[t] +πʳᵘˢᴳ³⁰_1[t] -πʳᵘˢᴳ³⁰_1[t+1] -πʳᵈˢᴳ³⁰_1[t] +πʳᵈˢᴳ³⁰_1[t+1]>=0)
    @constraint(model, Oᵐ₂[6] -λᴱ[t] +μᵐᵃˣˢᴳ³⁰_2[t] -μᵐⁱⁿˢᴳ³⁰_2[t] +πʳᵘˢᴳ³⁰_2[t] -πʳᵘˢᴳ³⁰_2[t+1] -πʳᵈˢᴳ³⁰_2[t] +πʳᵈˢᴳ³⁰_2[t+1]>=0)                
end



@constraint(model,σˢᵗˢᴳ²_1.<=1)                                 # dual constraints for on/off costs
@constraint(model,σˢᵗˢᴳ²_2.<=1)
@constraint(model,σˢᵗˢᴳ³_1.<=1)
@constraint(model,σˢᵗˢᴳ³_2.<=1)
@constraint(model,σˢᵗˢᴳ⁴_1.<=1)
@constraint(model,σˢᵗˢᴳ⁴_2.<=1)
@constraint(model,σˢᵗˢᴳ⁵_1.<=1)
@constraint(model,σˢᵗˢᴳ⁵_2.<=1)
@constraint(model,σˢᵗˢᴳ²⁷_1.<=1)
@constraint(model,σˢᵗˢᴳ²⁷_2.<=1)
@constraint(model,σˢᵗˢᴳ³⁰_1.<=1)
@constraint(model,σˢᵗˢᴳ³⁰_2.<=1)
@constraint(model,σˢʰˢᴳ²_1.<=1)
@constraint(model,σˢʰˢᴳ²_2.<=1)
@constraint(model,σˢʰˢᴳ³_1.<=1)
@constraint(model,σˢʰˢᴳ³_2.<=1)
@constraint(model,σˢʰˢᴳ⁴_1.<=1)
@constraint(model,σˢʰˢᴳ⁴_2.<=1)
@constraint(model,σˢʰˢᴳ⁵_1.<=1)
@constraint(model,σˢʰˢᴳ⁵_2.<=1)
@constraint(model,σˢʰˢᴳ²⁷_1.<=1)
@constraint(model,σˢʰˢᴳ²⁷_2.<=1)
@constraint(model,σˢʰˢᴳ³⁰_1.<=1)
@constraint(model,σˢʰˢᴳ³⁰_2.<=1)

for t in 1:T                                            # dual constraints for generation of IBRs
    @constraint(model, Oᴱ_c[1] -λᴱ[t] +ζᵐᵃˣ¹[t]>=0)
    @constraint(model, Oᴱ_c[2] -λᴱ[t] +ζᵐᵃˣ²³[t]>=0)
    @constraint(model, Oᴱ_c[3] -λᴱ[t] +ζᵐᵃˣ²⁶[t]>=0)
end

k=11
for t in 1:T                                            # dual constraints for online capacity factor of IBRs
    @constraint(model, φᵐᵃˣ¹[t]   -K_c[1,k]*λ_F[1,t] -IBG₁[t]*ζᵐᵃˣ¹[t]>=0)
    @constraint(model, φᵐᵃˣ²³[t]  -K_c[2,k]*λ_F[1,t] -IBG₂₃[t]*ζᵐᵃˣ²³[t]>=0)
    @constraint(model, φᵐᵃˣ²⁶[t]  -K_c[3,k]*λ_F[1,t] -IBG₂₆[t]*ζᵐᵃˣ²⁶[t]>=0)
end

k=26
for t in 1:T                                            # dual constraints for online capacity factor of IBRs
    @constraint(model, φᵐᵃˣ¹[t]   -K_c[1,k]*λ_F[2,t] -IBG₁[t]*ζᵐᵃˣ¹[t]>=0)
    @constraint(model, φᵐᵃˣ²³[t]  -K_c[2,k]*λ_F[2,t] -IBG₂₃[t]*ζᵐᵃˣ²³[t]>=0)
    @constraint(model, φᵐᵃˣ²⁶[t]  -K_c[3,k]*λ_F[2,t] -IBG₂₆[t]*ζᵐᵃˣ²⁶[t]>=0)
end

k=29
for t in 1:T                                            # dual constraints for online capacity factor of IBRs
    @constraint(model, φᵐᵃˣ¹[t]   -K_c[1,k]*λ_F[3,t] -IBG₁[t]*ζᵐᵃˣ¹[t]>=0)
    @constraint(model, φᵐᵃˣ²³[t]  -K_c[2,k]*λ_F[3,t] -IBG₂₃[t]*ζᵐᵃˣ²³[t]>=0)
    @constraint(model, φᵐᵃˣ²⁶[t]  -K_c[3,k]*λ_F[3,t] -IBG₂₆[t]*ζᵐᵃˣ²⁶[t]>=0)
end

k=30
for t in 1:T                                            # dual constraints for online capacity factor of IBRs
    @constraint(model, φᵐᵃˣ¹[t]   -K_c[1,k]*λ_F[4,t] -IBG₁[t]*ζᵐᵃˣ¹[t]>=0)
    @constraint(model, φᵐᵃˣ²³[t]  -K_c[2,k]*λ_F[4,t] -IBG₂₃[t]*ζᵐᵃˣ²³[t]>=0)
    @constraint(model, φᵐᵃˣ²⁶[t]  -K_c[3,k]*λ_F[4,t] -IBG₂₆[t]*ζᵐᵃˣ²⁶[t]>=0)
end



#-------Define Objective Functions 
#-Primal obj
cost_onoff_Primal=sum(Cᵁ²_1)+sum(Cᴰ²_1)+sum(Cᵁ³_1)+sum(Cᴰ³_1)+sum(Cᵁ⁴_1)+sum(Cᴰ⁴_1)+sum(Cᵁ⁵_1)+sum(Cᴰ⁵_1)+sum(Cᵁ²⁷_1)+sum(Cᴰ²⁷_1)+sum(Cᵁ³⁰_1)+sum(Cᴰ³⁰_1)  +sum(Cᵁ²_2)+sum(Cᴰ²_2)+sum(Cᵁ³_2)+sum(Cᴰ³_2)+sum(Cᵁ⁴_2)+sum(Cᴰ⁴_2)+sum(Cᵁ⁵_2)+sum(Cᴰ⁵_2)+sum(Cᵁ²⁷_2)+sum(Cᴰ²⁷_2)+sum(Cᵁ³⁰_2)+sum(Cᴰ³⁰_2)       
cost_nl_Primal=sum(Oⁿˡ[1].*(yˢᴳ²_1+yˢᴳ²_2))+sum(Oⁿˡ[2].*(yˢᴳ³_1+yˢᴳ³_2))+sum(Oⁿˡ[3].*(yˢᴳ⁴_1+yˢᴳ⁴_2))+sum(Oⁿˡ[4].*(yˢᴳ⁵_1+yˢᴳ⁵_2))+sum(Oⁿˡ[5].*(yˢᴳ²⁷_1+yˢᴳ²⁷_2))+sum(Oⁿˡ[6].*(yˢᴳ³⁰_1+yˢᴳ³⁰_2))    
cost_gene_Primal=sum(Oᵐ₁[1].*Pˢᴳ²_1+Oᵐ₂[1].*Pˢᴳ²_2 )+ sum(Oᵐ₁[2].*Pˢᴳ³_1+Oᵐ₂[2].*Pˢᴳ³_2 )+sum(Oᵐ₁[3].*Pˢᴳ⁴_1+Oᵐ₂[3].*Pˢᴳ⁴_2)+sum(Oᵐ₁[4].*Pˢᴳ⁵_1+Oᵐ₂[4].*Pˢᴳ⁵_2)+sum(Oᵐ₁[5].*Pˢᴳ²⁷_1+Oᵐ₂[5].*Pˢᴳ²⁷_2)+sum(Oᵐ₁[6].*Pˢᴳ³⁰_1+Oᵐ₂[6].*Pˢᴳ³⁰_2)   
cost_IBR_Primal=sum(Oᴱ_c[1].*Pᴵᴮᴳ¹) +sum(Oᴱ_c[2].*Pᴵᴮᴳ²³) +sum(Oᴱ_c[3].*Pᴵᴮᴳ²⁶)

obj_Primal=cost_onoff_Primal +cost_nl_Primal +cost_gene_Primal +cost_IBR_Primal

#-Dual obj
@variable(model, obj_Dual_1[1:T])
for t in 1:T
    @constraint(model, obj_Dual_1[t]== Load_total[t]*λᴱ[t] -(φᵐᵃˣ¹[t] +φᵐᵃˣ²³[t] +φᵐᵃˣ²⁶[t]) -ψᵐᵃˣˢᴳ²_1[t] -ψᵐᵃˣˢᴳ²_2[t] -ψᵐᵃˣˢᴳ³_1[t] -ψᵐᵃˣˢᴳ³_2[t] -ψᵐᵃˣˢᴳ⁴_1[t] -ψᵐᵃˣˢᴳ⁴_2[t] -ψᵐᵃˣˢᴳ⁵_1[t] -ψᵐᵃˣˢᴳ⁵_2[t] -ψᵐᵃˣˢᴳ²⁷_1[t] -ψᵐᵃˣˢᴳ²⁷_2[t] -ψᵐᵃˣˢᴳ³⁰_1[t] -ψᵐᵃˣˢᴳ³⁰_2[t])
end


@variable(model, obj_Dual_2[1:T])
for t in 1:T
    @constraint(model, obj_Dual_2[t]==      -Rₘₐₓ[1]*πʳᵈˢᴳ²_1[t] -Rₘₐₓ[1]*πʳᵈˢᴳ²_2[t] -Rₘₐₓ[1]*πʳᵘˢᴳ²_1[t] -Rₘₐₓ[1]*πʳᵘˢᴳ²_2[t]
                                            -Rₘₐₓ[2]*πʳᵈˢᴳ³_1[t] -Rₘₐₓ[2]*πʳᵈˢᴳ³_2[t] -Rₘₐₓ[2]*πʳᵘˢᴳ³_1[t] -Rₘₐₓ[2]*πʳᵘˢᴳ³_2[t]
                                            -Rₘₐₓ[3]*πʳᵈˢᴳ⁴_1[t] -Rₘₐₓ[3]*πʳᵈˢᴳ⁴_2[t] -Rₘₐₓ[3]*πʳᵘˢᴳ⁴_1[t] -Rₘₐₓ[3]*πʳᵘˢᴳ⁴_2[t]
                                            -Rₘₐₓ[4]*πʳᵈˢᴳ⁵_1[t] -Rₘₐₓ[4]*πʳᵈˢᴳ⁵_2[t] -Rₘₐₓ[4]*πʳᵘˢᴳ⁵_1[t] -Rₘₐₓ[4]*πʳᵘˢᴳ⁵_2[t]
                                            -Rₘₐₓ[5]*πʳᵈˢᴳ²⁷_1[t] -Rₘₐₓ[5]*πʳᵈˢᴳ²⁷_2[t] -Rₘₐₓ[5]*πʳᵘˢᴳ²⁷_1[t] -Rₘₐₓ[5]*πʳᵘˢᴳ²⁷_2[t]
                                            -Rₘₐₓ[6]*πʳᵈˢᴳ³⁰_1[t] -Rₘₐₓ[6]*πʳᵈˢᴳ³⁰_2[t] -Rₘₐₓ[6]*πʳᵘˢᴳ³⁰_1[t] -Rₘₐₓ[6]*πʳᵘˢᴳ³⁰_2[t])
end

obj_Dual_3=  (P_g₀[1]*πʳᵈˢᴳ²_1[1] +P_g₀[1]*πʳᵈˢᴳ²_2[1]) -(P_g₀[1]*πʳᵘˢᴳ²_1[1] +P_g₀[1]*πʳᵘˢᴳ²_2[1]) -(yˢᴳ₀[1]*Kˢᵗ[1]*σˢᵗˢᴳ²_1[1] +yˢᴳ₀[1]*Kˢᵗ[1]*σˢᵗˢᴳ²_2[1]) +(yˢᴳ₀[1]*Kˢʰ[1]*σˢʰˢᴳ²_1[1] +yˢᴳ₀[1]*Kˢʰ[1]*σˢʰˢᴳ²_2[1])
           +(P_g₀[2]*πʳᵈˢᴳ³_1[1] +P_g₀[2]*πʳᵈˢᴳ³_2[1]) -(P_g₀[2]*πʳᵘˢᴳ³_1[1] +P_g₀[2]*πʳᵘˢᴳ³_2[1]) -(yˢᴳ₀[2]*Kˢᵗ[2]*σˢᵗˢᴳ³_1[1] +yˢᴳ₀[2]*Kˢᵗ[2]*σˢᵗˢᴳ³_2[1]) +(yˢᴳ₀[2]*Kˢʰ[2]*σˢʰˢᴳ³_1[1] +yˢᴳ₀[2]*Kˢʰ[2]*σˢʰˢᴳ³_2[1])
           +(P_g₀[3]*πʳᵈˢᴳ⁴_1[1] +P_g₀[3]*πʳᵈˢᴳ⁴_2[1]) -(P_g₀[3]*πʳᵘˢᴳ⁴_1[1] +P_g₀[3]*πʳᵘˢᴳ⁴_2[1]) -(yˢᴳ₀[3]*Kˢᵗ[3]*σˢᵗˢᴳ⁴_1[1] +yˢᴳ₀[3]*Kˢᵗ[3]*σˢᵗˢᴳ⁴_2[1]) +(yˢᴳ₀[3]*Kˢʰ[3]*σˢʰˢᴳ⁴_1[1] +yˢᴳ₀[3]*Kˢʰ[3]*σˢʰˢᴳ⁴_2[1])
           +(P_g₀[4]*πʳᵈˢᴳ⁵_1[1] +P_g₀[4]*πʳᵈˢᴳ⁵_2[1]) -(P_g₀[4]*πʳᵘˢᴳ⁵_1[1] +P_g₀[4]*πʳᵘˢᴳ⁵_2[1]) -(yˢᴳ₀[4]*Kˢᵗ[4]*σˢᵗˢᴳ⁵_1[1] +yˢᴳ₀[4]*Kˢᵗ[4]*σˢᵗˢᴳ⁵_2[1]) +(yˢᴳ₀[4]*Kˢʰ[4]*σˢʰˢᴳ⁵_1[1] +yˢᴳ₀[4]*Kˢʰ[4]*σˢʰˢᴳ⁵_2[1])
           +(P_g₀[5]*πʳᵈˢᴳ²⁷_1[1] +P_g₀[5]*πʳᵈˢᴳ²⁷_2[1]) -(P_g₀[5]*πʳᵘˢᴳ²⁷_1[1] +P_g₀[5]*πʳᵘˢᴳ²⁷_2[1]) -(yˢᴳ₀[5]*Kˢᵗ[5]*σˢᵗˢᴳ²⁷_1[1] +yˢᴳ₀[5]*Kˢᵗ[5]*σˢᵗˢᴳ²⁷_2[1]) +(yˢᴳ₀[5]*Kˢʰ[5]*σˢʰˢᴳ²⁷_1[1] +yˢᴳ₀[5]*Kˢʰ[5]*σˢʰˢᴳ²⁷_2[1])
           +(P_g₀[6]*πʳᵈˢᴳ³⁰_1[1] +P_g₀[6]*πʳᵈˢᴳ³⁰_2[1]) -(P_g₀[6]*πʳᵘˢᴳ³⁰_1[1] +P_g₀[6]*πʳᵘˢᴳ³⁰_2[1]) -(yˢᴳ₀[6]*Kˢᵗ[6]*σˢᵗˢᴳ³⁰_1[1] +yˢᴳ₀[6]*Kˢᵗ[6]*σˢᵗˢᴳ³⁰_2[1]) +(yˢᴳ₀[6]*Kˢʰ[6]*σˢʰˢᴳ³⁰_1[1] +yˢᴳ₀[6]*Kˢʰ[6]*σˢʰˢᴳ³⁰_2[1])


@variable(model, obj_Dual_4[1:66,1:T])
for i in 1:66
    @constraint(model, obj_Dual_4[i,:] == -γ_min[i,1,:])
end


obj_Dual=sum( obj_Dual_1 ) +sum(obj_Dual_2) +obj_Dual_3 +sum(obj_Dual_4) + sum(Iₗᵢₘ.*λ_F[1,:])+ sum(Iₗᵢₘ.*λ_F[2,:])+ sum(Iₗᵢₘ.*λ_F[3,:]) + sum(Iₗᵢₘ.*λ_F[4,:])

@constraint(model, obj_Primal >= obj_Dual)

@objective(model, Min, obj_Primal-obj_Dual)  # single-level objective function
#-------Solve and Output Results
set_optimizer(model , Gurobi.Optimizer)
optimize!(model)



obj_Primal=JuMP.value(obj_Primal)
obj_Dual=JuMP.value(obj_Dual)
DG=obj_Primal-obj_Dual


#-----------economic metrics for strategic one
revenue_energy_UL=JuMP.value(revenue_energy_UL)
revenue_SCL_UL=JuMP.value(revenue_SCL_UL)
cost_nl_UL=JuMP.value(cost_nl_UL)
cost_gene_UL=JuMP.value(cost_gene_UL)
cost_onoff_UL=JuMP.value(cost_onoff_UL)
reveunes_UL=revenue_energy_UL +revenue_SCL_UL

costs_UL=cost_nl_UL +cost_gene_UL +cost_onoff_UL


#-----------economic metrics for non-strategic one

Pˢᴳ³_1=JuMP.value.(Pˢᴳ³_1)
Pˢᴳ³_2=JuMP.value.(Pˢᴳ³_2)
Pˢᴳ⁴_1=JuMP.value.(Pˢᴳ⁴_1)
Pˢᴳ⁴_2=JuMP.value.(Pˢᴳ⁴_2)
Pˢᴳ⁵_1=JuMP.value.(Pˢᴳ⁵_1)
Pˢᴳ⁵_2=JuMP.value.(Pˢᴳ⁵_2)
Pˢᴳ²⁷_1=JuMP.value.(Pˢᴳ²⁷_1)
Pˢᴳ²⁷_2=JuMP.value.(Pˢᴳ²⁷_2)
Pˢᴳ³⁰_1=JuMP.value.(Pˢᴳ³⁰_1)
Pˢᴳ³⁰_2=JuMP.value.(Pˢᴳ³⁰_2)

yˢᴳ²_1=JuMP.value.(yˢᴳ²_1)
yˢᴳ²_2=JuMP.value.(yˢᴳ²_2)
yˢᴳ³_1=JuMP.value.(yˢᴳ³_1)
yˢᴳ³_2=JuMP.value.(yˢᴳ³_2)
yˢᴳ⁴_1=JuMP.value.(yˢᴳ⁴_1)
yˢᴳ⁴_2=JuMP.value.(yˢᴳ⁴_2)
yˢᴳ⁵_1=JuMP.value.(yˢᴳ⁵_1)
yˢᴳ⁵_2=JuMP.value.(yˢᴳ⁵_2)
yˢᴳ²⁷_1=JuMP.value.(yˢᴳ²⁷_1)
yˢᴳ²⁷_2=JuMP.value.(yˢᴳ²⁷_2)
yˢᴳ³⁰_1=JuMP.value.(yˢᴳ³⁰_1)
yˢᴳ³⁰_2=JuMP.value.(yˢᴳ³⁰_2)

UC_SGs=zeros(12,24)   # UC of SGs
for t in 1:T
    UC_SGs[1,t]=yˢᴳ²_1[t]
    UC_SGs[2,t]=yˢᴳ²_2[t]
    UC_SGs[3,t]=yˢᴳ³_1[t] 
    UC_SGs[4,t]=yˢᴳ³_2[t]
    UC_SGs[5,t]=yˢᴳ⁴_1[t]
    UC_SGs[6,t]=yˢᴳ⁴_2[t]
    UC_SGs[7,t]=yˢᴳ⁵_1[t]
    UC_SGs[8,t]=yˢᴳ⁵_2[t]
    UC_SGs[9,t]=yˢᴳ²⁷_1[t]
    UC_SGs[10,t]=yˢᴳ²⁷_2[t]
    UC_SGs[11,t]=yˢᴳ³⁰_1[t]
    UC_SGs[12,t]=yˢᴳ³⁰_2[t]
end

λ_SCL=zeros(4,T)  # dual variables for SCL constraints
for t in 1:T
    λ_SCL[1,t]=JuMP.value(λ_F[1,t])  # bus 11
    λ_SCL[2,t]=JuMP.value(λ_F[2,t])  # bus 26
    λ_SCL[3,t]=JuMP.value(λ_F[3,t])  # bus 29
    λ_SCL[4,t]=JuMP.value(λ_F[4,t])  # bus 30
end

λ_SCL_ᴳ²_1, λ_SCL_ᴳ²_2, λ_SCL_ᴳ³_1, λ_SCL_ᴳ³_2, λ_SCL_ᴳ⁴_1, λ_SCL_ᴳ⁴_2, 
    λ_SCL_ᴳ⁵_1, λ_SCL_ᴳ⁵_2, λ_SCL_ᴳ²⁷_1, λ_SCL_ᴳ²⁷_2, λ_SCL_ᴳ³⁰_1, λ_SCL_ᴳ³⁰_2 = cal_SCL_contribution(UC_SGs, λ_SCL)


plot(λ_SCL_ᴳ²_1[4,:])
plot!(λ_SCL_ᴳ³_1[4,:])
plot!(λ_SCL_ᴳ⁴_1[4,:])
plot!(λ_SCL_ᴳ⁵_1[4,:])
plot!(λ_SCL_ᴳ²⁷_1[4,:])
plot!( λ_SCL_ᴳ³⁰_1[4,:])

Cᵁ³_1=JuMP.value.(Cᵁ³_1)
Cᴰ³_1=JuMP.value.(Cᴰ³_1)
Cᵁ⁴_1=JuMP.value.(Cᵁ⁴_1)
Cᴰ⁴_1=JuMP.value.(Cᴰ⁴_1)
Cᵁ⁵_1=JuMP.value.(Cᵁ⁵_1)
Cᴰ⁵_1=JuMP.value.(Cᴰ⁵_1)
Cᵁ²⁷_1=JuMP.value.(Cᵁ²⁷_1)
Cᴰ²⁷_1=JuMP.value.(Cᴰ²⁷_1)
Cᵁ³⁰_1=JuMP.value.(Cᵁ³⁰_1)
Cᴰ³⁰_1=JuMP.value.(Cᴰ³⁰_1)
Cᵁ³_2=JuMP.value.(Cᵁ³_2)
Cᴰ³_2=JuMP.value.(Cᴰ³_2)
Cᵁ⁴_2=JuMP.value.(Cᵁ⁴_2)
Cᴰ⁴_2=JuMP.value.(Cᴰ⁴_2)
Cᵁ⁵_2=JuMP.value.(Cᵁ⁵_2)
Cᴰ⁵_2=JuMP.value.(Cᴰ⁵_2)
Cᵁ²⁷_2=JuMP.value.(Cᵁ²⁷_2)
Cᴰ²⁷_2=JuMP.value.(Cᴰ²⁷_2)
Cᵁ³⁰_2=JuMP.value.(Cᵁ³⁰_2)
Cᴰ³⁰_2=JuMP.value.(Cᴰ³⁰_2)

λ_F=JuMP.value.(λ_F) 
λᴱ=JuMP.value.(λᴱ) 
plot(λᴱ)

plot(λ_F[1,:])
plot!(λ_F[2,:])
plot!(λ_F[3,:])
plot!(λ_F[4,:])

matwrite("power_balance.mat", Dict("power_balance" => λᴱ))
matwrite("dual_val_SCL_con_11.mat", Dict("dual_val_SCL_con_11" => λ_F[1,:]))
matwrite("dual_val_SCL_con_26.mat", Dict("dual_val_SCL_con_26" => λ_F[2,:]))
matwrite("dual_val_SCL_con_29.mat", Dict("dual_val_SCL_con_29" => λ_F[3,:]))
matwrite("dual_val_SCL_con_30.mat", Dict("dual_val_SCL_con_30" => λ_F[4,:]))


I_min=zeros(1,30)
I_scc=zeros(30,T)

for k in 1:30
for t in 1:T                                             # bounds for the SCL of buses  I_₂₆   
   I_scc[k,t]=K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
    K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
    K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
    K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
    K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
    K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+

    K_c[1,k]*α₁[1]+ K_c[2,k]*α₂₃[1]+ K_c[3,k]*α₂₆[1]+
    
    K_m[1,k]*yˢᴳ²_1[t]*yˢᴳ²_2[t]+ K_m[2,k]*yˢᴳ²_1[t]*yˢᴳ³_1[t]+ K_m[3,k]*yˢᴳ²_1[t]*yˢᴳ³_2[t]+ K_m[4,k]*yˢᴳ²_1[t]*yˢᴳ⁴_1[t]+
    K_m[5,k]*yˢᴳ²_1[t]*yˢᴳ⁴_2[t]+ K_m[6,k]*yˢᴳ²_1[t]*yˢᴳ⁵_1[t]+ K_m[7,k]*yˢᴳ²_1[t]*yˢᴳ⁵_2[t]+ K_m[8,k]*yˢᴳ²_1[t]*yˢᴳ²⁷_1[t]+
    K_m[9,k]*yˢᴳ²_1[t]*yˢᴳ²⁷_2[t]+ K_m[10,k]*yˢᴳ²_1[t]*yˢᴳ³⁰_1[t]+ K_m[11,k]*yˢᴳ²_1[t]*yˢᴳ³⁰_2[t]+

    K_m[12,k]*yˢᴳ²_2[t]*yˢᴳ³_1[t]+ K_m[13,k]*yˢᴳ²_2[t]*yˢᴳ³_2[t]+ K_m[14,k]*yˢᴳ²_2[t]*yˢᴳ⁴_1[t]+
    K_m[15,k]*yˢᴳ²_2[t]*yˢᴳ⁴_2[t]+ K_m[16,k]*yˢᴳ²_2[t]*yˢᴳ⁵_1[t]+ K_m[17,k]*yˢᴳ²_2[t]*yˢᴳ⁵_2[t]+ K_m[18,k]*yˢᴳ²_2[t]*yˢᴳ²⁷_1[t]+
    K_m[19,k]*yˢᴳ²_2[t]*yˢᴳ²⁷_2[t]+ K_m[20,k]*yˢᴳ²_2[t]*yˢᴳ³⁰_1[t]+ K_m[21,k]*yˢᴳ²_2[t]*yˢᴳ³⁰_2[t]+

    K_m[22,k]*yˢᴳ³_1[t]*yˢᴳ³_2[t]+ K_m[23,k]*yˢᴳ³_1[t]*yˢᴳ⁴_1[t]+
    K_m[24,k]*yˢᴳ³_1[t]*yˢᴳ⁴_2[t]+ K_m[25,k]*yˢᴳ³_1[t]*yˢᴳ⁵_1[t]+ K_m[26,k]*yˢᴳ³_1[t]*yˢᴳ⁵_2[t]+ K_m[27,k]*yˢᴳ³_1[t]*yˢᴳ²⁷_1[t]+
    K_m[28,k]*yˢᴳ³_1[t]*yˢᴳ²⁷_2[t]+ K_m[29,k]*yˢᴳ³_1[t]*yˢᴳ³⁰_1[t]+ K_m[30,k]*yˢᴳ³_1[t]*yˢᴳ³⁰_2[t]+

    K_m[31,k]*yˢᴳ³_2[t]*yˢᴳ⁴_1[t]+
    K_m[32,k]*yˢᴳ³_2[t]*yˢᴳ⁴_2[t]+ K_m[33,k]*yˢᴳ³_2[t]*yˢᴳ⁵_1[t]+ K_m[34,k]*yˢᴳ³_2[t]*yˢᴳ⁵_2[t]+ K_m[35,k]*yˢᴳ³_2[t]*yˢᴳ²⁷_1[t]+
    K_m[36,k]*yˢᴳ³_2[t]*yˢᴳ²⁷_2[t]+ K_m[37,k]*yˢᴳ³_2[t]*yˢᴳ³⁰_1[t]+ K_m[38,k]*yˢᴳ³_2[t]*yˢᴳ³⁰_2[t]+

    K_m[39,k]*yˢᴳ⁴_1[t]*yˢᴳ⁴_2[t]+ K_m[40,k]*yˢᴳ⁴_1[t]*yˢᴳ⁵_1[t]+ K_m[41,k]*yˢᴳ⁴_1[t]*yˢᴳ⁵_2[t]+ K_m[42,k]*yˢᴳ⁴_1[t]*yˢᴳ²⁷_1[t]+
    K_m[43,k]*yˢᴳ⁴_1[t]*yˢᴳ²⁷_2[t]+ K_m[44,k]*yˢᴳ⁴_1[t]*yˢᴳ³⁰_1[t]+ K_m[45,k]*yˢᴳ⁴_1[t]*yˢᴳ³⁰_2[t]+

    K_m[46,k]*yˢᴳ⁴_2[t]*yˢᴳ⁵_1[t]+ K_m[47,k]*yˢᴳ⁴_2[t]*yˢᴳ⁵_2[t]+ K_m[48,k]*yˢᴳ⁴_2[t]*yˢᴳ²⁷_1[t]+
    K_m[49,k]*yˢᴳ⁴_2[t]*yˢᴳ²⁷_2[t]+ K_m[50,k]*yˢᴳ⁴_2[t]*yˢᴳ³⁰_1[t]+ K_m[51,k]*yˢᴳ⁴_2[t]*yˢᴳ³⁰_2[t]+

    K_m[52,k]*yˢᴳ⁵_1[t]*yˢᴳ⁵_2[t]+ K_m[53,k]*yˢᴳ⁵_1[t]*yˢᴳ²⁷_1[t]+
    K_m[54,k]*yˢᴳ⁵_1[t]*yˢᴳ²⁷_2[t]+ K_m[55,k]*yˢᴳ⁵_1[t]*yˢᴳ³⁰_1[t]+ K_m[56,k]*yˢᴳ⁵_1[t]*yˢᴳ³⁰_2[t]+

    K_m[57,k]*yˢᴳ⁵_2[t]*yˢᴳ²⁷_1[t]+
    K_m[58,k]*yˢᴳ⁵_2[t]*yˢᴳ²⁷_2[t]+ K_m[59,k]*yˢᴳ⁵_2[t]*yˢᴳ³⁰_1[t]+ K_m[60,k]*yˢᴳ⁵_2[t]*yˢᴳ³⁰_2[t]+

    K_m[61,k]*yˢᴳ²⁷_1[t]*yˢᴳ²⁷_2[t]+ K_m[62,k]*yˢᴳ²⁷_1[t]*yˢᴳ³⁰_1[t]+ K_m[63,k]*yˢᴳ²⁷_1[t]*yˢᴳ³⁰_2[t]+

    K_m[64,k]*yˢᴳ²⁷_2[t]*yˢᴳ³⁰_1[t]+ K_m[65,k]*yˢᴳ²⁷_2[t]*yˢᴳ³⁰_2[t]+

    K_m[66,k]*yˢᴳ³⁰_1[t]*yˢᴳ³⁰_2[t]

end
I_min[k]=minimum(I_scc[k,:])
end

bar(I_min')

matwrite("costs_G30_2.mat", Dict("costs_G30_2" => costs_G30_2))

matwrite("revenue_SCL_G30_2.mat", Dict("revenue_SCL_G30_2" => revenue_SCL_G30_2))
matwrite("revenue_energy_G30_2.mat", Dict("revenue_energy_G30_2" => revenue_energy_G30_2))
matwrite("costs_G30_2.mat", Dict("costs_G30_2" => costs_G30_2))
