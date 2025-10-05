# Author: Peng Wang       from Technical University of Madrid (UPM)
# Supervisor: Luis Badesa

# Pricing SCL by primal-dual formulation
# 29.May.2025

import Pkg
using JuMP,Gurobi, CSV,DataFrames,LinearAlgebra, XLSX, IterTools, DelimitedFiles,Plots,MAT, CPLEX
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
#Rₘₐₓ=[3.3, 2.9, 1.6, 1.334, 1.951, 1.728]*10^3/5                    #  Ramp limits of SGs                      SGs, buses:2,3,4,5,27,30
Kˢᵗ=[200, 125, 92.5, 72, 55, 31]*10^3 /10                           #  Startup cost of SGs                     SGs, buses:2,3,4,5,27,30
Kˢʰ=[50, 28.5, 18.5, 14.4, 12, 10]*10^3  /10                        #  Shutdown cost of SGs                    SGs, buses:2,3,4,5,27,30
Oᵐ₁=[6.20, 7.10, 10.47, 12.28, 13.53, 15.36]                        #  Marginal generation cost of SG 1  in    SGs, buses:2,3,4,5,27,30
Oᵐ₂=[7.07, 8.72, 11.49, 12.84, 14.60, 15.02]                        #  Marginal generation cost of SG 2  in    SGs, buses:2,3,4,5,27,30
Oⁿˡ=[17.431, 15.005, 13.755, 10.930, 9.900, 8.570]*10^3 /10         #  No-load cost of SGs                     SGs, buses:2,3,4,5,27,30
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

@variable(model, yˢᴳ²_1[1:T])      # status of SGs, buses:2,3,4,5,27,30.  Include strategic and non-strategic players
@variable(model, yˢᴳ²_2[1:T])          
@variable(model, yˢᴳ³_1[1:T])  
@variable(model, yˢᴳ³_2[1:T])          
@variable(model, yˢᴳ⁴_1[1:T])      
@variable(model, yˢᴳ⁴_2[1:T])      
@variable(model, yˢᴳ⁵_1[1:T])      
@variable(model, yˢᴳ⁵_2[1:T])      
@variable(model, yˢᴳ²⁷_1[1:T])  
@variable(model, yˢᴳ²⁷_2[1:T])          
@variable(model, yˢᴳ³⁰_1[1:T])   
@variable(model, yˢᴳ³⁰_2[1:T])                       


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

#-------Define Primal Constraints
Power_balance=Dict()
for t in 1:T
    Power_balance[t]=@constraint(model, Pˢᴳ²_1[t]+Pˢᴳ²_2[t]+Pˢᴳ³_1[t]+Pˢᴳ³_2[t]+Pˢᴳ⁴_1[t]+Pˢᴳ⁴_2[t]+Pˢᴳ⁵_1[t]+Pˢᴳ⁵_2[t]+Pˢᴳ²⁷_1[t]+Pˢᴳ²⁷_2[t]+Pˢᴳ³⁰_1[t]+Pˢᴳ³⁰_2[t]+
                   Pᴵᴮᴳ¹[t]+Pᴵᴮᴳ²³[t]+Pᴵᴮᴳ²⁶[t]==Load_total[t])     # power balance , dual variable: λᴱₜ
end

price_yˢᴳ²_1=Dict()
price_yˢᴳ²_2=Dict()
price_yˢᴳ³_1=Dict()
price_yˢᴳ³_2=Dict()
price_yˢᴳ⁴_1=Dict()
price_yˢᴳ⁴_2=Dict()
price_yˢᴳ⁵_1=Dict()
price_yˢᴳ⁵_2=Dict()
price_yˢᴳ²⁷_1=Dict()
price_yˢᴳ²⁷_2=Dict()
price_yˢᴳ³⁰_1=Dict()
price_yˢᴳ³⁰_2=Dict()
for t in 1:T
    price_yˢᴳ²_1[t]=@constraint(model, yˢᴳ²_1[t]== yˢᴳ²_1_opt[t]) 
    price_yˢᴳ²_2[t]=@constraint(model, yˢᴳ²_2[t]== yˢᴳ²_2_opt[t])
    price_yˢᴳ³_1[t]=@constraint(model, yˢᴳ³_1[t]== yˢᴳ³_1_opt[t])
    price_yˢᴳ³_2[t]=@constraint(model, yˢᴳ³_2[t]== yˢᴳ³_2_opt[t])
    price_yˢᴳ⁴_1[t]=@constraint(model, yˢᴳ⁴_1[t]== yˢᴳ⁴_1_opt[t])
    price_yˢᴳ⁴_2[t]=@constraint(model, yˢᴳ⁴_2[t]== yˢᴳ⁴_2_opt[t])
    price_yˢᴳ⁵_1[t]=@constraint(model, yˢᴳ⁵_1[t]== yˢᴳ⁵_1_opt[t])
    price_yˢᴳ⁵_2[t]=@constraint(model, yˢᴳ⁵_2[t]== yˢᴳ⁵_2_opt[t])
    price_yˢᴳ²⁷_1[t]=@constraint(model, yˢᴳ²⁷_1[t]== yˢᴳ²⁷_1_opt[t])
    price_yˢᴳ²⁷_2[t]=@constraint(model, yˢᴳ²⁷_2[t]== yˢᴳ²⁷_2_opt[t])
    price_yˢᴳ³⁰_1[t]=@constraint(model, yˢᴳ³⁰_1[t]== yˢᴳ³⁰_1_opt[t])
    price_yˢᴳ³⁰_2[t]=@constraint(model, yˢᴳ³⁰_2[t]== yˢᴳ³⁰_2_opt[t])
end

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
    

Wind_dual_1=Dict()
Wind_dual_23=Dict()
Wind_dual_26=Dict()
for t in 1:T
    Wind_dual_1[t]=@constraint(model, Pᴵᴮᴳ¹[t] <= IBG₁[t])        # wind power limit  , dual variable: ζᵐᵃˣₜ
    Wind_dual_23[t]=@constraint(model, Pᴵᴮᴳ²³[t]<= IBG₂₃[t])       
    Wind_dual_26[t]=@constraint(model, Pᴵᴮᴳ²⁶[t]<= IBG₂₆[t])       
                 
end





cost_onoff_Primal=sum(Cᵁ²_1)+sum(Cᴰ²_1)+sum(Cᵁ³_1)+sum(Cᴰ³_1)+sum(Cᵁ⁴_1)+sum(Cᴰ⁴_1)+sum(Cᵁ⁵_1)+sum(Cᴰ⁵_1)+sum(Cᵁ²⁷_1)+sum(Cᴰ²⁷_1)+sum(Cᵁ³⁰_1)+sum(Cᴰ³⁰_1)  +sum(Cᵁ²_2)+sum(Cᴰ²_2)+sum(Cᵁ³_2)+sum(Cᴰ³_2)+sum(Cᵁ⁴_2)+sum(Cᴰ⁴_2)+sum(Cᵁ⁵_2)+sum(Cᴰ⁵_2)+sum(Cᵁ²⁷_2)+sum(Cᴰ²⁷_2)+sum(Cᵁ³⁰_2)+sum(Cᴰ³⁰_2)       
cost_nl_Primal=sum(Oⁿˡ[1].*(yˢᴳ²_1+yˢᴳ²_2))+sum(Oⁿˡ[2].*(yˢᴳ³_1+yˢᴳ³_2))+sum(Oⁿˡ[3].*(yˢᴳ⁴_1+yˢᴳ⁴_2))+sum(Oⁿˡ[4].*(yˢᴳ⁵_1+yˢᴳ⁵_2))+sum(Oⁿˡ[5].*(yˢᴳ²⁷_1+yˢᴳ²⁷_2))+sum(Oⁿˡ[6].*(yˢᴳ³⁰_1+yˢᴳ³⁰_2))    
cost_gene_Primal=sum(Oᵐ₁[1].*Pˢᴳ²_1+Oᵐ₂[1].*Pˢᴳ²_2 )+ sum(Oᵐ₁[2].*Pˢᴳ³_1+Oᵐ₂[2].*Pˢᴳ³_2 )+sum(Oᵐ₁[3].*Pˢᴳ⁴_1+Oᵐ₂[3].*Pˢᴳ⁴_2)+sum(Oᵐ₁[4].*Pˢᴳ⁵_1+Oᵐ₂[4].*Pˢᴳ⁵_2)+sum(Oᵐ₁[5].*Pˢᴳ²⁷_1+Oᵐ₂[5].*Pˢᴳ²⁷_2)+sum(Oᵐ₁[6].*Pˢᴳ³⁰_1+Oᵐ₂[6].*Pˢᴳ³⁰_2)   

obj_Primal=cost_onoff_Primal +cost_nl_Primal +cost_gene_Primal


@objective(model, Min, obj_Primal)  # single-level objective function
#-------Solve and Output Results
set_optimizer(model , Gurobi.Optimizer)
optimize!(model)


Pᴵᴮᴳ¹=value.(Pᴵᴮᴳ¹)
Pᴵᴮᴳ²³=value.(Pᴵᴮᴳ²³)
Pᴵᴮᴳ²⁶=value.(Pᴵᴮᴳ²⁶)
plot(Pᴵᴮᴳ¹)
plot(Pᴵᴮᴳ²³)
plot(Pᴵᴮᴳ²⁶)

energy_price_restricted=zeros(1,T)
SCC_price_bus11=zeros(1,T)
SCC_price_bus26=zeros(1,T)
SCC_price_bus29=zeros(1,T)
SCC_price_bus30=zeros(1,T)
commit_price_restricted_yˢᴳ²_1=zeros(1,T)
commit_price_restricted_yˢᴳ²_2=zeros(1,T)
commit_price_restricted_yˢᴳ³_1=zeros(1,T)
commit_price_restricted_yˢᴳ³_2=zeros(1,T)
commit_price_restricted_yˢᴳ⁴_1=zeros(1,T)
commit_price_restricted_yˢᴳ⁴_2=zeros(1,T)
commit_price_restricted_yˢᴳ⁵_1=zeros(1,T)
commit_price_restricted_yˢᴳ⁵_2=zeros(1,T)
commit_price_restricted_yˢᴳ²⁷_1=zeros(1,T)
commit_price_restricted_yˢᴳ²⁷_2=zeros(1,T)
commit_price_restricted_yˢᴳ³⁰_1=zeros(1,T)
commit_price_restricted_yˢᴳ³⁰_2=zeros(1,T)
dual_Wind_1=zeros(1,T)
dual_Wind_23=zeros(1,T)
dual_Wind_26=zeros(1,T)

for t in 1:T
    energy_price_restricted[t]=dual(Power_balance[t])
    SCC_price_bus11[t]=dual(price_SCC_bus11[t])
    SCC_price_bus26[t]=dual(price_SCC_bus26[t])
    SCC_price_bus29[t]=dual(price_SCC_bus29[t])
    SCC_price_bus30[t]=dual(price_SCC_bus30[t])
    commit_price_restricted_yˢᴳ²_1[t]=dual(price_yˢᴳ²_1[t])
    commit_price_restricted_yˢᴳ²_2[t]=dual(price_yˢᴳ²_2[t])
    commit_price_restricted_yˢᴳ³_1[t]=dual(price_yˢᴳ³_1[t])
    commit_price_restricted_yˢᴳ³_2[t]=dual(price_yˢᴳ³_2[t])
    commit_price_restricted_yˢᴳ⁴_1[t]=dual(price_yˢᴳ⁴_1[t])
    commit_price_restricted_yˢᴳ⁴_2[t]=dual(price_yˢᴳ⁴_2[t])
    commit_price_restricted_yˢᴳ⁵_1[t]=dual(price_yˢᴳ⁵_1[t])
    commit_price_restricted_yˢᴳ⁵_2[t]=dual(price_yˢᴳ⁵_2[t])
    commit_price_restricted_yˢᴳ²⁷_1[t]=dual(price_yˢᴳ²⁷_1[t])
    commit_price_restricted_yˢᴳ²⁷_2[t]=dual(price_yˢᴳ²⁷_2[t])
    commit_price_restricted_yˢᴳ³⁰_1[t]=dual(price_yˢᴳ³⁰_1[t])
    commit_price_restricted_yˢᴳ³⁰_2[t]=dual(price_yˢᴳ³⁰_2[t])
    dual_Wind_1[t]=dual(Wind_dual_1[t])
    dual_Wind_23[t]=dual(Wind_dual_23[t])
    dual_Wind_26[t]=dual(Wind_dual_26[t])
end


plot(energy_price_restricted')

plot(commit_price_restricted_yˢᴳ²_1')
plot!(commit_price_restricted_yˢᴳ²_2')
plot!(commit_price_restricted_yˢᴳ³_1')
plot!(commit_price_restricted_yˢᴳ³_2')
plot(commit_price_restricted_yˢᴳ⁴_1')
plot!(commit_price_restricted_yˢᴳ⁴_2')
plot!(commit_price_restricted_yˢᴳ⁵_1')
plot!(commit_price_restricted_yˢᴳ⁵_2')   
plot(commit_price_restricted_yˢᴳ²⁷_1')
plot!(commit_price_restricted_yˢᴳ²⁷_2')
plot!(commit_price_restricted_yˢᴳ³⁰_1')
plot!(commit_price_restricted_yˢᴳ³⁰_2')

Cᵁ²_1=value.(Cᵁ²_1)
Cᴰ²_1=value.(Cᴰ²_1)
Cᵁ³_1=value.(Cᵁ³_1)
Cᴰ³_1=value.(Cᴰ³_1)
Cᵁ⁴_1=value.(Cᵁ⁴_1)
Cᴰ⁴_1=value.(Cᴰ⁴_1)
Cᵁ⁵_1=value.(Cᵁ⁵_1)
Cᴰ⁵_1=value.(Cᴰ⁵_1)
Cᵁ²⁷_1=value.(Cᵁ²⁷_1)
Cᴰ²⁷_1=value.(Cᴰ²⁷_1)
Cᵁ³⁰_1=value.(Cᵁ³⁰_1)
Cᴰ³⁰_1=value.(Cᴰ³⁰_1)
Cᵁ²_2=value.(Cᵁ²_2)
Cᴰ²_2=value.(Cᴰ²_2)
Cᵁ³_2=value.(Cᵁ³_2)
Cᴰ³_2=value.(Cᴰ³_2)
Cᵁ⁴_2=value.(Cᵁ⁴_2)
Cᴰ⁴_2=value.(Cᴰ⁴_2)
Cᵁ⁵_2=value.(Cᵁ⁵_2)
Cᴰ⁵_2=value.(Cᴰ⁵_2)
Cᵁ²⁷_2=value.(Cᵁ²⁷_2)
Cᴰ²⁷_2=value.(Cᴰ²⁷_2)
Cᵁ³⁰_2=value.(Cᵁ³⁰_2)
Cᴰ³⁰_2=value.(Cᴰ³⁰_2)

Pˢᴳ²_1=JuMP.value.(Pˢᴳ²_1)
Pˢᴳ²_2=JuMP.value.(Pˢᴳ²_2)
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


sum(commit_price_restricted_yˢᴳ²_1)+sum(commit_price_restricted_yˢᴳ²_2)+
sum(commit_price_restricted_yˢᴳ³_1)+sum(commit_price_restricted_yˢᴳ³_2)+
sum(commit_price_restricted_yˢᴳ⁴_1)+sum(commit_price_restricted_yˢᴳ⁴_2)+
sum(commit_price_restricted_yˢᴳ⁵_1)+sum(commit_price_restricted_yˢᴳ⁵_2)+
sum(commit_price_restricted_yˢᴳ²⁷_1)+sum(commit_price_restricted_yˢᴳ²⁷_2)+
sum(commit_price_restricted_yˢᴳ³⁰_1)+sum(commit_price_restricted_yˢᴳ³⁰_2)

#----------
sum(Pˢᴳ²_1 .*energy_price_restricted')-      #energy revenue for SGs-1 in buses 2 
(   sum(Pˢᴳ²_1 .*Oᵐ₁[1])  +
sum(Cᵁ²_1)+sum(Cᴰ²_1) + 
sum(yˢᴳ²_1.*Oⁿˡ[1]) ) +
sum(commit_price_restricted_yˢᴳ²_1.*yˢᴳ²_1')

sum(Pˢᴳ²_2 .*energy_price_restricted')-      #energy revenue for SGs-2 in buses 2 
(   sum(Pˢᴳ²_2 .*Oᵐ₂[1])  +
sum(Cᵁ²_2)+sum(Cᴰ²_2) + 
sum(yˢᴳ²_2.*Oⁿˡ[1]) ) +
sum(commit_price_restricted_yˢᴳ²_2.*yˢᴳ²_2')

sum(Pˢᴳ³_1 .*energy_price_restricted')-      #energy revenue for SGs-1 in buses  3
(   sum(Pˢᴳ³_1 .*Oᵐ₁[2])  +
sum(Cᵁ³_1)+sum(Cᴰ³_1) + 
sum(yˢᴳ³_1.*Oⁿˡ[2]) ) +
sum(commit_price_restricted_yˢᴳ³_1.*yˢᴳ³_1')

sum(Pˢᴳ³_2 .*energy_price_restricted')-      #energy revenue for SGs-2 in buses  3
(   sum(Pˢᴳ³_2 .*Oᵐ₂[2])  +
sum(Cᵁ³_2)+sum(Cᴰ³_2) + 
sum(yˢᴳ³_2.*Oⁿˡ[2]) ) +
sum(commit_price_restricted_yˢᴳ³_2.*yˢᴳ³_2')

sum(Pˢᴳ⁴_1 .*energy_price_restricted')-      #energy revenue for SGs-1 in buses 4 
(   sum(Pˢᴳ⁴_1 .*Oᵐ₁[3])  +
sum(Cᵁ⁴_1)+sum(Cᴰ⁴_1) + 
sum(yˢᴳ⁴_1.*Oⁿˡ[3]) ) +
sum(commit_price_restricted_yˢᴳ⁴_1.*yˢᴳ⁴_1')

sum(Pˢᴳ⁴_2 .*energy_price_restricted')-      #energy revenue for SGs-2 in buses  4
(   sum(Pˢᴳ⁴_2 .*Oᵐ₂[3])  +
sum(Cᵁ⁴_2) +sum(Cᴰ⁴_2) + 
sum(yˢᴳ⁴_2.*Oⁿˡ[3]) ) +
sum(commit_price_restricted_yˢᴳ⁴_2.*yˢᴳ⁴_2')

sum(Pˢᴳ⁵_1 .*energy_price_restricted')-      #energy revenue for SGs-1 in buses 5 
(   sum(Pˢᴳ⁵_1 .*Oᵐ₁[4])  +
sum(Cᵁ⁵_1) +sum(Cᴰ⁵_1) + 
sum(yˢᴳ⁵_1.*Oⁿˡ[4]) ) +
sum(commit_price_restricted_yˢᴳ⁵_1.*yˢᴳ⁵_1')

sum(Pˢᴳ⁵_2 .*energy_price_restricted')-      #energy revenue for SGs-2 in buses  5
(   sum(Pˢᴳ⁵_2 .*Oᵐ₂[4])  +
sum(Cᵁ⁵_2)+sum(Cᴰ⁵_2) + 
sum(yˢᴳ⁵_2.*Oⁿˡ[4]) ) +
sum(commit_price_restricted_yˢᴳ⁵_2.*yˢᴳ⁵_2')

sum(Pˢᴳ²⁷_1 .*energy_price_restricted')-      #energy revenue for SGs-1 in buses 27
(   sum(Pˢᴳ²⁷_1 .*Oᵐ₁[5])  +
sum(Cᵁ²⁷_1)+sum(Cᴰ²⁷_1) + 
sum(yˢᴳ²⁷_1.*Oⁿˡ[5]) ) +
sum(commit_price_restricted_yˢᴳ²⁷_1.*yˢᴳ²⁷_1')

sum(Pˢᴳ²⁷_2 .*energy_price_restricted')-      #energy revenue for SGs-2 in buses 27
(   sum(Pˢᴳ²⁷_2 .*Oᵐ₂[5])  +
sum(Cᵁ²⁷_2)+sum(Cᴰ²⁷_2) + 
sum(yˢᴳ²⁷_2.*Oⁿˡ[5]) ) +
sum(commit_price_restricted_yˢᴳ²⁷_2.*yˢᴳ²⁷_2')

sum(Pˢᴳ³⁰_1 .*energy_price_restricted')-      #energy revenue for SGs-1 in buses 30
(   sum(Pˢᴳ³⁰_1 .*Oᵐ₁[6])  +
sum(Cᵁ³⁰_1)+sum(Cᴰ³⁰_1) + 
sum(yˢᴳ³⁰_1.*Oⁿˡ[6]) ) +
sum(commit_price_restricted_yˢᴳ³⁰_1.*yˢᴳ³⁰_1')

sum(Pˢᴳ³⁰_2 .*energy_price_restricted')-      #energy revenue for SGs-2 in buses 30
(   sum(Pˢᴳ³⁰_2 .*Oᵐ₂[6])  +
sum(Cᵁ³⁰_2)+sum(Cᴰ³⁰_2) + 
sum(yˢᴳ³⁰_2.*Oⁿˡ[6]) ) +
sum(commit_price_restricted_yˢᴳ³⁰_2.*yˢᴳ³⁰_2')



matwrite("energy_price_restricted.mat", Dict("energy_price_restricted" => energy_price_restricted))
matwrite("Load_total.mat", Dict("Load_total" => Load_total))


matwrite("commit_price_restricted_y_21.mat", Dict("commit_price_restricted_y_21" => commit_price_restricted_yˢᴳ²_1))
matwrite("commit_price_restricted_y_22.mat", Dict("commit_price_restricted_y_22" => commit_price_restricted_yˢᴳ²_2))
matwrite("commit_price_restricted_y_301.mat", Dict("commit_price_restricted_y_301" => commit_price_restricted_yˢᴳ³⁰_1))
matwrite("commit_price_restricted_y_302.mat", Dict("commit_price_restricted_y_302" => commit_price_restricted_yˢᴳ³⁰_2))

matwrite("commit_price_restricted_y_31.mat", Dict("commit_price_restricted_y_31" => commit_price_restricted_yˢᴳ³_1))
matwrite("commit_price_restricted_y_32.mat", Dict("commit_price_restricted_y_32" => commit_price_restricted_yˢᴳ³_2))
matwrite("commit_price_restricted_y_271.mat", Dict("commit_price_restricted_y_271" => commit_price_restricted_yˢᴳ²⁷_1))
matwrite("commit_price_restricted_y_272.mat", Dict("commit_price_restricted_y_272" => commit_price_restricted_yˢᴳ²⁷_2))


matwrite("commit_price_restricted_y_41.mat", Dict("commit_price_restricted_y_41" => commit_price_restricted_yˢᴳ⁴_1))
matwrite("commit_price_restricted_y_42.mat", Dict("commit_price_restricted_y_42" => commit_price_restricted_yˢᴳ⁴_2))
matwrite("commit_price_restricted_y_51.mat", Dict("commit_price_restricted_y_51" => commit_price_restricted_yˢᴳ⁵_1))
matwrite("commit_price_restricted_y_52.mat", Dict("commit_price_restricted_y_52" => commit_price_restricted_yˢᴳ⁵_2))



@variable(model, I_₁₁[1:T])             # define SCL on bus 26,29,30 , dual variables: λ_₂₆, λ_₂₉, λ_₃₀, λ_lim₂₆, λ_lim₂₉, λ_lim₃₀
@variable(model, I_₂₆[1:T])
@variable(model, I_₂₉[1:T])
@variable(model, I_₃₀[1:T])

@variable(model, η_m[1:66,1:T]>=0)         #McCormick for SCL constraints, i.e., products of binary variables
#@constraint(model, η_m.<=1) 


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


price_SCC_bus11=Dict()
k=11   
for t in 1:T                                              # bounds for the SCL of buses  I_₃₀
        @constraint(model, I_₁₁[t]==                     
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+
        K_c[1,k]+ K_c[2,k]+ K_c[3,k]+
        
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

       price_SCC_bus11[t]=@constraint(model, I_₁₁[t]>=Iₗᵢₘ)                  # AS requirement for SCL on bus F=30  , dual variable: λ_F  
end

price_SCC_bus26=Dict()
k=26    # for bus 26   
for t in 1:T                                             # bounds for the SCL of buses  I_₂₆   
        @constraint(model, I_₂₆[t]==                     # SCL on bus F=26
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+
        K_c[1,k]+ K_c[2,k]+ K_c[3,k]+

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

        price_SCC_bus26[t]=@constraint(model, I_₂₆[t]>=Iₗᵢₘ)                  # AS requirement for SCL on bus F=26  , dual variable: λ_F  
end

price_SCC_bus29=Dict()
k=29    
for t in 1:T                                              # bounds for the SCL of buses   I_₂₉  
        @constraint(model, I_₂₉[t]==                      # SCL on bus F=29  
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+
        K_c[1,k]+ K_c[2,k]+ K_c[3,k]+
        
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

        price_SCC_bus29[t]=@constraint(model, I_₂₉[t]>=Iₗᵢₘ)                  # AS requirement for SCL on bus F=29  , dual variable: λ_F  
end

price_SCC_bus30=Dict()
k=30    
for t in 1:T                                              # bounds for the SCL of buses  I_₃₀
        @constraint(model, I_₃₀[t]==                     # SCL on bus F=30  
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+
        K_c[1,k]+ K_c[2,k]+ K_c[3,k]+
        
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

        price_SCC_bus30[t]=@constraint(model, I_₃₀[t]>=Iₗᵢₘ)                  # AS requirement for SCL on bus F=30  , dual variable: λ_F  
end