#=
Author: Peng Wang       from Technical University of Madrid (UPM)
Supervisor: Luis Badesa

Pricing SCC by primal-dual formulation (single-period)
17. March. 2026
=#

import Pkg
using JuMP,Gurobi, CSV,DataFrames,LinearAlgebra, XLSX, IterTools, DelimitedFiles,Plots,MAT, Dualization
include("dataset_gene.jl")
include("offline_trainning.jl")
include("admittance_matrix_calculation.jl") 
# SGs, buses:2,3,4,5,27,30    IBRs, buses:1,23,26



#-----------------------------------Define Parameters for Calculating SCC-----------------------------------
I_IBG=1      # pre-defined SCC contribution from IBG
Iₗᵢₘ= 5       # SCC limit
β=0.95      # percentage of nominal voltage, range from 0951
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
Kˢᵗ=[200, 125, 92.5, 72, 55, 31]*10^3 /10                           #  Startup cost of SGs                     SGs, buses:2,3,4,5,27,30
Oᵐ₁=[6.20, 7.10, 10.47, 12.28, 13.53, 15.36]                        #  Marginal generation cost of SG 1  in    SGs, buses:2,3,4,5,27,30
Oᵐ₂=[7.07, 8.72, 11.49, 12.84, 14.60, 15.02]                        #  Marginal generation cost of SG 2  in    SGs, buses:2,3,4,5,27,30
Oⁿˡ=[17.431, 15.005, 13.755, 10.930, 9.900, 8.570]*10^3 /10         #  No-load cost of SGs                     SGs, buses:2,3,4,5,27,30
yˢᴳ₀=[1 1 1 0 0 0] 


Energy_price_PD = zeros(4,T)
SCC_price_PD = zeros(4,4,T)
Primal_obj_PD = zeros(4)
Dual_obj_PD = zeros(4)
t_record_PD = zeros(4)
#N = 5                   # Number of binary variables for binary expansion of Pˢᴳ variables
for N in 4:7
        N = 20
#-----------------------------------Define Primal-Dual Model-----------------------------------
model= Model()
#-------Define Primal Variales
 
@variable(model, Pˢᴳ²_1[1:T])        
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

@variable(model, Pᴵᴮᴳ¹[1:T]>=0)         
@variable(model, Pᴵᴮᴳ²³[1:T]>=0)              
@variable(model, Pᴵᴮᴳ²⁶[1:T]>=0)  

@variable(model, yˢᴳ²_1[1:T] ,Bin )     #,Bin    >=0
@variable(model, yˢᴳ²_2[1:T] ,Bin )          
@variable(model, yˢᴳ³_1[1:T] ,Bin )
@variable(model, yˢᴳ³_2[1:T] ,Bin )         
@variable(model, yˢᴳ⁴_1[1:T] ,Bin )      
@variable(model, yˢᴳ⁴_2[1:T] ,Bin )      
@variable(model, yˢᴳ⁵_1[1:T] ,Bin )      
@variable(model, yˢᴳ⁵_2[1:T] ,Bin )      
@variable(model, yˢᴳ²⁷_1[1:T] ,Bin )  
@variable(model, yˢᴳ²⁷_2[1:T] ,Bin )          
@variable(model, yˢᴳ³⁰_1[1:T] ,Bin )   
@variable(model, yˢᴳ³⁰_2[1:T] ,Bin )  

@variable(model, Cᵁ²_1[1:T]>=0)                 
@variable(model, Cᵁ²_2[1:T]>=0)                                
@variable(model, Cᵁ³_1[1:T]>=0)    
@variable(model, Cᵁ³_2[1:T]>=0)                           
@variable(model, Cᵁ⁴_1[1:T]>=0)     
@variable(model, Cᵁ⁴_2[1:T]>=0)                        
@variable(model, Cᵁ⁵_1[1:T]>=0)  
@variable(model, Cᵁ⁵_2[1:T]>=0)                              
@variable(model, Cᵁ²⁷_1[1:T]>=0)    
@variable(model, Cᵁ²⁷_2[1:T]>=0)                          
@variable(model, Cᵁ³⁰_1[1:T]>=0)      
@variable(model, Cᵁ³⁰_2[1:T]>=0)               

@variable(model, ηₘ[1:66,1:T]>=0)         

@variable(model, I_₁₁[1:T])             
@variable(model, I_₂₆[1:T])
@variable(model, I_₂₉[1:T])
@variable(model, I_₃₀[1:T])

#-------Define Dual Variales
@variable(model, λᴱ[1:T])                
@variable(model, λ_F[1:4,1:T]>=0)       

@variable(model, ζᵐᵃˣ¹[1:T]>=0) 
@variable(model, ζᵐᵃˣ²³[1:T]>=0)
@variable(model, ζᵐᵃˣ²⁶[1:T]>=0)

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

@variable(model, σˢᵗˢᴳ²_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ²_2[1:T]>=0)
@variable(model, σˢᵗˢᴳ³_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ³_2[1:T]>=0)
@variable(model, σˢᵗˢᴳ⁴_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ⁴_2[1:T]>=0)
@variable(model, σˢᵗˢᴳ⁵_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ⁵_2[1:T]>=0)
@variable(model, σˢᵗˢᴳ²⁷_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ²⁷_2[1:T]>=0)
@variable(model, σˢᵗˢᴳ³⁰_1[1:T]>=0)
@variable(model, σˢᵗˢᴳ³⁰_2[1:T]>=0)

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

@variable(model, γ_max[1:66,1:2,1:T] >=0)
@variable(model, γ_min[1:66,1:T] >=0)


#-------Define Primal Constraints
#=
@constraint(model, yˢᴳ²_1.<=1 )
@constraint(model, yˢᴳ²_2.<=1 )
@constraint(model, yˢᴳ³_1.<=1 )
@constraint(model, yˢᴳ³_2.<=1 )
@constraint(model, yˢᴳ⁴_1.<=1 )
@constraint(model, yˢᴳ⁴_2.<=1 )
@constraint(model, yˢᴳ⁵_1.<=1 )
@constraint(model, yˢᴳ⁵_2.<=1 )
@constraint(model, yˢᴳ²⁷_1.<=1 )
@constraint(model, yˢᴳ²⁷_2.<=1 )
@constraint(model, yˢᴳ³⁰_1.<=1 )
@constraint(model, yˢᴳ³⁰_2.<=1 ) 
=#
for t in 1:T
    @constraint( model, Pˢᴳ²_1[t]+Pˢᴳ²_2[t]+Pˢᴳ³_1[t]+Pˢᴳ³_2[t]+Pˢᴳ⁴_1[t]+Pˢᴳ⁴_2[t]+Pˢᴳ⁵_1[t]+Pˢᴳ⁵_2[t]+Pˢᴳ²⁷_1[t]+Pˢᴳ²⁷_2[t]+Pˢᴳ³⁰_1[t]+Pˢᴳ³⁰_2[t]+
                   Pᴵᴮᴳ¹[t]+Pᴵᴮᴳ²³[t]+Pᴵᴮᴳ²⁶[t] == Load_total[t] )     # power balance , dual variable: λᴱₜ
end

@constraint(model, Pˢᴳ²_1.<=yˢᴳ²_1*Pˢᴳₘₐₓ[1])           # #= =# bounds for the output of SGs with UC , dual variables: μᵐⁱⁿₜ , μᵐᵃˣₜ
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

@constraint(model, Cᵁ²_1[1]>=(yˢᴳ²_1[1]-yˢᴳ₀[1])*Kˢᵗ[1])      
@constraint(model, Cᵁ²_2[1]>=(yˢᴳ²_2[1]-yˢᴳ₀[1])*Kˢᵗ[1])               
@constraint(model, Cᵁ³_1[1]>=(yˢᴳ³_1[1]-yˢᴳ₀[2])*Kˢᵗ[2])      
@constraint(model, Cᵁ³_2[1]>=(yˢᴳ³_2[1]-yˢᴳ₀[2])*Kˢᵗ[2])      
@constraint(model, Cᵁ⁴_1[1]>=(yˢᴳ⁴_1[1]-yˢᴳ₀[3])*Kˢᵗ[3])      
@constraint(model, Cᵁ⁴_2[1]>=(yˢᴳ⁴_2[1]-yˢᴳ₀[3])*Kˢᵗ[3])      
@constraint(model, Cᵁ⁵_1[1]>=(yˢᴳ⁵_1[1]-yˢᴳ₀[4])*Kˢᵗ[4])     
@constraint(model, Cᵁ⁵_2[1]>=(yˢᴳ⁵_2[1]-yˢᴳ₀[4])*Kˢᵗ[4])      
@constraint(model, Cᵁ²⁷_1[1]>=(yˢᴳ²⁷_1[1]-yˢᴳ₀[5])*Kˢᵗ[5])    
@constraint(model, Cᵁ²⁷_2[1]>=(yˢᴳ²⁷_2[1]-yˢᴳ₀[5])*Kˢᵗ[5])    
@constraint(model, Cᵁ³⁰_1[1]>=(yˢᴳ³⁰_1[1]-yˢᴳ₀[6])*Kˢᵗ[6])    
@constraint(model, Cᵁ³⁰_2[1]>=(yˢᴳ³⁰_2[1]-yˢᴳ₀[6])*Kˢᵗ[6])    

for t in 2:T
    @constraint(model, Cᵁ²_1[t]>=(yˢᴳ²_1[t]-yˢᴳ²_1[t-1])*Kˢᵗ[1])         
    @constraint(model, Cᵁ²_2[t]>=(yˢᴳ²_2[t]-yˢᴳ²_2[t-1])*Kˢᵗ[1])         

    @constraint(model, Cᵁ³_1[t]>=(yˢᴳ³_1[t]-yˢᴳ³_1[t-1])*Kˢᵗ[2]) 
    @constraint(model, Cᵁ³_2[t]>=(yˢᴳ³_2[t]-yˢᴳ³_2[t-1])*Kˢᵗ[2])

    @constraint(model, Cᵁ⁴_1[t]>=(yˢᴳ⁴_1[t]-yˢᴳ⁴_1[t-1])*Kˢᵗ[3])
    @constraint(model, Cᵁ⁴_2[t]>=(yˢᴳ⁴_2[t]-yˢᴳ⁴_2[t-1])*Kˢᵗ[3])

    @constraint(model, Cᵁ⁵_1[t]>=(yˢᴳ⁵_1[t]-yˢᴳ⁵_1[t-1])*Kˢᵗ[4])
    @constraint(model, Cᵁ⁵_2[t]>=(yˢᴳ⁵_2[t]-yˢᴳ⁵_2[t-1])*Kˢᵗ[4])

    @constraint(model, Cᵁ²⁷_1[t]>=(yˢᴳ²⁷_1[t]-yˢᴳ²⁷_1[t-1])*Kˢᵗ[5])
    @constraint(model, Cᵁ²⁷_2[t]>=(yˢᴳ²⁷_2[t]-yˢᴳ²⁷_2[t-1])*Kˢᵗ[5])

    @constraint(model, Cᵁ³⁰_1[t]>=(yˢᴳ³⁰_1[t]-yˢᴳ³⁰_1[t-1])*Kˢᵗ[6])
    @constraint(model, Cᵁ³⁰_2[t]>=(yˢᴳ³⁰_2[t]-yˢᴳ³⁰_2[t-1])*Kˢᵗ[6])
end
    
for t in 1:T
    @constraint(model, Pᴵᴮᴳ¹[t] <= IBG₁[t])        # wind power limit  , dual variable: ζᵐᵃˣₜ
    @constraint(model, Pᴵᴮᴳ²³[t]<= IBG₂₃[t])       
    @constraint(model, Pᴵᴮᴳ²⁶[t]<= IBG₂₆[t])       
                
end

for t in 1:T
    # yˢᴳ²_1
    @constraint(model, ηₘ[1,t]<=yˢᴳ²_1[t])
    @constraint(model, ηₘ[1,t]<=yˢᴳ²_2[t])
    @constraint(model, ηₘ[1,t]>=yˢᴳ²_1[t]+yˢᴳ²_2[t]-1)
    @constraint(model, ηₘ[2,t]<=yˢᴳ²_1[t])
    @constraint(model, ηₘ[2,t]<=yˢᴳ³_1[t])
    @constraint(model, ηₘ[2,t]>=yˢᴳ²_1[t]+yˢᴳ³_1[t]-1)
    @constraint(model, ηₘ[3,t]<=yˢᴳ²_1[t])
    @constraint(model, ηₘ[3,t]<=yˢᴳ³_2[t])
    @constraint(model, ηₘ[3,t]>=yˢᴳ²_1[t]+yˢᴳ³_2[t]-1)
    @constraint(model, ηₘ[4,t]<=yˢᴳ²_1[t])
    @constraint(model, ηₘ[4,t]<=yˢᴳ⁴_1[t])
    @constraint(model, ηₘ[4,t]>=yˢᴳ²_1[t]+yˢᴳ⁴_1[t]-1)
    @constraint(model, ηₘ[5,t]<=yˢᴳ²_1[t])
    @constraint(model, ηₘ[5,t]<=yˢᴳ⁴_2[t])
    @constraint(model, ηₘ[5,t]>=yˢᴳ²_1[t]+yˢᴳ⁴_2[t]-1)
    @constraint(model, ηₘ[6,t]<=yˢᴳ²_1[t])
    @constraint(model, ηₘ[6,t]<=yˢᴳ⁵_1[t])
    @constraint(model, ηₘ[6,t]>=yˢᴳ²_1[t]+yˢᴳ⁵_1[t]-1)
    @constraint(model, ηₘ[7,t]<=yˢᴳ²_1[t])
    @constraint(model, ηₘ[7,t]<=yˢᴳ⁵_2[t])
    @constraint(model, ηₘ[7,t]>=yˢᴳ²_1[t]+yˢᴳ⁵_2[t]-1)
    @constraint(model, ηₘ[8,t]<=yˢᴳ²_1[t])
    @constraint(model, ηₘ[8,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, ηₘ[8,t]>=yˢᴳ²_1[t]+yˢᴳ²⁷_1[t]-1)
    @constraint(model, ηₘ[9,t]<=yˢᴳ²_1[t])
    @constraint(model, ηₘ[9,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, ηₘ[9,t]>=yˢᴳ²_1[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, ηₘ[10,t]<=yˢᴳ²_1[t])
    @constraint(model, ηₘ[10,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, ηₘ[10,t]>=yˢᴳ²_1[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, ηₘ[11,t]<=yˢᴳ²_1[t])
    @constraint(model, ηₘ[11,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, ηₘ[11,t]>=yˢᴳ²_1[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ²_2
    @constraint(model, ηₘ[12,t]<=yˢᴳ²_2[t])
    @constraint(model, ηₘ[12,t]<=yˢᴳ³_1[t])
    @constraint(model, ηₘ[12,t]>=yˢᴳ²_2[t]+yˢᴳ³_1[t]-1)
    @constraint(model, ηₘ[13,t]<=yˢᴳ²_2[t])
    @constraint(model, ηₘ[13,t]<=yˢᴳ³_2[t])
    @constraint(model, ηₘ[13,t]>=yˢᴳ²_2[t]+yˢᴳ³_2[t]-1)
    @constraint(model, ηₘ[14,t]<=yˢᴳ²_2[t])
    @constraint(model, ηₘ[14,t]<=yˢᴳ⁴_1[t])
    @constraint(model, ηₘ[14,t]>=yˢᴳ²_2[t]+yˢᴳ⁴_1[t]-1)
    @constraint(model, ηₘ[15,t]<=yˢᴳ²_2[t])
    @constraint(model, ηₘ[15,t]<=yˢᴳ⁴_2[t])
    @constraint(model, ηₘ[15,t]>=yˢᴳ²_2[t]+yˢᴳ⁴_2[t]-1)
    @constraint(model, ηₘ[16,t]<=yˢᴳ²_2[t])
    @constraint(model, ηₘ[16,t]<=yˢᴳ⁵_1[t])
    @constraint(model, ηₘ[16,t]>=yˢᴳ²_2[t]+yˢᴳ⁵_1[t]-1)
    @constraint(model, ηₘ[17,t]<=yˢᴳ²_2[t])
    @constraint(model, ηₘ[17,t]<=yˢᴳ⁵_2[t])
    @constraint(model, ηₘ[17,t]>=yˢᴳ²_2[t]+yˢᴳ⁵_2[t]-1)
    @constraint(model, ηₘ[18,t]<=yˢᴳ²_2[t])
    @constraint(model, ηₘ[18,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, ηₘ[18,t]>=yˢᴳ²_2[t]+yˢᴳ²⁷_1[t]-1)
    @constraint(model, ηₘ[19,t]<=yˢᴳ²_2[t])
    @constraint(model, ηₘ[19,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, ηₘ[19,t]>=yˢᴳ²_2[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, ηₘ[20,t]<=yˢᴳ²_2[t])
    @constraint(model, ηₘ[20,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, ηₘ[20,t]>=yˢᴳ²_2[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, ηₘ[21,t]<=yˢᴳ²_2[t])
    @constraint(model, ηₘ[21,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, ηₘ[21,t]>=yˢᴳ²_2[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ³_1
    @constraint(model, ηₘ[22,t]<=yˢᴳ³_1[t])
    @constraint(model, ηₘ[22,t]<=yˢᴳ³_2[t])
    @constraint(model, ηₘ[22,t]>=yˢᴳ³_1[t]+yˢᴳ³_2[t]-1)
    @constraint(model, ηₘ[23,t]<=yˢᴳ³_1[t])
    @constraint(model, ηₘ[23,t]<=yˢᴳ⁴_1[t])
    @constraint(model, ηₘ[23,t]>=yˢᴳ³_1[t]+yˢᴳ⁴_1[t]-1)
    @constraint(model, ηₘ[24,t]<=yˢᴳ³_1[t])
    @constraint(model, ηₘ[24,t]<=yˢᴳ⁴_2[t])
    @constraint(model, ηₘ[24,t]>=yˢᴳ³_1[t]+yˢᴳ⁴_2[t]-1)
    @constraint(model, ηₘ[25,t]<=yˢᴳ³_1[t])
    @constraint(model, ηₘ[25,t]<=yˢᴳ⁵_1[t])
    @constraint(model, ηₘ[25,t]>=yˢᴳ³_1[t]+yˢᴳ⁵_1[t]-1)
    @constraint(model, ηₘ[26,t]<=yˢᴳ³_1[t])
    @constraint(model, ηₘ[26,t]<=yˢᴳ⁵_2[t])
    @constraint(model, ηₘ[26,t]>=yˢᴳ³_1[t]+yˢᴳ⁵_2[t]-1)
    @constraint(model, ηₘ[27,t]<=yˢᴳ³_1[t])
    @constraint(model, ηₘ[27,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, ηₘ[27,t]>=yˢᴳ³_1[t]+yˢᴳ²⁷_1[t]-1)
    @constraint(model, ηₘ[28,t]<=yˢᴳ³_1[t])
    @constraint(model, ηₘ[28,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, ηₘ[28,t]>=yˢᴳ³_1[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, ηₘ[29,t]<=yˢᴳ³_1[t])
    @constraint(model, ηₘ[29,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, ηₘ[29,t]>=yˢᴳ³_1[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, ηₘ[30,t]<=yˢᴳ³_1[t])
    @constraint(model, ηₘ[30,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, ηₘ[30,t]>=yˢᴳ³_1[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ³_2
    @constraint(model, ηₘ[31,t]<=yˢᴳ³_2[t])
    @constraint(model, ηₘ[31,t]<=yˢᴳ⁴_1[t])
    @constraint(model, ηₘ[31,t]>=yˢᴳ³_2[t]+yˢᴳ⁴_1[t]-1)
    @constraint(model, ηₘ[32,t]<=yˢᴳ³_2[t])
    @constraint(model, ηₘ[32,t]<=yˢᴳ⁴_2[t])
    @constraint(model, ηₘ[32,t]>=yˢᴳ³_2[t]+yˢᴳ⁴_2[t]-1)
    @constraint(model, ηₘ[33,t]<=yˢᴳ³_2[t])
    @constraint(model, ηₘ[33,t]<=yˢᴳ⁵_1[t])
    @constraint(model, ηₘ[33,t]>=yˢᴳ³_2[t]+yˢᴳ⁵_1[t]-1)
    @constraint(model, ηₘ[34,t]<=yˢᴳ³_2[t])
    @constraint(model, ηₘ[34,t]<=yˢᴳ⁵_2[t])
    @constraint(model, ηₘ[34,t]>=yˢᴳ³_2[t]+yˢᴳ⁵_2[t]-1)
    @constraint(model, ηₘ[35,t]<=yˢᴳ³_2[t])
    @constraint(model, ηₘ[35,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, ηₘ[35,t]>=yˢᴳ³_2[t]+yˢᴳ²⁷_1[t]-1)
    @constraint(model, ηₘ[36,t]<=yˢᴳ³_2[t])
    @constraint(model, ηₘ[36,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, ηₘ[36,t]>=yˢᴳ³_2[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, ηₘ[37,t]<=yˢᴳ³_2[t])
    @constraint(model, ηₘ[37,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, ηₘ[37,t]>=yˢᴳ³_2[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, ηₘ[38,t]<=yˢᴳ³_2[t])
    @constraint(model, ηₘ[38,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, ηₘ[38,t]>=yˢᴳ³_2[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ⁴_1
    @constraint(model, ηₘ[39,t]<=yˢᴳ⁴_1[t])
    @constraint(model, ηₘ[39,t]<=yˢᴳ⁴_2[t])
    @constraint(model, ηₘ[39,t]>=yˢᴳ⁴_1[t]+yˢᴳ⁴_2[t]-1)
    @constraint(model, ηₘ[40,t]<=yˢᴳ⁴_1[t])
    @constraint(model, ηₘ[40,t]<=yˢᴳ⁵_1[t])
    @constraint(model, ηₘ[40,t]>=yˢᴳ⁴_1[t]+yˢᴳ⁵_1[t]-1)
    @constraint(model, ηₘ[41,t]<=yˢᴳ⁴_1[t])
    @constraint(model, ηₘ[41,t]<=yˢᴳ⁵_2[t])
    @constraint(model, ηₘ[41,t]>=yˢᴳ⁴_1[t]+yˢᴳ⁵_2[t]-1)
    @constraint(model, ηₘ[42,t]<=yˢᴳ⁴_1[t])
    @constraint(model, ηₘ[42,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, ηₘ[42,t]>=yˢᴳ⁴_1[t]+yˢᴳ²⁷_1[t]-1)
    @constraint(model, ηₘ[43,t]<=yˢᴳ⁴_1[t])
    @constraint(model, ηₘ[43,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, ηₘ[43,t]>=yˢᴳ⁴_1[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, ηₘ[44,t]<=yˢᴳ⁴_1[t])
    @constraint(model, ηₘ[44,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, ηₘ[44,t]>=yˢᴳ⁴_1[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, ηₘ[45,t]<=yˢᴳ⁴_1[t])
    @constraint(model, ηₘ[45,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, ηₘ[45,t]>=yˢᴳ⁴_1[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ⁴_2
    @constraint(model, ηₘ[46,t]<=yˢᴳ⁴_2[t])
    @constraint(model, ηₘ[46,t]<=yˢᴳ⁵_1[t])
    @constraint(model, ηₘ[46,t]>=yˢᴳ⁴_2[t]+yˢᴳ⁵_1[t]-1)
    @constraint(model, ηₘ[47,t]<=yˢᴳ⁴_2[t])
    @constraint(model, ηₘ[47,t]<=yˢᴳ⁵_2[t])
    @constraint(model, ηₘ[47,t]>=yˢᴳ⁴_2[t]+yˢᴳ⁵_2[t]-1)
    @constraint(model, ηₘ[48,t]<=yˢᴳ⁴_2[t])
    @constraint(model, ηₘ[48,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, ηₘ[48,t]>=yˢᴳ⁴_2[t]+yˢᴳ²⁷_1[t]-1)
    @constraint(model, ηₘ[49,t]<=yˢᴳ⁴_2[t])
    @constraint(model, ηₘ[49,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, ηₘ[49,t]>=yˢᴳ⁴_2[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, ηₘ[50,t]<=yˢᴳ⁴_2[t])
    @constraint(model, ηₘ[50,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, ηₘ[50,t]>=yˢᴳ⁴_2[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, ηₘ[51,t]<=yˢᴳ⁴_2[t])
    @constraint(model, ηₘ[51,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, ηₘ[51,t]>=yˢᴳ⁴_2[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ⁵_1
    @constraint(model, ηₘ[52,t]<=yˢᴳ⁵_1[t])
    @constraint(model, ηₘ[52,t]<=yˢᴳ⁵_2[t])
    @constraint(model, ηₘ[52,t]>=yˢᴳ⁵_1[t]+yˢᴳ⁵_2[t]-1)
    @constraint(model, ηₘ[53,t]<=yˢᴳ⁵_1[t])
    @constraint(model, ηₘ[53,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, ηₘ[53,t]>=yˢᴳ⁵_1[t]+yˢᴳ²⁷_1[t]-1)
    @constraint(model, ηₘ[54,t]<=yˢᴳ⁵_1[t])
    @constraint(model, ηₘ[54,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, ηₘ[54,t]>=yˢᴳ⁵_1[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, ηₘ[55,t]<=yˢᴳ⁵_1[t])
    @constraint(model, ηₘ[55,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, ηₘ[55,t]>=yˢᴳ⁵_1[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, ηₘ[56,t]<=yˢᴳ⁵_1[t])
    @constraint(model, ηₘ[56,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, ηₘ[56,t]>=yˢᴳ⁵_1[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ⁵_2
    @constraint(model, ηₘ[57,t]<=yˢᴳ⁵_2[t])
    @constraint(model, ηₘ[57,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, ηₘ[57,t]>=yˢᴳ⁵_2[t]+yˢᴳ²⁷_1[t]-1)
    @constraint(model, ηₘ[58,t]<=yˢᴳ⁵_2[t])
    @constraint(model, ηₘ[58,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, ηₘ[58,t]>=yˢᴳ⁵_2[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, ηₘ[59,t]<=yˢᴳ⁵_2[t])
    @constraint(model, ηₘ[59,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, ηₘ[59,t]>=yˢᴳ⁵_2[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, ηₘ[60,t]<=yˢᴳ⁵_2[t])
    @constraint(model, ηₘ[60,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, ηₘ[60,t]>=yˢᴳ⁵_2[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ²⁷_1
    @constraint(model, ηₘ[61,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, ηₘ[61,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, ηₘ[61,t]>=yˢᴳ²⁷_1[t]+yˢᴳ²⁷_2[t]-1)
    @constraint(model, ηₘ[62,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, ηₘ[62,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, ηₘ[62,t]>=yˢᴳ²⁷_1[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, ηₘ[63,t]<=yˢᴳ²⁷_1[t])
    @constraint(model, ηₘ[63,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, ηₘ[63,t]>=yˢᴳ²⁷_1[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ²⁷_2
    @constraint(model, ηₘ[64,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, ηₘ[64,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, ηₘ[64,t]>=yˢᴳ²⁷_2[t]+yˢᴳ³⁰_1[t]-1)
    @constraint(model, ηₘ[65,t]<=yˢᴳ²⁷_2[t])
    @constraint(model, ηₘ[65,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, ηₘ[65,t]>=yˢᴳ²⁷_2[t]+yˢᴳ³⁰_2[t]-1)
    # yˢᴳ³⁰_1
    @constraint(model, ηₘ[66,t]<=yˢᴳ³⁰_1[t])
    @constraint(model, ηₘ[66,t]<=yˢᴳ³⁰_2[t])
    @constraint(model, ηₘ[66,t]>=yˢᴳ³⁰_1[t]+yˢᴳ³⁰_2[t]-1)
end

k=11         
for t in 1:T                                      
        @constraint(model,                  
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+

        K_c[1,k]+ K_c[2,k]+ K_c[3,k]+

        K_m[1,k]*ηₘ[1,t] +K_m[2,k]*ηₘ[2,t] +K_m[3,k]*ηₘ[3,t] +K_m[4,k]*ηₘ[4,t]+
        K_m[5,k]*ηₘ[5,t] +K_m[6,k]*ηₘ[6,t] +K_m[7,k]*ηₘ[7,t] +K_m[8,k]*ηₘ[8,t]+
        K_m[9,k]*ηₘ[9,t] +K_m[10,k]*ηₘ[10,t] +K_m[11,k]*ηₘ[11,t]+

        K_m[12,k]*ηₘ[12,t] +K_m[13,k]*ηₘ[13,t] +K_m[14,k]*ηₘ[14,t]+
        K_m[15,k]*ηₘ[15,t] +K_m[16,k]*ηₘ[16,t] +K_m[17,k]*ηₘ[17,t] +K_m[18,k]*ηₘ[18,t]+
        K_m[19,k]*ηₘ[19,t] +K_m[20,k]*ηₘ[20,t] +K_m[21,k]*ηₘ[21,t]+

        K_m[22,k]*ηₘ[22,t] +K_m[23,k]*ηₘ[23,t]+
        K_m[24,k]*ηₘ[24,t] +K_m[25,k]*ηₘ[25,t] +K_m[26,k]*ηₘ[26,t] +K_m[27,k]*ηₘ[27,t]+
        K_m[28,k]*ηₘ[28,t] +K_m[29,k]*ηₘ[29,t] +K_m[30,k]*ηₘ[30,t]+

        K_m[31,k]*ηₘ[31,t]+
        K_m[32,k]*ηₘ[32,t] +K_m[33,k]*ηₘ[33,t] +K_m[34,k]*ηₘ[34,t] +K_m[35,k]*ηₘ[35,t]+
        K_m[36,k]*ηₘ[36,t] +K_m[37,k]*ηₘ[37,t] +K_m[38,k]*ηₘ[38,t]+

        K_m[39,k]*ηₘ[39,t] +K_m[40,k]*ηₘ[40,t] +K_m[41,k]*ηₘ[41,t] +K_m[42,k]*ηₘ[42,t]+
        K_m[43,k]*ηₘ[43,t] +K_m[44,k]*ηₘ[44,t] +K_m[45,k]*ηₘ[45,t]+

        K_m[46,k]*ηₘ[46,t] +K_m[47,k]*ηₘ[47,t] +K_m[48,k]*ηₘ[48,t]+
        K_m[49,k]*ηₘ[49,t] +K_m[50,k]*ηₘ[50,t] +K_m[51,k]*ηₘ[51,t]+

        K_m[52,k]*ηₘ[52,t] +K_m[53,k]*ηₘ[53,t]+
        K_m[54,k]*ηₘ[54,t] +K_m[55,k]*ηₘ[55,t] +K_m[56,k]*ηₘ[56,t]+

        K_m[57,k]*ηₘ[57,t]+
        K_m[58,k]*ηₘ[58,t] +K_m[59,k]*ηₘ[59,t] +K_m[60,k]*ηₘ[60,t]+

        K_m[61,k]*ηₘ[61,t] +K_m[62,k]*ηₘ[62,t] +K_m[63,k]*ηₘ[63,t]+

        K_m[64,k]*ηₘ[64,t] +K_m[65,k]*ηₘ[65,t]+

        K_m[66,k]*ηₘ[66,t] >=Iₗᵢₘ)        
end              

k=26                                           
 for t in 1:T                                      
        @constraint(model,                  
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+

        K_c[1,k]+ K_c[2,k]+ K_c[3,k]+

        K_m[1,k]*ηₘ[1,t] +K_m[2,k]*ηₘ[2,t] +K_m[3,k]*ηₘ[3,t] +K_m[4,k]*ηₘ[4,t]+
        K_m[5,k]*ηₘ[5,t] +K_m[6,k]*ηₘ[6,t] +K_m[7,k]*ηₘ[7,t] +K_m[8,k]*ηₘ[8,t]+
        K_m[9,k]*ηₘ[9,t] +K_m[10,k]*ηₘ[10,t] +K_m[11,k]*ηₘ[11,t]+

        K_m[12,k]*ηₘ[12,t] +K_m[13,k]*ηₘ[13,t] +K_m[14,k]*ηₘ[14,t]+
        K_m[15,k]*ηₘ[15,t] +K_m[16,k]*ηₘ[16,t] +K_m[17,k]*ηₘ[17,t] +K_m[18,k]*ηₘ[18,t]+
        K_m[19,k]*ηₘ[19,t] +K_m[20,k]*ηₘ[20,t] +K_m[21,k]*ηₘ[21,t]+

        K_m[22,k]*ηₘ[22,t] +K_m[23,k]*ηₘ[23,t]+
        K_m[24,k]*ηₘ[24,t] +K_m[25,k]*ηₘ[25,t] +K_m[26,k]*ηₘ[26,t] +K_m[27,k]*ηₘ[27,t]+
        K_m[28,k]*ηₘ[28,t] +K_m[29,k]*ηₘ[29,t] +K_m[30,k]*ηₘ[30,t]+

        K_m[31,k]*ηₘ[31,t]+
        K_m[32,k]*ηₘ[32,t] +K_m[33,k]*ηₘ[33,t] +K_m[34,k]*ηₘ[34,t] +K_m[35,k]*ηₘ[35,t]+
        K_m[36,k]*ηₘ[36,t] +K_m[37,k]*ηₘ[37,t] +K_m[38,k]*ηₘ[38,t]+

        K_m[39,k]*ηₘ[39,t] +K_m[40,k]*ηₘ[40,t] +K_m[41,k]*ηₘ[41,t] +K_m[42,k]*ηₘ[42,t]+
        K_m[43,k]*ηₘ[43,t] +K_m[44,k]*ηₘ[44,t] +K_m[45,k]*ηₘ[45,t]+

        K_m[46,k]*ηₘ[46,t] +K_m[47,k]*ηₘ[47,t] +K_m[48,k]*ηₘ[48,t]+
        K_m[49,k]*ηₘ[49,t] +K_m[50,k]*ηₘ[50,t] +K_m[51,k]*ηₘ[51,t]+

        K_m[52,k]*ηₘ[52,t] +K_m[53,k]*ηₘ[53,t]+
        K_m[54,k]*ηₘ[54,t] +K_m[55,k]*ηₘ[55,t] +K_m[56,k]*ηₘ[56,t]+

        K_m[57,k]*ηₘ[57,t]+
        K_m[58,k]*ηₘ[58,t] +K_m[59,k]*ηₘ[59,t] +K_m[60,k]*ηₘ[60,t]+

        K_m[61,k]*ηₘ[61,t] +K_m[62,k]*ηₘ[62,t] +K_m[63,k]*ηₘ[63,t]+

        K_m[64,k]*ηₘ[64,t] +K_m[65,k]*ηₘ[65,t]+

        K_m[66,k]*ηₘ[66,t] >=Iₗᵢₘ)        
end


k=29                                            
 for t in 1:T                                      
        @constraint(model,                  
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+

        K_c[1,k]+ K_c[2,k]+ K_c[3,k]+

        K_m[1,k]*ηₘ[1,t] +K_m[2,k]*ηₘ[2,t] +K_m[3,k]*ηₘ[3,t] +K_m[4,k]*ηₘ[4,t]+
        K_m[5,k]*ηₘ[5,t] +K_m[6,k]*ηₘ[6,t] +K_m[7,k]*ηₘ[7,t] +K_m[8,k]*ηₘ[8,t]+
        K_m[9,k]*ηₘ[9,t] +K_m[10,k]*ηₘ[10,t] +K_m[11,k]*ηₘ[11,t]+

        K_m[12,k]*ηₘ[12,t] +K_m[13,k]*ηₘ[13,t] +K_m[14,k]*ηₘ[14,t]+
        K_m[15,k]*ηₘ[15,t] +K_m[16,k]*ηₘ[16,t] +K_m[17,k]*ηₘ[17,t] +K_m[18,k]*ηₘ[18,t]+
        K_m[19,k]*ηₘ[19,t] +K_m[20,k]*ηₘ[20,t] +K_m[21,k]*ηₘ[21,t]+

        K_m[22,k]*ηₘ[22,t] +K_m[23,k]*ηₘ[23,t]+
        K_m[24,k]*ηₘ[24,t] +K_m[25,k]*ηₘ[25,t] +K_m[26,k]*ηₘ[26,t] +K_m[27,k]*ηₘ[27,t]+
        K_m[28,k]*ηₘ[28,t] +K_m[29,k]*ηₘ[29,t] +K_m[30,k]*ηₘ[30,t]+

        K_m[31,k]*ηₘ[31,t]+
        K_m[32,k]*ηₘ[32,t] +K_m[33,k]*ηₘ[33,t] +K_m[34,k]*ηₘ[34,t] +K_m[35,k]*ηₘ[35,t]+
        K_m[36,k]*ηₘ[36,t] +K_m[37,k]*ηₘ[37,t] +K_m[38,k]*ηₘ[38,t]+

        K_m[39,k]*ηₘ[39,t] +K_m[40,k]*ηₘ[40,t] +K_m[41,k]*ηₘ[41,t] +K_m[42,k]*ηₘ[42,t]+
        K_m[43,k]*ηₘ[43,t] +K_m[44,k]*ηₘ[44,t] +K_m[45,k]*ηₘ[45,t]+

        K_m[46,k]*ηₘ[46,t] +K_m[47,k]*ηₘ[47,t] +K_m[48,k]*ηₘ[48,t]+
        K_m[49,k]*ηₘ[49,t] +K_m[50,k]*ηₘ[50,t] +K_m[51,k]*ηₘ[51,t]+

        K_m[52,k]*ηₘ[52,t] +K_m[53,k]*ηₘ[53,t]+
        K_m[54,k]*ηₘ[54,t] +K_m[55,k]*ηₘ[55,t] +K_m[56,k]*ηₘ[56,t]+

        K_m[57,k]*ηₘ[57,t]+
        K_m[58,k]*ηₘ[58,t] +K_m[59,k]*ηₘ[59,t] +K_m[60,k]*ηₘ[60,t]+

        K_m[61,k]*ηₘ[61,t] +K_m[62,k]*ηₘ[62,t] +K_m[63,k]*ηₘ[63,t]+

        K_m[64,k]*ηₘ[64,t] +K_m[65,k]*ηₘ[65,t]+

        K_m[66,k]*ηₘ[66,t] >=Iₗᵢₘ)        
end


k=30                                            
 for t in 1:T                                      
        @constraint(model,                  
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+

        K_c[1,k]+ K_c[2,k]+ K_c[3,k]+

        K_m[1,k]*ηₘ[1,t] +K_m[2,k]*ηₘ[2,t] +K_m[3,k]*ηₘ[3,t] +K_m[4,k]*ηₘ[4,t]+
        K_m[5,k]*ηₘ[5,t] +K_m[6,k]*ηₘ[6,t] +K_m[7,k]*ηₘ[7,t] +K_m[8,k]*ηₘ[8,t]+
        K_m[9,k]*ηₘ[9,t] +K_m[10,k]*ηₘ[10,t] +K_m[11,k]*ηₘ[11,t]+

        K_m[12,k]*ηₘ[12,t] +K_m[13,k]*ηₘ[13,t] +K_m[14,k]*ηₘ[14,t]+
        K_m[15,k]*ηₘ[15,t] +K_m[16,k]*ηₘ[16,t] +K_m[17,k]*ηₘ[17,t] +K_m[18,k]*ηₘ[18,t]+
        K_m[19,k]*ηₘ[19,t] +K_m[20,k]*ηₘ[20,t] +K_m[21,k]*ηₘ[21,t]+

        K_m[22,k]*ηₘ[22,t] +K_m[23,k]*ηₘ[23,t]+
        K_m[24,k]*ηₘ[24,t] +K_m[25,k]*ηₘ[25,t] +K_m[26,k]*ηₘ[26,t] +K_m[27,k]*ηₘ[27,t]+
        K_m[28,k]*ηₘ[28,t] +K_m[29,k]*ηₘ[29,t] +K_m[30,k]*ηₘ[30,t]+

        K_m[31,k]*ηₘ[31,t]+
        K_m[32,k]*ηₘ[32,t] +K_m[33,k]*ηₘ[33,t] +K_m[34,k]*ηₘ[34,t] +K_m[35,k]*ηₘ[35,t]+
        K_m[36,k]*ηₘ[36,t] +K_m[37,k]*ηₘ[37,t] +K_m[38,k]*ηₘ[38,t]+

        K_m[39,k]*ηₘ[39,t] +K_m[40,k]*ηₘ[40,t] +K_m[41,k]*ηₘ[41,t] +K_m[42,k]*ηₘ[42,t]+
        K_m[43,k]*ηₘ[43,t] +K_m[44,k]*ηₘ[44,t] +K_m[45,k]*ηₘ[45,t]+

        K_m[46,k]*ηₘ[46,t] +K_m[47,k]*ηₘ[47,t] +K_m[48,k]*ηₘ[48,t]+
        K_m[49,k]*ηₘ[49,t] +K_m[50,k]*ηₘ[50,t] +K_m[51,k]*ηₘ[51,t]+

        K_m[52,k]*ηₘ[52,t] +K_m[53,k]*ηₘ[53,t]+
        K_m[54,k]*ηₘ[54,t] +K_m[55,k]*ηₘ[55,t] +K_m[56,k]*ηₘ[56,t]+

        K_m[57,k]*ηₘ[57,t]+
        K_m[58,k]*ηₘ[58,t] +K_m[59,k]*ηₘ[59,t] +K_m[60,k]*ηₘ[60,t]+

        K_m[61,k]*ηₘ[61,t] +K_m[62,k]*ηₘ[62,t] +K_m[63,k]*ηₘ[63,t]+

        K_m[64,k]*ηₘ[64,t] +K_m[65,k]*ηₘ[65,t]+

        K_m[66,k]*ηₘ[66,t] >=Iₗᵢₘ)        
end




#-------Define Dual Constraints 

@constraint(model, Oⁿˡ[1]  - ( K_g[1,11]*λ_F[1,T] + K_g[1,26]*λ_F[2,T] + K_g[1,29]*λ_F[3,T] + K_g[1,30]*λ_F[4,T] ) -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[T] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[T] +Kˢᵗ[1]*σˢᵗˢᴳ²_1[T] + ψᵐᵃˣˢᴳ²_1[T] 
                            - sum(γ_max[m,1,T] for m in 1:11) + sum(γ_min[m,T] for m in 1:11) >=0 )  

@constraint(model, Oⁿˡ[1]  - ( K_g[2,11]*λ_F[1,T] + K_g[2,26]*λ_F[2,T] + K_g[2,29]*λ_F[3,T] + K_g[2,30]*λ_F[4,T] ) -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[T] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[T] +Kˢᵗ[1]*σˢᵗˢᴳ²_2[T] + ψᵐᵃˣˢᴳ²_2[T] 
                            - γ_max[1,2,T] + γ_min[1,T] 
                            - sum(γ_max[m,1,T] for m in 12:21) + sum(γ_min[m,T] for m in 12:21) >=0 )

@constraint(model, Oⁿˡ[2]  - ( K_g[3,11]*λ_F[1,T] + K_g[3,26]*λ_F[2,T] + K_g[3,29]*λ_F[3,T] + K_g[3,30]*λ_F[4,T] ) -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[T] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[T] +Kˢᵗ[2]*σˢᵗˢᴳ³_1[T] + ψᵐᵃˣˢᴳ³_1[T] 
                            -γ_max[2,2,T] +γ_min[2,T] -γ_max[12,2,T] +γ_min[12,T] 
                            - sum(γ_max[m,1,T] for m in 22:30) + sum(γ_min[m,T] for m in 22:30) >=0 )

@constraint(model, Oⁿˡ[2]  - ( K_g[4,11]*λ_F[1,T] + K_g[4,26]*λ_F[2,T] + K_g[4,29]*λ_F[3,T] + K_g[4,30]*λ_F[4,T] ) -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[T] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[T] +Kˢᵗ[2]*σˢᵗˢᴳ³_2[T] + ψᵐᵃˣˢᴳ³_2[T] 
                            -γ_max[3,2,T] +γ_min[3,T] -γ_max[13,2,T] +γ_min[13,T] -γ_max[22,2,T] +γ_min[22,T] 
                            - sum(γ_max[m,1,T] for m in 31:38) + sum(γ_min[m,T] for m in 31:38) >=0 )

@constraint(model, Oⁿˡ[3]  - ( K_g[5,11]*λ_F[1,T] + K_g[5,26]*λ_F[2,T] + K_g[5,29]*λ_F[3,T] + K_g[5,30]*λ_F[4,T] ) -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[T] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[T] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_1[T] + ψᵐᵃˣˢᴳ⁴_1[T] 
                            -γ_max[4,2,T] +γ_min[4,T] -γ_max[14,2,T] +γ_min[14,T] -γ_max[23,2,T] +γ_min[23,T] -γ_max[31,2,T] +γ_min[31,T] 
                            - sum(γ_max[m,1,T] for m in 39:45) + sum(γ_min[m,T] for m in 39:45) >=0 )

@constraint(model, Oⁿˡ[3]  - ( K_g[6,11]*λ_F[1,T] + K_g[6,26]*λ_F[2,T] + K_g[6,29]*λ_F[3,T] + K_g[6,30]*λ_F[4,T] ) -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[T] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[T] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_2[T] + ψᵐᵃˣˢᴳ⁴_2[T] 
                            -γ_max[5,2,T] +γ_min[5,T]-γ_max[15,2,T] +γ_min[15,T] -γ_max[24,2,T] +γ_min[24,T] -γ_max[32,2,T] +γ_min[32,T] -γ_max[39,2,T] +γ_min[39,T]
                            - sum(γ_max[m,1,T] for m in 46:51) + sum(γ_min[m,T] for m in 46:51) >=0 )

@constraint(model, Oⁿˡ[4]  - ( K_g[7,11]*λ_F[1,T] + K_g[7,26]*λ_F[2,T] + K_g[7,29]*λ_F[3,T] + K_g[7,30]*λ_F[4,T] ) -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[T] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[T] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_1[T] + ψᵐᵃˣˢᴳ⁵_1[T] 
                            -γ_max[6,2,T] +γ_min[6,T]-γ_max[16,2,T] +γ_min[16,T] -γ_max[25,2,T] +γ_min[25,T] -γ_max[33,2,T] +γ_min[33,T] -γ_max[40,2,T] +γ_min[40,T]
                            -γ_max[46,2,T] +γ_min[46,T] 
                            - sum(γ_max[m,1,T] for m in 52:56) + sum(γ_min[m,T] for m in 52:56) >=0 )

@constraint(model, Oⁿˡ[4]  - ( K_g[8,11]*λ_F[1,T] + K_g[8,26]*λ_F[2,T] + K_g[8,29]*λ_F[3,T] + K_g[8,30]*λ_F[4,T] ) -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[T] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[T] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_2[T] + ψᵐᵃˣˢᴳ⁵_2[T] 
                            -γ_max[7,2,T] +γ_min[7,T]-γ_max[17,2,T] +γ_min[17,T] -γ_max[26,2,T] +γ_min[26,T] -γ_max[34,2,T] +γ_min[34,T] -γ_max[41,2,T] +γ_min[41,T]
                            -γ_max[47,2,T] +γ_min[47,T] -γ_max[52,2,T] +γ_min[52,T] 
                            - sum(γ_max[m,1,T] for m in 57:60) + sum(γ_min[m,T] for m in 57:60) >=0 )

@constraint(model, Oⁿˡ[5]  - ( K_g[9,11]*λ_F[1,T] + K_g[9,26]*λ_F[2,T] + K_g[9,29]*λ_F[3,T] + K_g[9,30]*λ_F[4,T] ) -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[T] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[T] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_1[T] + ψᵐᵃˣˢᴳ²⁷_1[T] 
                            -γ_max[8,2,T] +γ_min[8,T]-γ_max[18,2,T] +γ_min[18,T] -γ_max[27,2,T] +γ_min[27,T] -γ_max[35,2,T] +γ_min[35,T] -γ_max[42,2,T] +γ_min[42,T]
                            -γ_max[48,2,T] +γ_min[48,T] -γ_max[53,2,T] +γ_min[53,T] -γ_max[57,2,T] +γ_min[57,T] 
                            - sum(γ_max[m,1,T] for m in 61:63) + sum(γ_min[m,T] for m in 61:63) >=0 )

@constraint(model, Oⁿˡ[5]  - ( K_g[10,11]*λ_F[1,T] + K_g[10,26]*λ_F[2,T] + K_g[10,29]*λ_F[3,T] + K_g[10,30]*λ_F[4,T] ) -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[T] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[T] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_2[T] + ψᵐᵃˣˢᴳ²⁷_2[T] 
                            -γ_max[9,2,T] +γ_min[9,T]-γ_max[19,2,T] +γ_min[19,T] -γ_max[28,2,T] +γ_min[28,T] -γ_max[36,2,T] +γ_min[36,T] -γ_max[43,2,T] +γ_min[43,T]
                            -γ_max[49,2,T] +γ_min[49,T] -γ_max[54,2,T] +γ_min[54,T] -γ_max[58,2,T] +γ_min[58,T] -γ_max[61,2,T] +γ_min[61,T]-γ_max[64,1,T] +γ_min[64,T] -γ_max[65,1,T] +γ_min[65,T]>=0 )

@constraint(model, Oⁿˡ[6]  - ( K_g[11,11]*λ_F[1,T] + K_g[11,26]*λ_F[2,T] + K_g[11,29]*λ_F[3,T] + K_g[11,30]*λ_F[4,T] ) -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[T] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[T] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_1[T] + ψᵐᵃˣˢᴳ³⁰_1[T] 
                            -γ_max[10,2,T] +γ_min[10,T]-γ_max[20,2,T] +γ_min[20,T] -γ_max[29,2,T] +γ_min[29,T] -γ_max[37,2,T] +γ_min[37,T] -γ_max[44,2,T] +γ_min[44,T]
                            -γ_max[50,2,T] +γ_min[50,T] -γ_max[55,2,T] +γ_min[55,T] -γ_max[59,2,T] +γ_min[59,T] -γ_max[62,2,T] +γ_min[62,T]-γ_max[64,2,T] +γ_min[64,T] -γ_max[66,1,T] +γ_min[66,T]>=0 )

@constraint(model, Oⁿˡ[6]  - ( K_g[12,11]*λ_F[1,T] + K_g[12,26]*λ_F[2,T] + K_g[12,29]*λ_F[3,T] + K_g[12,30]*λ_F[4,T] ) -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[T] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[T] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_2[T] + ψᵐᵃˣˢᴳ³⁰_2[T] 
                            -γ_max[11,2,T] +γ_min[11,T]-γ_max[21,2,T] +γ_min[21,T] -γ_max[30,2,T] +γ_min[30,T] -γ_max[38,2,T] +γ_min[38,T] -γ_max[45,2,T] +γ_min[45,T]
                            -γ_max[51,2,T] +γ_min[51,T] -γ_max[56,2,T] +γ_min[56,T] -γ_max[60,2,T] +γ_min[60,T] -γ_max[63,2,T] +γ_min[63,T]-γ_max[65,2,T] +γ_min[65,T] -γ_max[66,2,T] +γ_min[66,T]>=0 )

for t in 1:T-1     
@constraint(model, Oⁿˡ[1]  - ( K_g[1,11]*λ_F[1,t] + K_g[1,26]*λ_F[2,t] + K_g[1,29]*λ_F[3,t] + K_g[1,30]*λ_F[4,t] ) -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1[t] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1[t] +Kˢᵗ[1]*σˢᵗˢᴳ²_1[t] -Kˢᵗ[1]*σˢᵗˢᴳ²_1[t+1] + ψᵐᵃˣˢᴳ²_1[t] 
                            - sum(γ_max[m,1,t] for m in 1:11) + sum(γ_min[m,t] for m in 1:11) >=0 )  

@constraint(model, Oⁿˡ[1]  - ( K_g[2,11]*λ_F[1,t] + K_g[2,26]*λ_F[2,t] + K_g[2,29]*λ_F[3,t] + K_g[2,30]*λ_F[4,t] ) -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2[t] +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2[t] +Kˢᵗ[1]*σˢᵗˢᴳ²_2[t] -Kˢᵗ[1]*σˢᵗˢᴳ²_2[t+1] + ψᵐᵃˣˢᴳ²_2[t] 
                            - γ_max[1,2,t] + γ_min[1,t] 
                            - sum(γ_max[m,1,t] for m in 12:21) + sum(γ_min[m,t] for m in 12:21) >=0 )

@constraint(model, Oⁿˡ[2]  - ( K_g[3,11]*λ_F[1,t] + K_g[3,26]*λ_F[2,t] + K_g[3,29]*λ_F[3,t] + K_g[3,30]*λ_F[4,t] ) -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1[t] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1[t] +Kˢᵗ[2]*σˢᵗˢᴳ³_1[t] -Kˢᵗ[2]*σˢᵗˢᴳ³_1[t+1] + ψᵐᵃˣˢᴳ³_1[t] 
                            -γ_max[2,2,t] +γ_min[2,t] -γ_max[12,2,t] +γ_min[12,t] 
                            - sum(γ_max[m,1,t] for m in 22:30) + sum(γ_min[m,t] for m in 22:30) >=0 )

@constraint(model, Oⁿˡ[2]  - ( K_g[4,11]*λ_F[1,t] + K_g[4,26]*λ_F[2,t] + K_g[4,29]*λ_F[3,t] + K_g[4,30]*λ_F[4,t] ) -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2[t] +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2[t] +Kˢᵗ[2]*σˢᵗˢᴳ³_2[t] -Kˢᵗ[2]*σˢᵗˢᴳ³_2[t+1] + ψᵐᵃˣˢᴳ³_2[t] 
                            -γ_max[3,2,t] +γ_min[3,t] -γ_max[13,2,t] +γ_min[13,t] -γ_max[22,2,t] +γ_min[22,t] 
                            - sum(γ_max[m,1,t] for m in 31:38) + sum(γ_min[m,t] for m in 31:38) >=0 )

@constraint(model, Oⁿˡ[3]  - ( K_g[5,11]*λ_F[1,t] + K_g[5,26]*λ_F[2,t] + K_g[5,29]*λ_F[3,t] + K_g[5,30]*λ_F[4,t] ) -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1[t] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1[t] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_1[t] -Kˢᵗ[3]*σˢᵗˢᴳ⁴_1[t+1] + ψᵐᵃˣˢᴳ⁴_1[t] 
                            -γ_max[4,2,t] +γ_min[4,t] -γ_max[14,2,t] +γ_min[14,t] -γ_max[23,2,t] +γ_min[23,t] -γ_max[31,2,t] +γ_min[31,t] 
                            - sum(γ_max[m,1,t] for m in 39:45) + sum(γ_min[m,t] for m in 39:45) >=0 )

@constraint(model, Oⁿˡ[3]  - ( K_g[6,11]*λ_F[1,t] + K_g[6,26]*λ_F[2,t] + K_g[6,29]*λ_F[3,t] + K_g[6,30]*λ_F[4,t] ) -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2[t] +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2[t] +Kˢᵗ[3]*σˢᵗˢᴳ⁴_2[t] -Kˢᵗ[3]*σˢᵗˢᴳ⁴_2[t+1] + ψᵐᵃˣˢᴳ⁴_2[t] 
                            -γ_max[5,2,t] +γ_min[5,t]-γ_max[15,2,t] +γ_min[15,t] -γ_max[24,2,t] +γ_min[24,t] -γ_max[32,2,t] +γ_min[32,t] -γ_max[39,2,t] +γ_min[39,t]
                            - sum(γ_max[m,1,t] for m in 46:51) + sum(γ_min[m,t] for m in 46:51) >=0 )

@constraint(model, Oⁿˡ[4]  - ( K_g[7,11]*λ_F[1,t] + K_g[7,26]*λ_F[2,t] + K_g[7,29]*λ_F[3,t] + K_g[7,30]*λ_F[4,t] ) -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1[t] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1[t] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_1[t] -Kˢᵗ[4]*σˢᵗˢᴳ⁵_1[t+1] + ψᵐᵃˣˢᴳ⁵_1[t] 
                            -γ_max[6,2,t] +γ_min[6,t]-γ_max[16,2,t] +γ_min[16,t] -γ_max[25,2,t] +γ_min[25,t] -γ_max[33,2,t] +γ_min[33,t] -γ_max[40,2,t] +γ_min[40,t]
                            -γ_max[46,2,t] +γ_min[46,t] 
                            - sum(γ_max[m,1,t] for m in 52:56) + sum(γ_min[m,t] for m in 52:56) >=0 )

@constraint(model, Oⁿˡ[4]  - ( K_g[8,11]*λ_F[1,t] + K_g[8,26]*λ_F[2,t] + K_g[8,29]*λ_F[3,t] + K_g[8,30]*λ_F[4,t] ) -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2[t] +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2[t] +Kˢᵗ[4]*σˢᵗˢᴳ⁵_2[t] -Kˢᵗ[4]*σˢᵗˢᴳ⁵_2[t+1] + ψᵐᵃˣˢᴳ⁵_2[t] 
                            -γ_max[7,2,t] +γ_min[7,t]-γ_max[17,2,t] +γ_min[17,t] -γ_max[26,2,t] +γ_min[26,t] -γ_max[34,2,t] +γ_min[34,t] -γ_max[41,2,t] +γ_min[41,t]
                            -γ_max[47,2,t] +γ_min[47,t] -γ_max[52,2,t] +γ_min[52,t] 
                            - sum(γ_max[m,1,t] for m in 57:60) + sum(γ_min[m,t] for m in 57:60) >=0 )

@constraint(model, Oⁿˡ[5]  - ( K_g[9,11]*λ_F[1,t] + K_g[9,26]*λ_F[2,t] + K_g[9,29]*λ_F[3,t] + K_g[9,30]*λ_F[4,t] ) -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1[t] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1[t] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_1[t] -Kˢᵗ[5]*σˢᵗˢᴳ²⁷_1[t+1] + ψᵐᵃˣˢᴳ²⁷_1[t] 
                            -γ_max[8,2,t] +γ_min[8,t]-γ_max[18,2,t] +γ_min[18,t] -γ_max[27,2,t] +γ_min[27,t] -γ_max[35,2,t] +γ_min[35,t] -γ_max[42,2,t] +γ_min[42,t]
                            -γ_max[48,2,t] +γ_min[48,t] -γ_max[53,2,t] +γ_min[53,t] -γ_max[57,2,t] +γ_min[57,t] 
                            - sum(γ_max[m,1,t] for m in 61:63) + sum(γ_min[m,t] for m in 61:63) >=0 )

@constraint(model, Oⁿˡ[5]  - ( K_g[10,11]*λ_F[1,t] + K_g[10,26]*λ_F[2,t] + K_g[10,29]*λ_F[3,t] + K_g[10,30]*λ_F[4,t] ) -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2[t] +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2[t] +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_2[t] -Kˢᵗ[5]*σˢᵗˢᴳ²⁷_2[t+1] + ψᵐᵃˣˢᴳ²⁷_2[t] 
                            -γ_max[9,2,t] +γ_min[9,t]-γ_max[19,2,t] +γ_min[19,t] -γ_max[28,2,t] +γ_min[28,t] -γ_max[36,2,t] +γ_min[36,t] -γ_max[43,2,t] +γ_min[43,t]
                            -γ_max[49,2,t] +γ_min[49,t] -γ_max[54,2,t] +γ_min[54,t] -γ_max[58,2,t] +γ_min[58,t] -γ_max[61,2,t] +γ_min[61,t]-γ_max[64,1,t] +γ_min[64,t] -γ_max[65,1,t] +γ_min[65,t]>=0 )

@constraint(model, Oⁿˡ[6]  - ( K_g[11,11]*λ_F[1,t] + K_g[11,26]*λ_F[2,t] + K_g[11,29]*λ_F[3,t] + K_g[11,30]*λ_F[4,t] ) -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1[t] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1[t] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_1[t] -Kˢᵗ[6]*σˢᵗˢᴳ³⁰_1[t+1] + ψᵐᵃˣˢᴳ³⁰_1[t] 
                            -γ_max[10,2,t] +γ_min[10,t]-γ_max[20,2,t] +γ_min[20,t] -γ_max[29,2,t] +γ_min[29,t] -γ_max[37,2,t] +γ_min[37,t] -γ_max[44,2,t] +γ_min[44,t]
                            -γ_max[50,2,t] +γ_min[50,t] -γ_max[55,2,t] +γ_min[55,t] -γ_max[59,2,t] +γ_min[59,t] -γ_max[62,2,t] +γ_min[62,t]-γ_max[64,2,t] +γ_min[64,t] -γ_max[66,1,t] +γ_min[66,t]>=0 )

@constraint(model, Oⁿˡ[6]  - ( K_g[12,11]*λ_F[1,t] + K_g[12,26]*λ_F[2,t] + K_g[12,29]*λ_F[3,t] + K_g[12,30]*λ_F[4,t] ) -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2[t] +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2[t] +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_2[t] -Kˢᵗ[6]*σˢᵗˢᴳ³⁰_2[t+1] + ψᵐᵃˣˢᴳ³⁰_2[t] 
                            -γ_max[11,2,t] +γ_min[11,t]-γ_max[21,2,t] +γ_min[21,t] -γ_max[30,2,t] +γ_min[30,t] -γ_max[38,2,t] +γ_min[38,t] -γ_max[45,2,t] +γ_min[45,t]
                            -γ_max[51,2,t] +γ_min[51,t] -γ_max[56,2,t] +γ_min[56,t] -γ_max[60,2,t] +γ_min[60,t] -γ_max[63,2,t] +γ_min[63,t]-γ_max[65,2,t] +γ_min[65,t] -γ_max[66,2,t] +γ_min[66,t]>=0 )
end

for t in 1:T                            
        @constraint(model, Oᵐ₁[1] - λᴱ[t] +μᵐᵃˣˢᴳ²_1[t] -μᵐⁱⁿˢᴳ²_1[t] >=0 )                
        @constraint(model, Oᵐ₂[1] - λᴱ[t] +μᵐᵃˣˢᴳ²_2[t] -μᵐⁱⁿˢᴳ²_2[t] >=0 )
        @constraint(model, Oᵐ₁[2] - λᴱ[t] +μᵐᵃˣˢᴳ³_1[t] -μᵐⁱⁿˢᴳ³_1[t] >=0 )                
        @constraint(model, Oᵐ₂[2] - λᴱ[t] +μᵐᵃˣˢᴳ³_2[t] -μᵐⁱⁿˢᴳ³_2[t] >=0 )
        @constraint(model, Oᵐ₁[3] - λᴱ[t] +μᵐᵃˣˢᴳ⁴_1[t] -μᵐⁱⁿˢᴳ⁴_1[t] >=0 )
        @constraint(model, Oᵐ₂[3] - λᴱ[t] +μᵐᵃˣˢᴳ⁴_2[t] -μᵐⁱⁿˢᴳ⁴_2[t] >=0 )
        @constraint(model, Oᵐ₁[4] - λᴱ[t] +μᵐᵃˣˢᴳ⁵_1[t] -μᵐⁱⁿˢᴳ⁵_1[t] >=0 )
        @constraint(model, Oᵐ₂[4] - λᴱ[t] +μᵐᵃˣˢᴳ⁵_2[t] -μᵐⁱⁿˢᴳ⁵_2[t] >=0 )
        @constraint(model, Oᵐ₁[5] - λᴱ[t] +μᵐᵃˣˢᴳ²⁷_1[t] -μᵐⁱⁿˢᴳ²⁷_1[t] >=0 )
        @constraint(model, Oᵐ₂[5] - λᴱ[t] +μᵐᵃˣˢᴳ²⁷_2[t] -μᵐⁱⁿˢᴳ²⁷_2[t] >=0 )
        @constraint(model, Oᵐ₁[6] - λᴱ[t] +μᵐᵃˣˢᴳ³⁰_1[t] -μᵐⁱⁿˢᴳ³⁰_1[t] >=0 )
        @constraint(model, Oᵐ₂[6] - λᴱ[t] +μᵐᵃˣˢᴳ³⁰_2[t] -μᵐⁱⁿˢᴳ³⁰_2[t] >=0 )                
end

for t in 1:T
        for m in 1:66
                @constraint(model,  -( K_m[m,11]*λ_F[1,t] + K_m[m,26]*λ_F[2,t] + K_m[m,29]*λ_F[3,t] + K_m[m,30]*λ_F[4,t] )
                                + γ_max[m,1,t] + γ_max[m,2,t] - γ_min[m,t] >=0   )
        end

        @constraint(model, σˢᵗˢᴳ²_1[t]<=1)                   
        @constraint(model, σˢᵗˢᴳ²_2[t]<=1)
        @constraint(model, σˢᵗˢᴳ³_1[t]<=1)
        @constraint(model, σˢᵗˢᴳ³_2[t]<=1)
        @constraint(model, σˢᵗˢᴳ⁴_1[t]<=1)
        @constraint(model, σˢᵗˢᴳ⁴_2[t]<=1)
        @constraint(model, σˢᵗˢᴳ⁵_1[t]<=1)
        @constraint(model, σˢᵗˢᴳ⁵_2[t]<=1)
        @constraint(model, σˢᵗˢᴳ²⁷_1[t]<=1)
        @constraint(model, σˢᵗˢᴳ²⁷_2[t]<=1)
        @constraint(model, σˢᵗˢᴳ³⁰_1[t]<=1)
        @constraint(model, σˢᵗˢᴳ³⁰_2[t]<=1)

        @constraint(model,  -λᴱ[t] +ζᵐᵃˣ¹[t] >=0)
        @constraint(model,  -λᴱ[t] +ζᵐᵃˣ²³[t] >=0)
        @constraint(model,  -λᴱ[t] +ζᵐᵃˣ²⁶[t] >=0)
end




#-------Non-negative profit constraints for generators, working for both primal and dual 

M = 2 * Oᵐ₁[6]          # A sufficiently large number
ΔP = zeros(12)           # Step sizes for binary expansion of Pˢᴳ variables
ΔP[1] = ( Pˢᴳₘₐₓ[1] - Pˢᴳₘᵢₙ[1] ) / ( 2^N - 1)   # Step size for binary expansion of Pˢᴳ²_1
ΔP[2] = ( Pˢᴳₘₐₓ[1] - Pˢᴳₘᵢₙ[1] ) / ( 2^N - 1)   # Step size for binary expansion of Pˢᴳ²_2
ΔP[3] = ( Pˢᴳₘₐₓ[2] - Pˢᴳₘᵢₙ[2] ) / ( 2^N - 1)   # Step size for binary expansion of Pˢᴳ³_1
ΔP[4] = ( Pˢᴳₘₐₓ[2] - Pˢᴳₘᵢₙ[2] ) / ( 2^N - 1)   # Step size for binary expansion of Pˢᴳ³_2
ΔP[5] = ( Pˢᴳₘₐₓ[3] - Pˢᴳₘᵢₙ[3] ) / ( 2^N - 1)   # Step size for binary expansion of Pˢᴳ⁴_1
ΔP[6] = ( Pˢᴳₘₐₓ[3] - Pˢᴳₘᵢₙ[3] ) / ( 2^N - 1)   # Step size for binary expansion of Pˢᴳ⁴_2
ΔP[7] = ( Pˢᴳₘₐₓ[4] - Pˢᴳₘᵢₙ[4] ) / ( 2^N - 1)   # Step size for binary expansion of Pˢᴳ⁵_1
ΔP[8] = ( Pˢᴳₘₐₓ[4] - Pˢᴳₘᵢₙ[4] ) / ( 2^N - 1)   # Step size for binary expansion of Pˢᴳ⁵_2
ΔP[9] = ( Pˢᴳₘₐₓ[5] - Pˢᴳₘᵢₙ[5] ) / ( 2^N - 1)   # Step size for binary expansion of Pˢᴳ²⁷_1
ΔP[10] = ( Pˢᴳₘₐₓ[5] - Pˢᴳₘᵢₙ[5] ) / ( 2^N  - 1)   # Step size for binary expansion of Pˢᴳ²⁷_2
ΔP[11] = ( Pˢᴳₘₐₓ[6] - Pˢᴳₘᵢₙ[6] ) / ( 2^N - 1)   # Step size for binary expansion of Pˢᴳ³⁰_1
ΔP[12] = ( Pˢᴳₘₐₓ[6] - Pˢᴳₘᵢₙ[6] ) / ( 2^N - 1)   # Step size for binary expansion of Pˢᴳ³⁰_2

@variable(model, s[1:12,1:N,1:T], Bin)   # Auxiliary binary variables for binary expansion

@variable(model, λP[1:12,1:T] )     # Auxiliary binary variables for binary expansion

@constraint(model, λᴱ.>= 0  )           # Lower bound for energy price 
@constraint(model, λᴱ.<= M  )           # Upper bound for energy price 
@constraint(model, λ_F .<= 2*( Kˢᵗ[1] + Oⁿˡ[1] ) )               # Upper bound for SCC price of buses

for t in 1:T
#=
        @constraint(model, Pˢᴳ²_1[t] == yˢᴳ²_1[t] * (Pˢᴳₘᵢₙ[1] + ΔP[1] * sum( s[1,n,t]*2^(n-1) for n in 1:N )  ) )
        @constraint(model, Pˢᴳ²_2[t] == yˢᴳ²_2[t] * (Pˢᴳₘᵢₙ[1] + ΔP[2] * sum( s[2,n,t]*2^(n-1) for n in 1:N ) )  )
        @constraint(model, Pˢᴳ³_1[t] == yˢᴳ³_1[t] * (Pˢᴳₘᵢₙ[2] + ΔP[3] * sum( s[3,n,t]*2^(n-1) for n in 1:N ) )  )
        @constraint(model, Pˢᴳ³_2[t] == yˢᴳ³_2[t] * (Pˢᴳₘᵢₙ[2] + ΔP[4] * sum( s[4,n,t]*2^(n-1) for n in 1:N )  ) )
        @constraint(model, Pˢᴳ⁴_1[t] == yˢᴳ⁴_1[t] * (Pˢᴳₘᵢₙ[3] + ΔP[5] * sum( s[5,n,t]*2^(n-1) for n in 1:N ) )  )
        @constraint(model, Pˢᴳ⁴_2[t] == yˢᴳ⁴_2[t] * (Pˢᴳₘᵢₙ[3] + ΔP[6] * sum( s[6,n,t]*2^(n-1) for n in 1:N )  ) )
        @constraint(model, Pˢᴳ⁵_1[t] == yˢᴳ⁵_1[t] * (Pˢᴳₘᵢₙ[4] + ΔP[7] * sum( s[7,n,t]*2^(n-1) for n in 1:N ) )  )
        @constraint(model, Pˢᴳ⁵_2[t] == yˢᴳ⁵_2[t] * (Pˢᴳₘᵢₙ[4] + ΔP[8] * sum( s[8,n,t]*2^(n-1) for n in 1:N )  ) )
        @constraint(model, Pˢᴳ²⁷_1[t] == yˢᴳ²⁷_1[t] * (Pˢᴳₘᵢₙ[5] + ΔP[9] * sum( s[9,n,t]*2^(n-1) for n in 1:N )  ) )
        @constraint(model, Pˢᴳ²⁷_2[t] == yˢᴳ²⁷_2[t] * (Pˢᴳₘᵢₙ[5] + ΔP[10] * sum( s[10,n,t]*2^(n-1) for n in 1:N ) )  )
        @constraint(model, Pˢᴳ³⁰_1[t] == yˢᴳ³⁰_1[t] * (Pˢᴳₘᵢₙ[6] + ΔP[11] * sum( s[11,n,t]*2^(n-1) for n in 1:N ) )  )
        @constraint(model, Pˢᴳ³⁰_2[t] == yˢᴳ³⁰_2[t] * (Pˢᴳₘᵢₙ[6] + ΔP[12] * sum( s[12,n,t]*2^(n-1) for n in 1:N )  ) )
=#
        @constraint(model, λP[1,t] == λᴱ[t] * Pˢᴳ²_1[t]  )
        @constraint(model, λP[2,t] == λᴱ[t] * Pˢᴳ²_2[t]  )
        @constraint(model, λP[3,t] == λᴱ[t] * Pˢᴳ³_1[t]  )
        @constraint(model, λP[4,t] == λᴱ[t] * Pˢᴳ³_2[t]  )
        @constraint(model, λP[5,t] == λᴱ[t] * Pˢᴳ⁴_1[t]  )
        @constraint(model, λP[6,t] == λᴱ[t] * Pˢᴳ⁴_2[t]  )
        @constraint(model, λP[7,t] == λᴱ[t] * Pˢᴳ⁵_1[t]  )
        @constraint(model, λP[8,t] == λᴱ[t] * Pˢᴳ⁵_2[t]  )
        @constraint(model, λP[9,t] == λᴱ[t] * Pˢᴳ²⁷_1[t]  )
        @constraint(model, λP[10,t] == λᴱ[t] * Pˢᴳ²⁷_2[t]  )
        @constraint(model, λP[11,t] == λᴱ[t] * Pˢᴳ³⁰_1[t]  )
        @constraint(model, λP[12,t] == λᴱ[t] * Pˢᴳ³⁰_2[t]  )

@constraint(model,  λP[1,t] + ( K_g[1,11]*λ_F[1,t] + K_g[1,26]*λ_F[2,t] + K_g[1,29]*λ_F[3,t] + K_g[1,30]*λ_F[4,t] ) * yˢᴳ²_1[t]
                        + ( K_m[1,11]*λ_F[1,t] + K_m[1,26]*λ_F[2,t] + K_m[1,29]*λ_F[3,t] + K_m[1,30]*λ_F[4,t] ) * ηₘ[1,t]
                        + ( K_m[2,11]*λ_F[1,t] + K_m[2,26]*λ_F[2,t] + K_m[2,29]*λ_F[3,t] + K_m[2,30]*λ_F[4,t] ) * ηₘ[2,t]
                        + ( K_m[3,11]*λ_F[1,t] + K_m[3,26]*λ_F[2,t] + K_m[3,29]*λ_F[3,t] + K_m[3,30]*λ_F[4,t] ) * ηₘ[3,t]
                        + ( K_m[4,11]*λ_F[1,t] + K_m[4,26]*λ_F[2,t] + K_m[4,29]*λ_F[3,t] + K_m[4,30]*λ_F[4,t] ) * ηₘ[4,t]
                        + ( K_m[5,11]*λ_F[1,t] + K_m[5,26]*λ_F[2,t] + K_m[5,29]*λ_F[3,t] + K_m[5,30]*λ_F[4,t] ) * ηₘ[5,t]
                        + ( K_m[6,11]*λ_F[1,t] + K_m[6,26]*λ_F[2,t] + K_m[6,29]*λ_F[3,t] + K_m[6,30]*λ_F[4,t] ) * ηₘ[6,t]
                        + ( K_m[7,11]*λ_F[1,t] + K_m[7,26]*λ_F[2,t] + K_m[7,29]*λ_F[3,t] + K_m[7,30]*λ_F[4,t] ) * ηₘ[7,t]
                        + ( K_m[8,11]*λ_F[1,t] + K_m[8,26]*λ_F[2,t] + K_m[8,29]*λ_F[3,t] + K_m[8,30]*λ_F[4,t] ) * ηₘ[8,t]
                        + ( K_m[9,11]*λ_F[1,t] + K_m[9,26]*λ_F[2,t] + K_m[9,29]*λ_F[3,t] + K_m[9,30]*λ_F[4,t] ) * ηₘ[9,t]
                        + ( K_m[10,11]*λ_F[1,t] + K_m[10,26]*λ_F[2,t] + K_m[10,29]*λ_F[3,t] + K_m[10,30]*λ_F[4,t] ) * ηₘ[10,t]
                        + ( K_m[11,11]*λ_F[1,t] + K_m[11,26]*λ_F[2,t] + K_m[11,29]*λ_F[3,t] + K_m[11,30]*λ_F[4,t] ) * ηₘ[11,t]
                        - Oⁿˡ[1]*yˢᴳ²_1[t] - Oᵐ₁[1]*Pˢᴳ²_1[t] - Cᵁ²_1[t] >=0 )

@constraint(model,  λP[2,t] + ( K_g[2,11]*λ_F[1,t] + K_g[2,26]*λ_F[2,t] + K_g[2,29]*λ_F[3,t] + K_g[2,30]*λ_F[4,t] ) * yˢᴳ²_2[t]
                        + ( K_m[12,11]*λ_F[1,t] + K_m[12,26]*λ_F[2,t] + K_m[12,29]*λ_F[3,t] + K_m[12,30]*λ_F[4,t] ) * ηₘ[12,t]
                        + ( K_m[13,11]*λ_F[1,t] + K_m[13,26]*λ_F[2,t] + K_m[13,29]*λ_F[3,t] + K_m[13,30]*λ_F[4,t] ) * ηₘ[13,t]
                        + ( K_m[14,11]*λ_F[1,t] + K_m[14,26]*λ_F[2,t] + K_m[14,29]*λ_F[3,t] + K_m[14,30]*λ_F[4,t] ) * ηₘ[14,t]
                        + ( K_m[15,11]*λ_F[1,t] + K_m[15,26]*λ_F[2,t] + K_m[15,29]*λ_F[3,t] + K_m[15,30]*λ_F[4,t] ) * ηₘ[15,t]
                        + ( K_m[16,11]*λ_F[1,t] + K_m[16,26]*λ_F[2,t] + K_m[16,29]*λ_F[3,t] + K_m[16,30]*λ_F[4,t] ) * ηₘ[16,t]
                        + ( K_m[17,11]*λ_F[1,t] + K_m[17,26]*λ_F[2,t] + K_m[17,29]*λ_F[3,t] + K_m[17,30]*λ_F[4,t] ) * ηₘ[17,t]
                        + ( K_m[18,11]*λ_F[1,t] + K_m[18,26]*λ_F[2,t] + K_m[18,29]*λ_F[3,t] + K_m[18,30]*λ_F[4,t] ) * ηₘ[18,t]
                        + ( K_m[19,11]*λ_F[1,t] + K_m[19,26]*λ_F[2,t] + K_m[19,29]*λ_F[3,t] + K_m[19,30]*λ_F[4,t] ) * ηₘ[19,t]
                        + ( K_m[20,11]*λ_F[1,t] + K_m[20,26]*λ_F[2,t] + K_m[20,29]*λ_F[3,t] + K_m[20,30]*λ_F[4,t] ) * ηₘ[20,t]
                        + ( K_m[21,11]*λ_F[1,t] + K_m[21,26]*λ_F[2,t] + K_m[21,29]*λ_F[3,t] + K_m[21,30]*λ_F[4,t] ) * ηₘ[21,t]
                        + ( K_m[1,11]*λ_F[1,t] + K_m[1,26]*λ_F[2,t] + K_m[1,29]*λ_F[3,t] + K_m[1,30]*λ_F[4,t] ) * ηₘ[1,t]
                        - Oⁿˡ[1]*yˢᴳ²_2[t] - Oᵐ₂[1]*Pˢᴳ²_2[t] - Cᵁ²_2[t] >=0 )

@constraint(model,  λP[3,t] + ( K_g[3,11]*λ_F[1,t] + K_g[3,26]*λ_F[2,t] + K_g[3,29]*λ_F[3,t] + K_g[3,30]*λ_F[4,t] ) * yˢᴳ³_1[t]
                        + ( K_m[22,11]*λ_F[1,t] + K_m[22,26]*λ_F[2,t] + K_m[22,29]*λ_F[3,t] + K_m[22,30]*λ_F[4,t] ) * ηₘ[22,t]
                        + ( K_m[23,11]*λ_F[1,t] + K_m[23,26]*λ_F[2,t] + K_m[23,29]*λ_F[3,t] + K_m[23,30]*λ_F[4,t] ) * ηₘ[23,t]
                        + ( K_m[24,11]*λ_F[1,t] + K_m[24,26]*λ_F[2,t] + K_m[24,29]*λ_F[3,t] + K_m[24,30]*λ_F[4,t] ) * ηₘ[24,t]
                        + ( K_m[25,11]*λ_F[1,t] + K_m[25,26]*λ_F[2,t] + K_m[25,29]*λ_F[3,t] + K_m[25,30]*λ_F[4,t] ) * ηₘ[25,t]
                        + ( K_m[26,11]*λ_F[1,t] + K_m[26,26]*λ_F[2,t] + K_m[26,29]*λ_F[3,t] + K_m[26,30]*λ_F[4,t] ) * ηₘ[26,t]
                        + ( K_m[27,11]*λ_F[1,t] + K_m[27,26]*λ_F[2,t] + K_m[27,29]*λ_F[3,t] + K_m[27,30]*λ_F[4,t] ) * ηₘ[27,t]
                        + ( K_m[28,11]*λ_F[1,t] + K_m[28,26]*λ_F[2,t] + K_m[28,29]*λ_F[3,t] + K_m[28,30]*λ_F[4,t] ) * ηₘ[28,t]
                        + ( K_m[29,11]*λ_F[1,t] + K_m[29,26]*λ_F[2,t] + K_m[29,29]*λ_F[3,t] + K_m[29,30]*λ_F[4,t] ) * ηₘ[29,t]
                        + ( K_m[30,11]*λ_F[1,t] + K_m[30,26]*λ_F[2,t] + K_m[30,29]*λ_F[3,t] + K_m[30,30]*λ_F[4,t] ) * ηₘ[30,t]
                        + ( K_m[2,11]*λ_F[1,t]  + K_m[2,26]*λ_F[2,t] + K_m[2,29]*λ_F[3,t] + K_m[2,30]*λ_F[4,t] ) * ηₘ[2,t]
                        + ( K_m[12,11]*λ_F[1,t] + K_m[12,26]*λ_F[2,t] + K_m[12,29]*λ_F[3,t] + K_m[12,30]*λ_F[4,t] ) * ηₘ[12,t]
                        - Oⁿˡ[2]*yˢᴳ³_1[t] - Oᵐ₁[2]*Pˢᴳ³_1[t] - Cᵁ³_1[t] >=0 )
 
@constraint(model,  λP[4,t] + ( K_g[4,11]*λ_F[1,t] + K_g[4,26]*λ_F[2,t] + K_g[4,29]*λ_F[3,t] + K_g[4,30]*λ_F[4,t] ) * yˢᴳ³_2[t]
                        + ( K_m[31,11]*λ_F[1,t] + K_m[31,26]*λ_F[2,t] + K_m[31,29]*λ_F[3,t] + K_m[31,30]*λ_F[4,t] ) * ηₘ[31,t]
                        + ( K_m[32,11]*λ_F[1,t] + K_m[32,26]*λ_F[2,t] + K_m[32,29]*λ_F[3,t] + K_m[32,30]*λ_F[4,t] ) * ηₘ[32,t]
                        + ( K_m[33,11]*λ_F[1,t] + K_m[33,26]*λ_F[2,t] + K_m[33,29]*λ_F[3,t] + K_m[33,30]*λ_F[4,t] ) * ηₘ[33,t]
                        + ( K_m[34,11]*λ_F[1,t] + K_m[34,26]*λ_F[2,t] + K_m[34,29]*λ_F[3,t] + K_m[34,30]*λ_F[4,t] ) * ηₘ[34,t]
                        + ( K_m[35,11]*λ_F[1,t] + K_m[35,26]*λ_F[2,t] + K_m[35,29]*λ_F[3,t] + K_m[35,30]*λ_F[4,t] ) * ηₘ[35,t]
                        + ( K_m[36,11]*λ_F[1,t] + K_m[36,26]*λ_F[2,t] + K_m[36,29]*λ_F[3,t] + K_m[36,30]*λ_F[4,t] ) * ηₘ[36,t]
                        + ( K_m[37,11]*λ_F[1,t] + K_m[37,26]*λ_F[2,t] + K_m[37,29]*λ_F[3,t] + K_m[37,30]*λ_F[4,t] ) * ηₘ[37,t]
                        + ( K_m[38,11]*λ_F[1,t] + K_m[38,26]*λ_F[2,t] + K_m[38,29]*λ_F[3,t] + K_m[38,30]*λ_F[4,t] ) * ηₘ[38,t]
                        + ( K_m[3,11]*λ_F[1,t] + K_m[3,26]*λ_F[2,t] + K_m[3,29]*λ_F[3,t] + K_m[3,30]*λ_F[4,t] ) * ηₘ[3,t]
                        + ( K_m[13,11]*λ_F[1,t] + K_m[13,26]*λ_F[2,t] + K_m[13,29]*λ_F[3,t] + K_m[13,30]*λ_F[4,t] ) * ηₘ[13,t]
                        + ( K_m[22,11]*λ_F[1,t] + K_m[22,26]*λ_F[2,t] + K_m[22,29]*λ_F[3,t] + K_m[22,30]*λ_F[4,t] ) * ηₘ[22,t]
                        - Oⁿˡ[2]*yˢᴳ³_2[t] - Oᵐ₂[2]*Pˢᴳ³_2[t] - Cᵁ³_2[t] >=0 )

@constraint(model,  λP[5,t] + ( K_g[5,11]*λ_F[1,t] + K_g[5,26]*λ_F[2,t] + K_g[5,29]*λ_F[3,t] + K_g[5,30]*λ_F[4,t] ) * yˢᴳ⁴_1[t]
                        + ( K_m[39,11]*λ_F[1,t] + K_m[39,26]*λ_F[2,t] + K_m[39,29]*λ_F[3,t] + K_m[39,30]*λ_F[4,t] ) * ηₘ[39,t]
                        + ( K_m[40,11]*λ_F[1,t] + K_m[40,26]*λ_F[2,t] + K_m[40,29]*λ_F[3,t] + K_m[40,30]*λ_F[4,t] ) * ηₘ[40,t]
                        + ( K_m[41,11]*λ_F[1,t] + K_m[41,26]*λ_F[2,t] + K_m[41,29]*λ_F[3,t] + K_m[41,30]*λ_F[4,t] ) * ηₘ[41,t]
                        + ( K_m[42,11]*λ_F[1,t] + K_m[42,26]*λ_F[2,t] + K_m[42,29]*λ_F[3,t] + K_m[42,30]*λ_F[4,t] ) * ηₘ[42,t]
                        + ( K_m[43,11]*λ_F[1,t] + K_m[43,26]*λ_F[2,t] + K_m[43,29]*λ_F[3,t] + K_m[43,30]*λ_F[4,t] ) * ηₘ[43,t]
                        + ( K_m[44,11]*λ_F[1,t] + K_m[44,26]*λ_F[2,t] + K_m[44,29]*λ_F[3,t] + K_m[44,30]*λ_F[4,t] ) * ηₘ[44,t]
                        + ( K_m[45,11]*λ_F[1,t] + K_m[45,26]*λ_F[2,t] + K_m[45,29]*λ_F[3,t] + K_m[45,30]*λ_F[4,t] ) * ηₘ[45,t]
                        + ( K_m[4,11]*λ_F[1,t] + K_m[4,26]*λ_F[2,t] + K_m[4,29]*λ_F[3,t] + K_m[4,30]*λ_F[4,t] ) * ηₘ[4,t]
                        + ( K_m[14,11]*λ_F[1,t] + K_m[14,26]*λ_F[2,t] + K_m[14,29]*λ_F[3,t] + K_m[14,30]*λ_F[4,t] ) * ηₘ[14,t]
                        + ( K_m[23,11]*λ_F[1,t] + K_m[23,26]*λ_F[2,t] + K_m[23,29]*λ_F[3,t] + K_m[23,30]*λ_F[4,t] ) * ηₘ[23,t]
                        + ( K_m[31,11]*λ_F[1,t] + K_m[31,26]*λ_F[2,t] + K_m[31,29]*λ_F[3,t] + K_m[31,30]*λ_F[4,t] ) * ηₘ[31,t]
                        - Oⁿˡ[3]*yˢᴳ⁴_1[t] - Oᵐ₁[3]*Pˢᴳ⁴_1[t] - Cᵁ⁴_1[t] >=0 )

@constraint(model,  λP[6,t] + ( K_g[6,11]*λ_F[1,t] + K_g[6,26]*λ_F[2,t] + K_g[6,29]*λ_F[3,t] + K_g[6,30]*λ_F[4,t] ) * yˢᴳ⁴_2[t]
                        + ( K_m[46,11]*λ_F[1,t] + K_m[46,26]*λ_F[2,t] + K_m[46,29]*λ_F[3,t] + K_m[46,30]*λ_F[4,t] ) * ηₘ[46,t]
                        + ( K_m[47,11]*λ_F[1,t] + K_m[47,26]*λ_F[2,t] + K_m[47,29]*λ_F[3,t] + K_m[47,30]*λ_F[4,t] ) * ηₘ[47,t]
                        + ( K_m[48,11]*λ_F[1,t] + K_m[48,26]*λ_F[2,t] + K_m[48,29]*λ_F[3,t] + K_m[48,30]*λ_F[4,t] ) * ηₘ[48,t]
                        + ( K_m[49,11]*λ_F[1,t] + K_m[49,26]*λ_F[2,t] + K_m[49,29]*λ_F[3,t] + K_m[49,30]*λ_F[4,t] ) * ηₘ[49,t]
                        + ( K_m[50,11]*λ_F[1,t] + K_m[50,26]*λ_F[2,t] + K_m[50,29]*λ_F[3,t] + K_m[50,30]*λ_F[4,t] ) * ηₘ[50,t]
                        + ( K_m[51,11]*λ_F[1,t] + K_m[51,26]*λ_F[2,t] + K_m[51,29]*λ_F[3,t] + K_m[51,30]*λ_F[4,t] ) * ηₘ[51,t]
                        + ( K_m[5,11]*λ_F[1,t] + K_m[5,26]*λ_F[2,t] + K_m[5,29]*λ_F[3,t] + K_m[5,30]*λ_F[4,t] ) * ηₘ[5,t]
                        + ( K_m[15,11]*λ_F[1,t] + K_m[15,26]*λ_F[2,t] + K_m[15,29]*λ_F[3,t] + K_m[15,30]*λ_F[4,t] ) * ηₘ[15,t]
                        + ( K_m[24,11]*λ_F[1,t] + K_m[24,26]*λ_F[2,t] + K_m[24,29]*λ_F[3,t] + K_m[24,30]*λ_F[4,t] ) * ηₘ[24,t]
                        + ( K_m[32,11]*λ_F[1,t] + K_m[32,26]*λ_F[2,t] + K_m[32,29]*λ_F[3,t] + K_m[32,30]*λ_F[4,t] ) * ηₘ[32,t]
                        + ( K_m[39,11]*λ_F[1,t] + K_m[39,26]*λ_F[2,t] + K_m[39,29]*λ_F[3,t] + K_m[39,30]*λ_F[4,t] ) * ηₘ[39,t]
                        - Oⁿˡ[3]*yˢᴳ⁴_2[t] - Oᵐ₂[3]*Pˢᴳ⁴_2[t] - Cᵁ⁴_2[t] >=0 )

@constraint(model,  λP[7,t] + ( K_g[7,11]*λ_F[1,t] + K_g[7,26]*λ_F[2,t] + K_g[7,29]*λ_F[3,t] + K_g[7,30]*λ_F[4,t] ) * yˢᴳ⁵_1[t]
                        + ( K_m[52,11]*λ_F[1,t] + K_m[52,26]*λ_F[2,t] + K_m[52,29]*λ_F[3,t] + K_m[52,30]*λ_F[4,t] ) * ηₘ[52,t]
                        + ( K_m[53,11]*λ_F[1,t] + K_m[53,26]*λ_F[2,t] + K_m[53,29]*λ_F[3,t] + K_m[53,30]*λ_F[4,t] ) * ηₘ[53,t]
                        + ( K_m[54,11]*λ_F[1,t] + K_m[54,26]*λ_F[2,t] + K_m[54,29]*λ_F[3,t] + K_m[54,30]*λ_F[4,t] ) * ηₘ[54,t]
                        + ( K_m[55,11]*λ_F[1,t] + K_m[55,26]*λ_F[2,t] + K_m[55,29]*λ_F[3,t] + K_m[55,30]*λ_F[4,t] ) * ηₘ[55,t]
                        + ( K_m[56,11]*λ_F[1,t] + K_m[56,26]*λ_F[2,t] + K_m[56,29]*λ_F[3,t] + K_m[56,30]*λ_F[4,t] ) * ηₘ[56,t]
                        + ( K_m[6,11]*λ_F[1,t] + K_m[6,26]*λ_F[2,t] + K_m[6,29]*λ_F[3,t] + K_m[6,30]*λ_F[4,t] ) * ηₘ[6,t]
                        + ( K_m[16,11]*λ_F[1,t] + K_m[16,26]*λ_F[2,t] + K_m[16,29]*λ_F[3,t] + K_m[16,30]*λ_F[4,t] ) * ηₘ[16,t]
                        + ( K_m[25,11]*λ_F[1,t] + K_m[25,26]*λ_F[2,t] + K_m[25,29]*λ_F[3,t] + K_m[25,30]*λ_F[4,t] ) * ηₘ[25,t]
                        + ( K_m[33,11]*λ_F[1,t] + K_m[33,26]*λ_F[2,t] + K_m[33,29]*λ_F[3,t] + K_m[33,30]*λ_F[4,t] ) * ηₘ[33,t]
                        + ( K_m[40,11]*λ_F[1,t] + K_m[40,26]*λ_F[2,t] + K_m[40,29]*λ_F[3,t] + K_m[40,30]*λ_F[4,t] ) * ηₘ[40,t]
                        + ( K_m[46,11]*λ_F[1,t] + K_m[46,26]*λ_F[2,t] + K_m[46,29]*λ_F[3,t] + K_m[46,30]*λ_F[4,t] ) * ηₘ[46,t]
                        - Oⁿˡ[4]*yˢᴳ⁵_1[t] - Oᵐ₁[4]*Pˢᴳ⁵_1[t] - Cᵁ⁵_1[t] >=0 )

@constraint(model,  λP[8,t] + ( K_g[8,11]*λ_F[1,t] + K_g[8,26]*λ_F[2,t] + K_g[8,29]*λ_F[3,t] + K_g[8,30]*λ_F[4,t] ) * yˢᴳ⁵_2[t]
                        + ( K_m[57,11]*λ_F[1,t] + K_m[57,26]*λ_F[2,t] + K_m[57,29]*λ_F[3,t] + K_m[57,30]*λ_F[4,t] ) * ηₘ[57,t]
                        + ( K_m[58,11]*λ_F[1,t] + K_m[58,26]*λ_F[2,t] + K_m[58,29]*λ_F[3,t] + K_m[58,30]*λ_F[4,t] ) * ηₘ[58,t]
                        + ( K_m[59,11]*λ_F[1,t]   + K_m[59,26]*λ_F[2,t] + K_m[59,29]*λ_F[3,t] + K_m[59,30]*λ_F[4,t] ) * ηₘ[59,t]
                        + ( K_m[60,11]*λ_F[1,t]   + K_m[60,26]*λ_F[2,t] + K_m[60,29]*λ_F[3,t] + K_m[60,30]*λ_F[4,t] ) * ηₘ[60,t]
                        + ( K_m[7,11]*λ_F[1,t]   + K_m[7,26]*λ_F[2,t] + K_m[7,29]*λ_F[3,t] + K_m[7,30]*λ_F[4,t] ) * ηₘ[7,t]
                        + ( K_m[17,11]*λ_F[1,t]   + K_m[17,26]*λ_F[2,t] + K_m[17,29]*λ_F[3,t] + K_m[17,30]*λ_F[4,t] ) * ηₘ[17,t]
                        + ( K_m[26,11]*λ_F[1,t]   + K_m[26,26]*λ_F[2,t] + K_m[26,29]*λ_F[3,t] + K_m[26,30]*λ_F[4,t] ) * ηₘ[26,t]
                        + ( K_m[34,11]*λ_F[1,t]   + K_m[34,26]*λ_F[2,t] + K_m[34,29]*λ_F[3,t] + K_m[34,30]*λ_F[4,t] ) * ηₘ[34,t]
                        + ( K_m[41,11]*λ_F[1,t]   + K_m[41,26]*λ_F[2,t] + K_m[41,29]*λ_F[3,t] + K_m[41,30]*λ_F[4,t] ) * ηₘ[41,t]
                        + ( K_m[47,11]*λ_F[1,t]   + K_m[47,26]*λ_F[2,t] + K_m[47,29]*λ_F[3,t] + K_m[47,30]*λ_F[4,t] ) * ηₘ[47,t]
                        + ( K_m[52,11]*λ_F[1,t]   + K_m[52,26]*λ_F[2,t] + K_m[52,29]*λ_F[3,t] + K_m[52,30]*λ_F[4,t] ) * ηₘ[52,t]
                        - Oⁿˡ[4]*yˢᴳ⁵_2[t] - Oᵐ₂[4]*Pˢᴳ⁵_2[t] - Cᵁ⁵_2[t] >=0 )

@constraint(model,  λP[9,t] + ( K_g[9,11]*λ_F[1,t] + K_g[9,26]*λ_F[2,t] + K_g[9,29]*λ_F[3,t] + K_g[9,30]*λ_F[4,t] ) * yˢᴳ²⁷_1[t]
                        + ( K_m[61,11]*λ_F[1,t] + K_m[61,26]*λ_F[2,t] + K_m[61,29]*λ_F[3,t] + K_m[61,30]*λ_F[4,t] ) * ηₘ[61,t]
                        + ( K_m[62,11]*λ_F[1,t] + K_m[62,26]*λ_F[2,t] + K_m[62,29]*λ_F[3,t] + K_m[62,30]*λ_F[4,t] ) * ηₘ[62,t]
                        + ( K_m[63,11]*λ_F[1,t] + K_m[63,26]*λ_F[2,t] + K_m[63,29]*λ_F[3,t] + K_m[63,30]*λ_F[4,t] ) * ηₘ[63,t]
                        + ( K_m[8,11]*λ_F[1,t]   + K_m[8,26]*λ_F[2,t] + K_m[8,29]*λ_F[3,t] + K_m[8,30]*λ_F[4,t] ) * ηₘ[8,t]
                        + ( K_m[18,11]*λ_F[1,t]   + K_m[18,26]*λ_F[2,t] + K_m[18,29]*λ_F[3,t] + K_m[18,30]*λ_F[4,t] ) * ηₘ[18,t]
                        + ( K_m[27,11]*λ_F[1,t]   + K_m[27,26]*λ_F[2,t] + K_m[27,29]*λ_F[3,t] + K_m[27,30]*λ_F[4,t] ) * ηₘ[27,t]
                        + ( K_m[35,11]*λ_F[1,t]   + K_m[35,26]*λ_F[2,t] + K_m[35,29]*λ_F[3,t] + K_m[35,30]*λ_F[4,t] ) * ηₘ[35,t]
                        + ( K_m[42,11]*λ_F[1,t]   + K_m[42,26]*λ_F[2,t] + K_m[42,29]*λ_F[3,t] + K_m[42,30]*λ_F[4,t] ) * ηₘ[42,t]
                        + ( K_m[48,11]*λ_F[1,t]   + K_m[48,26]*λ_F[2,t] + K_m[48,29]*λ_F[3,t] + K_m[48,30]*λ_F[4,t] ) * ηₘ[48,t]
                        + ( K_m[53,11]*λ_F[1,t]   + K_m[53,26]*λ_F[2,t] + K_m[53,29]*λ_F[3,t] + K_m[53,30]*λ_F[4,t] ) * ηₘ[53,t]
                        + ( K_m[57,11]*λ_F[1,t]   + K_m[57,26]*λ_F[2,t] + K_m[57,29]*λ_F[3,t] + K_m[57,30]*λ_F[4,t] ) * ηₘ[57,t]
                        - Oⁿˡ[5]*yˢᴳ²⁷_1[t] - Oᵐ₁[5]*Pˢᴳ²⁷_1[t] - Cᵁ²⁷_1[t] >=0 )

@constraint(model,  λP[10,t] + ( K_g[10,11]*λ_F[1,t] + K_g[10,26]*λ_F[2,t] + K_g[10,29]*λ_F[3,t] + K_g[10,30]*λ_F[4,t] ) * yˢᴳ²⁷_2[t]
                        + ( K_m[64,11]*λ_F[1,t] + K_m[64,26]*λ_F[2,t] + K_m[64,29]*λ_F[3,t] + K_m[64,30]*λ_F[4,t] ) * ηₘ[64,t]
                        + ( K_m[65,11]*λ_F[1,t] + K_m[65,26]*λ_F[2,t] + K_m[65,29]*λ_F[3,t] + K_m[65,30]*λ_F[4,t] ) * ηₘ[65,t]
                        + ( K_m[9,11]*λ_F[1,t]   + K_m[9,26]*λ_F[2,t] + K_m[9,29]*λ_F[3,t] + K_m[9,30]*λ_F[4,t] ) * ηₘ[9,t]
                        + ( K_m[19,11]*λ_F[1,t]   + K_m[19,26]*λ_F[2,t] + K_m[19,29]*λ_F[3,t] + K_m[19,30]*λ_F[4,t] ) * ηₘ[19,t]
                        + ( K_m[28,11]*λ_F[1,t]   + K_m[28,26]*λ_F[2,t] + K_m[28,29]*λ_F[3,t] + K_m[28,30]*λ_F[4,t] ) * ηₘ[28,t]
                        + ( K_m[36,11]*λ_F[1,t]   + K_m[36,26]*λ_F[2,t] + K_m[36,29]*λ_F[3,t] + K_m[36,30]*λ_F[4,t] ) * ηₘ[36,t]
                        + ( K_m[43,11]*λ_F[1,t]   + K_m[43,26]*λ_F[2,t] + K_m[43,29]*λ_F[3,t] + K_m[43,30]*λ_F[4,t] ) * ηₘ[43,t]
                        + ( K_m[49,11]*λ_F[1,t]   + K_m[49,26]*λ_F[2,t] + K_m[49,29]*λ_F[3,t] + K_m[49,30]*λ_F[4,t] ) * ηₘ[49,t]
                        + ( K_m[54,11]*λ_F[1,t]   + K_m[54,26]*λ_F[2,t] + K_m[54,29]*λ_F[3,t] + K_m[54,30]*λ_F[4,t] ) * ηₘ[54,t]
                        + ( K_m[58,11]*λ_F[1,t]   + K_m[58,26]*λ_F[2,t] + K_m[58,29]*λ_F[3,t] + K_m[58,30]*λ_F[4,t] ) * ηₘ[58,t]
                        + ( K_m[61,11]*λ_F[1,t]   + K_m[61,26]*λ_F[2,t] + K_m[61,29]*λ_F[3,t] + K_m[61,30]*λ_F[4,t] ) * ηₘ[61,t]
                        - Oⁿˡ[5]*yˢᴳ²⁷_2[t] - Oᵐ₂[5]*Pˢᴳ²⁷_2[t] - Cᵁ²⁷_2[t] >=0 )

@constraint(model,  λP[11,t] + ( K_g[11,11]*λ_F[1,t] + K_g[11,26]*λ_F[2,t] + K_g[11,29]*λ_F[3,t] + K_g[11,30]*λ_F[4,t] ) * yˢᴳ³⁰_1[t]
                        + ( K_m[66,11]*λ_F[1,t] + K_m[66,26]*λ_F[2,t] + K_m[66,29]*λ_F[3,t] + K_m[66,30]*λ_F[4,t] ) * ηₘ[66,t]
                        + ( K_m[10,11]*λ_F[1,t]   + K_m[10,26]*λ_F[2,t] + K_m[10,29]*λ_F[3,t] + K_m[10,30]*λ_F[4,t] ) * ηₘ[10,t]
                        + ( K_m[20,11]*λ_F[1,t]   + K_m[20,26]*λ_F[2,t] + K_m[20,29]*λ_F[3,t] + K_m[20,30]*λ_F[4,t] ) * ηₘ[20,t]
                        + ( K_m[29,11]*λ_F[1,t]   + K_m[29,26]*λ_F[2,t] + K_m[29,29]*λ_F[3,t] + K_m[29,30]*λ_F[4,t] ) * ηₘ[29,t]
                        + ( K_m[37,11]*λ_F[1,t]   + K_m[37,26]*λ_F[2,t] + K_m[37,29]*λ_F[3,t] + K_m[37,30]*λ_F[4,t] ) * ηₘ[37,t]
                        + ( K_m[44,11]*λ_F[1,t]   + K_m[44,26]*λ_F[2,t] + K_m[44,29]*λ_F[3,t] + K_m[44,30]*λ_F[4,t] ) * ηₘ[44,t]
                        + ( K_m[50,11]*λ_F[1,t]   + K_m[50,26]*λ_F[2,t] + K_m[50,29]*λ_F[3,t] + K_m[50,30]*λ_F[4,t] ) * ηₘ[50,t]
                        + ( K_m[55,11]*λ_F[1,t]   + K_m[55,26]*λ_F[2,t] + K_m[55,29]*λ_F[3,t] + K_m[55,30]*λ_F[4,t] ) * ηₘ[55,t]
                        + ( K_m[59,11]*λ_F[1,t]   + K_m[59,26]*λ_F[2,t] + K_m[59,29]*λ_F[3,t] + K_m[59,30]*λ_F[4,t] ) * ηₘ[59,t]
                        + ( K_m[62,11]*λ_F[1,t]   + K_m[62,26]*λ_F[2,t] + K_m[62,29]*λ_F[3,t] + K_m[62,30]*λ_F[4,t] ) * ηₘ[62,t]
                        + ( K_m[64,11]*λ_F[1,t]   + K_m[64,26]*λ_F[2,t] + K_m[64,29]*λ_F[3,t] + K_m[64,30]*λ_F[4,t] ) * ηₘ[64,t]
                        - Oⁿˡ[6]*yˢᴳ³⁰_1[t] - Oᵐ₁[6]*Pˢᴳ³⁰_1[t] - Cᵁ³⁰_1[t] >=0 )

@constraint(model,  λP[12,t] + ( K_g[12,11]*λ_F[1,t] + K_g[12,26]*λ_F[2,t] + K_g[12,29]*λ_F[3,t] + K_g[12,30]*λ_F[4,t] ) * yˢᴳ³⁰_2[t]
                        + ( K_m[66,11]*λ_F[1,t]   + K_m[66,26]*λ_F[2,t] + K_m[66,29]*λ_F[3,t] + K_m[66,30]*λ_F[4,t] ) * ηₘ[66,t]
                        + ( K_m[11,11]*λ_F[1,t]   + K_m[11,26]*λ_F[2,t] + K_m[11,29]*λ_F[3,t] + K_m[11,30]*λ_F[4,t] ) * ηₘ[11,t]
                        + ( K_m[21,11]*λ_F[1,t]   + K_m[21,26]*λ_F[2,t] + K_m[21,29]*λ_F[3,t] + K_m[21,30]*λ_F[4,t] ) * ηₘ[21,t]
                        + ( K_m[30,11]*λ_F[1,t]   + K_m[30,26]*λ_F[2,t] + K_m[30,29]*λ_F[3,t] + K_m[30,30]*λ_F[4,t] ) * ηₘ[30,t]
                        + ( K_m[38,11]*λ_F[1,t]   + K_m[38,26]*λ_F[2,t] + K_m[38,29]*λ_F[3,t] + K_m[38,30]*λ_F[4,t] ) * ηₘ[38,t]
                        + ( K_m[45,11]*λ_F[1,t]   + K_m[45,26]*λ_F[2,t] + K_m[45,29]*λ_F[3,t] + K_m[45,30]*λ_F[4,t] ) * ηₘ[45,t]
                        + ( K_m[51,11]*λ_F[1,t]   + K_m[51,26]*λ_F[2,t] + K_m[51,29]*λ_F[3,t] + K_m[51,30]*λ_F[4,t] ) * ηₘ[51,t]
                        + ( K_m[56,11]*λ_F[1,t]   + K_m[56,26]*λ_F[2,t] + K_m[56,29]*λ_F[3,t] + K_m[56,30]*λ_F[4,t] ) * ηₘ[56,t]
                        + ( K_m[60,11]*λ_F[1,t]   + K_m[60,26]*λ_F[2,t] + K_m[60,29]*λ_F[3,t] + K_m[60,30]*λ_F[4,t] ) * ηₘ[60,t]
                        + ( K_m[63,11]*λ_F[1,t]   + K_m[63,26]*λ_F[2,t] + K_m[63,29]*λ_F[3,t] + K_m[63,30]*λ_F[4,t] ) * ηₘ[63,t]
                        + ( K_m[65,11]*λ_F[1,t]   + K_m[65,26]*λ_F[2,t] + K_m[65,29]*λ_F[3,t] + K_m[65,30]*λ_F[4,t] ) * ηₘ[65,t]
                        - Oⁿˡ[6]*yˢᴳ³⁰_2[t] - Oᵐ₂[6]*Pˢᴳ³⁰_2[t] - Cᵁ³⁰_2[t] >=0 )
end







#-------Define Objective Functions 
#-Primal obj
@variable(model, cost[1:T])
@variable(model, cost_onoff_Primal[1:T])
@variable(model, cost_nl_Primal[1:T])
@variable(model, cost_gene_Primal[1:T])
for t in 1:T
    @constraint(model, cost_onoff_Primal[t]==Cᵁ²_1[t]+Cᵁ³_1[t]+Cᵁ⁴_1[t]+Cᵁ⁵_1[t]+Cᵁ²⁷_1[t]+Cᵁ³⁰_1[t]  +Cᵁ²_2[t]+Cᵁ³_2[t]+Cᵁ⁴_2[t]+Cᵁ⁵_2[t]+Cᵁ²⁷_2[t]+Cᵁ³⁰_2[t])       
    @constraint(model, cost_nl_Primal[t]==Oⁿˡ[1]*(yˢᴳ²_1[t]+yˢᴳ²_2[t])+Oⁿˡ[2]*(yˢᴳ³_1[t]+yˢᴳ³_2[t])+Oⁿˡ[3]*(yˢᴳ⁴_1[t]+yˢᴳ⁴_2[t])+Oⁿˡ[4]*(yˢᴳ⁵_1[t]+yˢᴳ⁵_2[t])+Oⁿˡ[5]*(yˢᴳ²⁷_1[t]+yˢᴳ²⁷_2[t])+Oⁿˡ[6]*(yˢᴳ³⁰_1[t]+yˢᴳ³⁰_2[t])  )  
    @constraint(model, cost_gene_Primal[t]==Oᵐ₁[1]*Pˢᴳ²_1[t]+Oᵐ₂[1]*Pˢᴳ²_2[t] + Oᵐ₁[2]*Pˢᴳ³_1[t]+Oᵐ₂[2]*Pˢᴳ³_2[t] +Oᵐ₁[3]*Pˢᴳ⁴_1[t]+Oᵐ₂[3]*Pˢᴳ⁴_2[t]+Oᵐ₁[4]*Pˢᴳ⁵_1[t]+Oᵐ₂[4]*Pˢᴳ⁵_2[t]+Oᵐ₁[5]*Pˢᴳ²⁷_1[t]+Oᵐ₂[5]*Pˢᴳ²⁷_2[t]+Oᵐ₁[6]*Pˢᴳ³⁰_1[t]+Oᵐ₂[6]*Pˢᴳ³⁰_2[t]   )
    @constraint(model, cost[t]==cost_onoff_Primal[t]+cost_nl_Primal[t]+cost_gene_Primal[t])
end
obj_Primal=sum(cost)

#-Dual obj
@variable(model, obj_Dual[1:T])
for t in 1:T
@constraint(model, obj_Dual[t] == Load_total[t] * λᴱ[t] 
                                + (Iₗᵢₘ- (K_c[1,11]+ K_c[2,11]+ K_c[3,11]))*λ_F[1,t]
                                + (Iₗᵢₘ- (K_c[1,26]+ K_c[2,26]+ K_c[3,26]))*λ_F[2,t]
                                + (Iₗᵢₘ- (K_c[1,29]+ K_c[2,29]+ K_c[3,29]))*λ_F[3,t]
                                + (Iₗᵢₘ- (K_c[1,30]+ K_c[2,30]+ K_c[3,30]))*λ_F[4,t]
                                -( ζᵐᵃˣ¹[t]*IBG₁[t] + ζᵐᵃˣ²³[t]*IBG₂₃[t] + ζᵐᵃˣ²⁶[t]*IBG₂₆[t] ) 
                                - (ψᵐᵃˣˢᴳ²_1[t] + ψᵐᵃˣˢᴳ²_2[t] + ψᵐᵃˣˢᴳ³_1[t] + ψᵐᵃˣˢᴳ³_2[t] + ψᵐᵃˣˢᴳ⁴_1[t] + ψᵐᵃˣˢᴳ⁴_2[t] + ψᵐᵃˣˢᴳ⁵_1[t] + ψᵐᵃˣˢᴳ⁵_2[t] + ψᵐᵃˣˢᴳ²⁷_1[t] + ψᵐᵃˣˢᴳ²⁷_2[t] + ψᵐᵃˣˢᴳ³⁰_1[t] + ψᵐᵃˣˢᴳ³⁰_2[t] ) 
                                - sum(γ_min[:,t])
                                -( yˢᴳ₀[1]*Kˢᵗ[1]*σˢᵗˢᴳ²_1[1] +yˢᴳ₀[1]*Kˢᵗ[1]*σˢᵗˢᴳ²_2[1] ) 
                                -( yˢᴳ₀[2]*Kˢᵗ[2]*σˢᵗˢᴳ³_1[1] +yˢᴳ₀[2]*Kˢᵗ[2]*σˢᵗˢᴳ³_2[1] ) 
                                -( yˢᴳ₀[3]*Kˢᵗ[3]*σˢᵗˢᴳ⁴_1[1] +yˢᴳ₀[3]*Kˢᵗ[3]*σˢᵗˢᴳ⁴_2[1] ) 
                                -( yˢᴳ₀[4]*Kˢᵗ[4]*σˢᵗˢᴳ⁵_1[1] +yˢᴳ₀[4]*Kˢᵗ[4]*σˢᵗˢᴳ⁵_2[1] ) 
                                -( yˢᴳ₀[5]*Kˢᵗ[5]*σˢᵗˢᴳ²⁷_1[1] +yˢᴳ₀[5]*Kˢᵗ[5]*σˢᵗˢᴳ²⁷_2[1] ) 
                                -( yˢᴳ₀[6]*Kˢᵗ[6]*σˢᵗˢᴳ³⁰_1[1] +yˢᴳ₀[6]*Kˢᵗ[6]*σˢᵗˢᴳ³⁰_2[1] ) )
end

@objective(model, Min,  obj_Primal - sum(obj_Dual)  ) 
set_optimizer(model , Gurobi.Optimizer)
set_optimizer_attribute(model, "MIPGap", 0.005) 
optimize!(model)
@elapsed optimize!(model)

#-----------economic metrics for non-strategic one
obj_Primal = value(obj_Primal)
obj_Dual = sum(value.(obj_Dual))
1-obj_Dual/obj_Primal
Energy_price_PD[index] = value(λᴱ)
SCC_price_PD[index] = value(λ_F[4,21])

JuMP.value.

Energy_price_PD[1,:] = JuMP.value.(λᴱ)
SCC_price_PD[1,1,:] = JuMP.value.(λ_F[1,:])
SCC_price_PD[1,2,:] = JuMP.value.(λ_F[2,:])
SCC_price_PD[1,3,:] = JuMP.value.(λ_F[3,:])
SCC_price_PD[1,4,:] = JuMP.value.(λ_F[4,:])

Primal_obj_PD[1] = JuMP.value.(obj_Primal)
Dual_obj_PD[1] = sum(JuMP.value.(obj_Dual))
t_record_PD[1] = @elapsed optimize!(model)

end


plot(SCC_price_PD)


Primal_solution_PD[index]=value(obj_Primal)
Dual_solution_PD[index]=value(obj_Dual)
t_record_PD[index]=@elapsed optimize!(model)

(1 .- Dual_solution_PD./Primal_solution_PD) * 100
t_record_PD