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
include("SCC_contribution.jl")
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
      
#Load_total=19.46 *10^3/3.5   # (MW)

IBG₁=250   
IBG₂₃=250
IBG₂₆=250

Pˢᴳₘₐₓ=[6.584, 5.760, 3.781, 3.335, 3.252, 2.880]*10^3/5            # Max generation of SGs                    SGs, buses:2,3,4,5,27,30
Pˢᴳₘᵢₙ=[3.292, 2.880, 1.512, 0.667, 0.650, 0.288]*10^3/5            #  Min generation of SGs                   SGs, buses:2,3,4,5,27,30
Kˢᵗ=[200, 125, 92.5, 72, 55, 31]*10^3 /10                           #  Startup cost of SGs                     SGs, buses:2,3,4,5,27,30
Oᵐ₁=[6.20, 7.10, 10.47, 12.28, 13.53, 15.36]                        #  Marginal generation cost of SG 1  in    SGs, buses:2,3,4,5,27,30
Oᵐ₂=[7.07, 8.72, 11.49, 12.84, 14.60, 15.02]                        #  Marginal generation cost of SG 2  in    SGs, buses:2,3,4,5,27,30
Oⁿˡ=[17.431, 15.005, 13.755, 10.930, 9.900, 8.570]*10^3 /10         #  No-load cost of SGs                     SGs, buses:2,3,4,5,27,30
yˢᴳ₀=[1 1 1 0 0 0] 

#=
Primal_solution_PD = zeros(6)
Dual_solution_PD = zeros(6)
t_record_PD = zeros(6)
=#

Load_step = 4000
Load_total = [ Load_step, 1.2*Load_step, 1.4*Load_step, 1.6*Load_step, 1.8*Load_step, 2*Load_step]

Energy_price_PD = zeros(6)
SCC_price_PD = zeros(6)

for index in 1:6

#-----------------------------------Define Primal-Dual Model-----------------------------------
model= Model()
#-------Define Primal Variales
 
@variable(model, Pˢᴳ²_1)        # generation of SGs 
@variable(model, Pˢᴳ²_2)  
@variable(model, Pˢᴳ³_1)     
@variable(model, Pˢᴳ³_2)                
@variable(model, Pˢᴳ⁴_1)   
@variable(model, Pˢᴳ⁴_2)             
@variable(model, Pˢᴳ⁵_1)    
@variable(model, Pˢᴳ⁵_2)            
@variable(model, Pˢᴳ²⁷_1) 
@variable(model, Pˢᴳ²⁷_2)                
@variable(model, Pˢᴳ³⁰_1)                
@variable(model, Pˢᴳ³⁰_2)                 

@variable(model, Pᴵᴮᴳ¹>=0)         # generation of IBRs (WT) 
@variable(model, Pᴵᴮᴳ²³>=0)              
@variable(model, Pᴵᴮᴳ²⁶>=0)  

@variable(model, yˢᴳ²_1,Bin)      # status of SGs
@variable(model, yˢᴳ²_2,Bin)          
@variable(model, yˢᴳ³_1,Bin)  
@variable(model, yˢᴳ³_2,Bin)          
@variable(model, yˢᴳ⁴_1,Bin)      
@variable(model, yˢᴳ⁴_2,Bin)      
@variable(model, yˢᴳ⁵_1,Bin)      
@variable(model, yˢᴳ⁵_2,Bin)      
@variable(model, yˢᴳ²⁷_1,Bin)  
@variable(model, yˢᴳ²⁷_2,Bin)          
@variable(model, yˢᴳ³⁰_1,Bin)   
@variable(model, yˢᴳ³⁰_2,Bin)  

@variable(model, Cᵁ²_1>=0)                 
@variable(model, Cᵁ²_2>=0)                                
@variable(model, Cᵁ³_1>=0)    
@variable(model, Cᵁ³_2>=0)                           
@variable(model, Cᵁ⁴_1>=0)     
@variable(model, Cᵁ⁴_2>=0)                        
@variable(model, Cᵁ⁵_1>=0)  
@variable(model, Cᵁ⁵_2>=0)                              
@variable(model, Cᵁ²⁷_1>=0)    
@variable(model, Cᵁ²⁷_2>=0)                          
@variable(model, Cᵁ³⁰_1>=0)      
@variable(model, Cᵁ³⁰_2>=0)               

@variable(model, ηₘ[1:66] , Bin )         



#-------Define Dual Variales
@variable(model, λᴱ)            # price for market clearing                
@variable(model, λ_F[1:4]>=0)       # price for SCL AS      n=1 for bus 26 ; n=2 for bus 29  ; n=3 for bus 30 

@variable(model, ζᵐᵃˣ¹>=0) 
@variable(model, ζᵐᵃˣ²³>=0)
@variable(model, ζᵐᵃˣ²⁶>=0)

@variable(model, μᵐⁱⁿˢᴳ²_1>=0)
@variable(model, μᵐᵃˣˢᴳ²_1>=0)
@variable(model, μᵐⁱⁿˢᴳ²_2>=0)
@variable(model, μᵐᵃˣˢᴳ²_2>=0)
@variable(model, μᵐⁱⁿˢᴳ³_1>=0)
@variable(model, μᵐᵃˣˢᴳ³_1>=0)
@variable(model, μᵐⁱⁿˢᴳ³_2>=0)
@variable(model, μᵐᵃˣˢᴳ³_2>=0)
@variable(model, μᵐⁱⁿˢᴳ⁴_1>=0)
@variable(model, μᵐᵃˣˢᴳ⁴_1>=0)
@variable(model, μᵐⁱⁿˢᴳ⁴_2>=0)
@variable(model, μᵐᵃˣˢᴳ⁴_2>=0)
@variable(model, μᵐⁱⁿˢᴳ⁵_1>=0)
@variable(model, μᵐᵃˣˢᴳ⁵_1>=0)
@variable(model, μᵐⁱⁿˢᴳ⁵_2>=0)
@variable(model, μᵐᵃˣˢᴳ⁵_2>=0)
@variable(model, μᵐⁱⁿˢᴳ²⁷_1>=0)
@variable(model, μᵐᵃˣˢᴳ²⁷_1>=0)
@variable(model, μᵐⁱⁿˢᴳ²⁷_2>=0)
@variable(model, μᵐᵃˣˢᴳ²⁷_2>=0)
@variable(model, μᵐⁱⁿˢᴳ³⁰_1>=0)
@variable(model, μᵐᵃˣˢᴳ³⁰_1>=0)
@variable(model, μᵐⁱⁿˢᴳ³⁰_2>=0)
@variable(model, μᵐᵃˣˢᴳ³⁰_2>=0)

@variable(model, σˢᵗˢᴳ²_1>=0)
@variable(model, σˢᵗˢᴳ²_2>=0)
@variable(model, σˢᵗˢᴳ³_1>=0)
@variable(model, σˢᵗˢᴳ³_2>=0)
@variable(model, σˢᵗˢᴳ⁴_1>=0)
@variable(model, σˢᵗˢᴳ⁴_2>=0)
@variable(model, σˢᵗˢᴳ⁵_1>=0)
@variable(model, σˢᵗˢᴳ⁵_2>=0)
@variable(model, σˢᵗˢᴳ²⁷_1>=0)
@variable(model, σˢᵗˢᴳ²⁷_2>=0)
@variable(model, σˢᵗˢᴳ³⁰_1>=0)
@variable(model, σˢᵗˢᴳ³⁰_2>=0)

@variable(model, ψᵐᵃˣˢᴳ²_1>=0)
@variable(model, ψᵐᵃˣˢᴳ²_2>=0)
@variable(model, ψᵐᵃˣˢᴳ³_1>=0)
@variable(model, ψᵐᵃˣˢᴳ³_2>=0)
@variable(model, ψᵐᵃˣˢᴳ⁴_1>=0)
@variable(model, ψᵐᵃˣˢᴳ⁴_2>=0)
@variable(model, ψᵐᵃˣˢᴳ⁵_1>=0)
@variable(model, ψᵐᵃˣˢᴳ⁵_2>=0)
@variable(model, ψᵐᵃˣˢᴳ²⁷_1>=0)
@variable(model, ψᵐᵃˣˢᴳ²⁷_2>=0)
@variable(model, ψᵐᵃˣˢᴳ³⁰_1>=0)
@variable(model, ψᵐᵃˣˢᴳ³⁰_2>=0)

@variable(model, γ_max[1:66,1:2] >=0)
@variable(model, γ_min[1:66] >=0)


#-------Define Primal Constraints

@constraint(model, Pˢᴳ²_1+Pˢᴳ²_2+Pˢᴳ³_1+Pˢᴳ³_2+Pˢᴳ⁴_1+Pˢᴳ⁴_2+Pˢᴳ⁵_1+Pˢᴳ⁵_2+Pˢᴳ²⁷_1+Pˢᴳ²⁷_2+Pˢᴳ³⁰_1+Pˢᴳ³⁰_2+Pᴵᴮᴳ¹+Pᴵᴮᴳ²³+Pᴵᴮᴳ²⁶ == Load_total[index])  # Dual variable: λᴱ 

#=
@constraint(model, Pˢᴳ²_1<=yˢᴳ²_1*Pˢᴳₘₐₓ[1])      # Dual variable: μᵐᵃˣˢᴳ²_1
@constraint(model, yˢᴳ²_1*Pˢᴳₘᵢₙ[1]<=Pˢᴳ²_1)      # Dual variable: μᵐⁱⁿˢᴳ²_1
@constraint(model, Pˢᴳ²_2<=yˢᴳ²_2*Pˢᴳₘₐₓ[1])      # Dual variable: μᵐᵃˣˢᴳ²_2
@constraint(model, yˢᴳ²_2*Pˢᴳₘᵢₙ[1]<=Pˢᴳ²_2)      # Dual variable: μᵐⁱⁿˢᴳ²_2  
@constraint(model, Pˢᴳ³_1<=yˢᴳ³_1*Pˢᴳₘₐₓ[2])      # Dual variable: μᵐᵃˣˢᴳ³_1
@constraint(model, yˢᴳ³_1*Pˢᴳₘᵢₙ[2]<=Pˢᴳ³_1)      # Dual variable: μᵐⁱⁿˢᴳ³_1
@constraint(model, Pˢᴳ³_2<=yˢᴳ³_2*Pˢᴳₘₐₓ[2])      # Dual variable: μᵐᵃˣˢᴳ³_2
@constraint(model, yˢᴳ³_2*Pˢᴳₘᵢₙ[2]<=Pˢᴳ³_2)      # Dual variable: μᵐⁱⁿˢᴳ³_2
@constraint(model, Pˢᴳ⁴_1<=yˢᴳ⁴_1*Pˢᴳₘₐₓ[3])      # Dual variable: μᵐᵃˣˢᴳ⁴_1
@constraint(model, yˢᴳ⁴_1*Pˢᴳₘᵢₙ[3]<=Pˢᴳ⁴_1)       # Dual variable: μᵐⁱⁿˢᴳ⁴_1
@constraint(model, Pˢᴳ⁴_2<=yˢᴳ⁴_2*Pˢᴳₘₐₓ[3])       # Dual variable: μᵐᵃˣˢᴳ⁴_2
@constraint(model, yˢᴳ⁴_2*Pˢᴳₘᵢₙ[3]<=Pˢᴳ⁴_2)     # Dual variable: μᵐⁱⁿˢᴳ⁴_2
@constraint(model, Pˢᴳ⁵_1<=yˢᴳ⁵_1*Pˢᴳₘₐₓ[4])      # Dual variable: μᵐᵃˣˢᴳ⁵_1 
@constraint(model, yˢᴳ⁵_1*Pˢᴳₘᵢₙ[4]<=Pˢᴳ⁵_1)    # Dual variable: μᵐⁱⁿˢᴳ⁵_1
@constraint(model, Pˢᴳ⁵_2<=yˢᴳ⁵_2*Pˢᴳₘₐₓ[4])    # Dual variable: μᵐᵃˣˢᴳ⁵_2   
@constraint(model, yˢᴳ⁵_2*Pˢᴳₘᵢₙ[4]<=Pˢᴳ⁵_2)    # Dual variable: μᵐⁱⁿˢᴳ⁵_2
@constraint(model, Pˢᴳ²⁷_1<=yˢᴳ²⁷_1*Pˢᴳₘₐₓ[5])     # Dual variable: μᵐᵃˣˢᴳ²⁷_1       
@constraint(model, yˢᴳ²⁷_1*Pˢᴳₘᵢₙ[5]<=Pˢᴳ²⁷_1)     # Dual variable: μᵐⁱⁿˢᴳ²⁷_1
@constraint(model, Pˢᴳ²⁷_2<=yˢᴳ²⁷_2*Pˢᴳₘₐₓ[5])   # Dual variable: μᵐᵃˣˢᴳ²⁷_2       
@constraint(model, yˢᴳ²⁷_2*Pˢᴳₘᵢₙ[5]<=Pˢᴳ²⁷_2)   # Dual variable: μᵐⁱⁿˢᴳ²⁷_2
@constraint(model, Pˢᴳ³⁰_1<=yˢᴳ³⁰_1*Pˢᴳₘₐₓ[6])     # Dual variable: μᵐᵃˣˢᴳ³⁰_1       
@constraint(model, yˢᴳ³⁰_1*Pˢᴳₘᵢₙ[6]<=Pˢᴳ³⁰_1)   # Dual variable: μᵐⁱⁿˢᴳ³⁰_1
@constraint(model, Pˢᴳ³⁰_2<=yˢᴳ³⁰_2*Pˢᴳₘₐₓ[6])   # Dual variable: μᵐᵃˣˢᴳ³⁰_2       
@constraint(model, yˢᴳ³⁰_2*Pˢᴳₘᵢₙ[6]<=Pˢᴳ³⁰_2)  # Dual variable: μᵐⁱⁿˢᴳ³⁰_2
=#

@constraint(model, Cᵁ²_1>=(yˢᴳ²_1-yˢᴳ₀[1])*Kˢᵗ[1])      # Dual variable: σˢᵗˢᴳ²_1
@constraint(model, Cᵁ²_2>=(yˢᴳ²_2-yˢᴳ₀[1])*Kˢᵗ[1])      # Dual variable: σˢᵗˢᴳ²_2                
@constraint(model, Cᵁ³_1>=(yˢᴳ³_1-yˢᴳ₀[2])*Kˢᵗ[2])      # Dual variable: σˢᵗˢᴳ³_1
@constraint(model, Cᵁ³_2>=(yˢᴳ³_2-yˢᴳ₀[2])*Kˢᵗ[2])      # Dual variable: σˢᵗˢᴳ³_2
@constraint(model, Cᵁ⁴_1>=(yˢᴳ⁴_1-yˢᴳ₀[3])*Kˢᵗ[3])      # Dual variable: σˢᵗˢᴳ⁴_1
@constraint(model, Cᵁ⁴_2>=(yˢᴳ⁴_2-yˢᴳ₀[3])*Kˢᵗ[3])      # Dual variable: σˢᵗˢᴳ⁴_2
@constraint(model, Cᵁ⁵_1>=(yˢᴳ⁵_1-yˢᴳ₀[4])*Kˢᵗ[4])      # Dual variable: σˢᵗˢᴳ⁵_1
@constraint(model, Cᵁ⁵_2>=(yˢᴳ⁵_2-yˢᴳ₀[4])*Kˢᵗ[4])      # Dual variable: σˢᵗˢᴳ⁵_2
@constraint(model, Cᵁ²⁷_1>=(yˢᴳ²⁷_1-yˢᴳ₀[5])*Kˢᵗ[5])    # Dual variable: σˢᵗˢᴳ²⁷_1
@constraint(model, Cᵁ²⁷_2>=(yˢᴳ²⁷_2-yˢᴳ₀[5])*Kˢᵗ[5])    # Dual variable: σˢᵗˢᴳ²⁷_2
@constraint(model, Cᵁ³⁰_1>=(yˢᴳ³⁰_1-yˢᴳ₀[6])*Kˢᵗ[6])    # Dual variable: σˢᵗˢᴳ³⁰_1
@constraint(model, Cᵁ³⁰_2>=(yˢᴳ³⁰_2-yˢᴳ₀[6])*Kˢᵗ[6])    # Dual variable: σˢᵗˢᴳ³⁰_2
    
@constraint(model, Pᴵᴮᴳ¹ <= IBG₁)       # Dual variable: ζᵐᵃˣ¹         
@constraint(model, Pᴵᴮᴳ²³<= IBG₂₃)      # Dual variable: ζᵐᵃˣ²³       
@constraint(model, Pᴵᴮᴳ²⁶<= IBG₂₆)      # Dual variable: ζᵐᵃˣ²⁶   
    
    @constraint(model, ηₘ[1]<=yˢᴳ²_1)                   # Dual variable: γ_max[1,1]
    @constraint(model, ηₘ[1]<=yˢᴳ²_2)                   # Dual variable: γ_max[1,2]
    @constraint(model, ηₘ[1]>=yˢᴳ²_1+yˢᴳ²_2-1)          # Dual variable: γ_min[1]
    @constraint(model, ηₘ[2]<=yˢᴳ²_1)                   # Dual variable: γ_max[2,1]
    @constraint(model, ηₘ[2]<=yˢᴳ³_1)                   # Dual variable: γ_max[2,2]
    @constraint(model, ηₘ[2]>=yˢᴳ²_1+yˢᴳ³_1-1)          # Dual variable: γ_min[2]
    @constraint(model, ηₘ[3]<=yˢᴳ²_1)                   # Dual variable: γ_max[3,1]
    @constraint(model, ηₘ[3]<=yˢᴳ³_2)                   # Dual variable: γ_max[3,2]
    @constraint(model, ηₘ[3]>=yˢᴳ²_1+yˢᴳ³_2-1)          # Dual variable: γ_min[3]
    @constraint(model, ηₘ[4]<=yˢᴳ²_1)                   # Dual variable: γ_max[4,1]
    @constraint(model, ηₘ[4]<=yˢᴳ⁴_1)                   # Dual variable: γ_max[4,2]
    @constraint(model, ηₘ[4]>=yˢᴳ²_1+yˢᴳ⁴_1-1)          # Dual variable: γ_min[4]
    @constraint(model, ηₘ[5]<=yˢᴳ²_1)                   # Dual variable: γ_max[5,1]
    @constraint(model, ηₘ[5]<=yˢᴳ⁴_2)                   # Dual variable: γ_max[5,2]
    @constraint(model, ηₘ[5]>=yˢᴳ²_1+yˢᴳ⁴_2-1)          # Dual variable: γ_min[5]
    @constraint(model, ηₘ[6]<=yˢᴳ²_1)                   # Dual variable: γ_max[6,1]
    @constraint(model, ηₘ[6]<=yˢᴳ⁵_1)                   # Dual variable: γ_max[6,2]
    @constraint(model, ηₘ[6]>=yˢᴳ²_1+yˢᴳ⁵_1-1)          # Dual variable: γ_min[6]
    @constraint(model, ηₘ[7]<=yˢᴳ²_1)                   # Dual variable: γ_max[7,1]
    @constraint(model, ηₘ[7]<=yˢᴳ⁵_2)                   # Dual variable: γ_max[7,2]
    @constraint(model, ηₘ[7]>=yˢᴳ²_1+yˢᴳ⁵_2-1)          # Dual variable: γ_min[7]
    @constraint(model, ηₘ[8]<=yˢᴳ²_1)                   # Dual variable: γ_max[8,1]
    @constraint(model, ηₘ[8]<=yˢᴳ²⁷_1)                  # Dual variable: γ_max[8,2]
    @constraint(model, ηₘ[8]>=yˢᴳ²_1+yˢᴳ²⁷_1-1)         # Dual variable: γ_min[8]
    @constraint(model, ηₘ[9]<=yˢᴳ²_1)                   # Dual variable: γ_max[9,1]
    @constraint(model, ηₘ[9]<=yˢᴳ²⁷_2)                  # Dual variable: γ_max[9,2]
    @constraint(model, ηₘ[9]>=yˢᴳ²_1+yˢᴳ²⁷_2-1)         # Dual variable: γ_min[9]
    @constraint(model, ηₘ[10]<=yˢᴳ²_1)                  # Dual variable: γ_max[10,1]
    @constraint(model, ηₘ[10]<=yˢᴳ³⁰_1)                 # Dual variable: γ_max[10,2]
    @constraint(model, ηₘ[10]>=yˢᴳ²_1+yˢᴳ³⁰_1-1)        # Dual variable: γ_min[10]
    @constraint(model, ηₘ[11]<=yˢᴳ²_1)                  # Dual variable: γ_max[11,1]
    @constraint(model, ηₘ[11]<=yˢᴳ³⁰_2)                 # Dual variable: γ_max[11,2]
    @constraint(model, ηₘ[11]>=yˢᴳ²_1+yˢᴳ³⁰_2-1)        # Dual variable: γ_min[11]
    # yˢᴳ²_2
    @constraint(model, ηₘ[12]<=yˢᴳ²_2)                  # Dual variable: γ_max[12,1]
    @constraint(model, ηₘ[12]<=yˢᴳ³_1)                  # Dual variable: γ_max[12,2]
    @constraint(model, ηₘ[12]>=yˢᴳ²_2+yˢᴳ³_1-1)         # Dual variable: γ_min[12]
    @constraint(model, ηₘ[13]<=yˢᴳ²_2)                  # Dual variable: γ_max[13,1]
    @constraint(model, ηₘ[13]<=yˢᴳ³_2)                  # Dual variable: γ_max[13,2]
    @constraint(model, ηₘ[13]>=yˢᴳ²_2+yˢᴳ³_2-1)         # Dual variable: γ_min[13]
    @constraint(model, ηₘ[14]<=yˢᴳ²_2)                  # Dual variable: γ_max[14,1]
    @constraint(model, ηₘ[14]<=yˢᴳ⁴_1)                  # Dual variable: γ_max[14,2]
    @constraint(model, ηₘ[14]>=yˢᴳ²_2+yˢᴳ⁴_1-1)         # Dual variable: γ_min[14]
    @constraint(model, ηₘ[15]<=yˢᴳ²_2)                  # Dual variable: γ_max[15,1]
    @constraint(model, ηₘ[15]<=yˢᴳ⁴_2)                  # Dual variable: γ_max[15,2]
    @constraint(model, ηₘ[15]>=yˢᴳ²_2+yˢᴳ⁴_2-1)         # Dual variable: γ_min[15]
    @constraint(model, ηₘ[16]<=yˢᴳ²_2)                  # Dual variable: γ_max[16,1]
    @constraint(model, ηₘ[16]<=yˢᴳ⁵_1)                  # Dual variable: γ_max[16,2]
    @constraint(model, ηₘ[16]>=yˢᴳ²_2+yˢᴳ⁵_1-1)         # Dual variable: γ_min[16]
    @constraint(model, ηₘ[17]<=yˢᴳ²_2)                  # Dual variable: γ_max[17,1]
    @constraint(model, ηₘ[17]<=yˢᴳ⁵_2)                  # Dual variable: γ_max[17,2]
    @constraint(model, ηₘ[17]>=yˢᴳ²_2+yˢᴳ⁵_2-1)         # Dual variable: γ_min[17]
    @constraint(model, ηₘ[18]<=yˢᴳ²_2)                  # Dual variable: γ_max[18,1]
    @constraint(model, ηₘ[18]<=yˢᴳ²⁷_1)                 # Dual variable: γ_max[18,2]
    @constraint(model, ηₘ[18]>=yˢᴳ²_2+yˢᴳ²⁷_1-1)        # Dual variable: γ_min[18]
    @constraint(model, ηₘ[19]<=yˢᴳ²_2)                  # Dual variable: γ_max[19,1]
    @constraint(model, ηₘ[19]<=yˢᴳ²⁷_2)                 # Dual variable: γ_max[19,2]
    @constraint(model, ηₘ[19]>=yˢᴳ²_2+yˢᴳ²⁷_2-1)        # Dual variable: γ_min[19]
    @constraint(model, ηₘ[20]<=yˢᴳ²_2)                  # Dual variable: γ_max[20,1]
    @constraint(model, ηₘ[20]<=yˢᴳ³⁰_1)                 # Dual variable: γ_max[20,2]
    @constraint(model, ηₘ[20]>=yˢᴳ²_2+yˢᴳ³⁰_1-1)        # Dual variable: γ_min[20]
    @constraint(model, ηₘ[21]<=yˢᴳ²_2)                  # Dual variable: γ_max[21,1]
    @constraint(model, ηₘ[21]<=yˢᴳ³⁰_2)                 # Dual variable: γ_max[21,2]
    @constraint(model, ηₘ[21]>=yˢᴳ²_2+yˢᴳ³⁰_2-1)        # Dual variable: γ_min[21]
    # yˢᴳ³_1
    @constraint(model, ηₘ[22]<=yˢᴳ³_1)                  # Dual variable: γ_max[22,1]
    @constraint(model, ηₘ[22]<=yˢᴳ³_2)                  # Dual variable: γ_max[22,2]
    @constraint(model, ηₘ[22]>=yˢᴳ³_1+yˢᴳ³_2-1)         # Dual variable: γ_min[22]
    @constraint(model, ηₘ[23]<=yˢᴳ³_1)                  # Dual variable: γ_max[23,1]
    @constraint(model, ηₘ[23]<=yˢᴳ⁴_1)                  # Dual variable: γ_max[23,2]
    @constraint(model, ηₘ[23]>=yˢᴳ³_1+yˢᴳ⁴_1-1)         # Dual variable: γ_min[23]
    @constraint(model, ηₘ[24]<=yˢᴳ³_1)                  # Dual variable: γ_max[24,1]
    @constraint(model, ηₘ[24]<=yˢᴳ⁴_2)                  # Dual variable: γ_max[24,2]
    @constraint(model, ηₘ[24]>=yˢᴳ³_1+yˢᴳ⁴_2-1)         # Dual variable: γ_min[24]
    @constraint(model, ηₘ[25]<=yˢᴳ³_1)                  # Dual variable: γ_max[25,1]
    @constraint(model, ηₘ[25]<=yˢᴳ⁵_1)                  # Dual variable: γ_max[25,2]
    @constraint(model, ηₘ[25]>=yˢᴳ³_1+yˢᴳ⁵_1-1)         # Dual variable: γ_min[25]
    @constraint(model, ηₘ[26]<=yˢᴳ³_1)                  # Dual variable: γ_max[26,1]
    @constraint(model, ηₘ[26]<=yˢᴳ⁵_2)                  # Dual variable: γ_max[26,2]
    @constraint(model, ηₘ[26]>=yˢᴳ³_1+yˢᴳ⁵_2-1)         # Dual variable: γ_min[26]
    @constraint(model, ηₘ[27]<=yˢᴳ³_1)                  # Dual variable: γ_max[27,1]
    @constraint(model, ηₘ[27]<=yˢᴳ²⁷_1)                 # Dual variable: γ_max[27,2]
    @constraint(model, ηₘ[27]>=yˢᴳ³_1+yˢᴳ²⁷_1-1)        # Dual variable: γ_min[27]
    @constraint(model, ηₘ[28]<=yˢᴳ³_1)                  # Dual variable: γ_max[28,1]
    @constraint(model, ηₘ[28]<=yˢᴳ²⁷_2)                 # Dual variable: γ_max[28,2]
    @constraint(model, ηₘ[28]>=yˢᴳ³_1+yˢᴳ²⁷_2-1)        # Dual variable: γ_min[28]
    @constraint(model, ηₘ[29]<=yˢᴳ³_1)                  # Dual variable: γ_max[29,1]
    @constraint(model, ηₘ[29]<=yˢᴳ³⁰_1)                 # Dual variable: γ_max[29,2]
    @constraint(model, ηₘ[29]>=yˢᴳ³_1+yˢᴳ³⁰_1-1)        # Dual variable: γ_min[29]
    @constraint(model, ηₘ[30]<=yˢᴳ³_1)                  # Dual variable: γ_max[30,1]
    @constraint(model, ηₘ[30]<=yˢᴳ³⁰_2)                 # Dual variable: γ_max[30,2]
    @constraint(model, ηₘ[30]>=yˢᴳ³_1+yˢᴳ³⁰_2-1)        # Dual variable: γ_min[30]
    # yˢᴳ³_2
    @constraint(model, ηₘ[31]<=yˢᴳ³_2)                  # Dual variable: γ_max[31,1]
    @constraint(model, ηₘ[31]<=yˢᴳ⁴_1)                  # Dual variable: γ_max[31,2]
    @constraint(model, ηₘ[31]>=yˢᴳ³_2+yˢᴳ⁴_1-1)         # Dual variable: γ_min[31]
    @constraint(model, ηₘ[32]<=yˢᴳ³_2)                  # Dual variable: γ_max[32,1]
    @constraint(model, ηₘ[32]<=yˢᴳ⁴_2)                  # Dual variable: γ_max[32,2]
    @constraint(model, ηₘ[32]>=yˢᴳ³_2+yˢᴳ⁴_2-1)         # Dual variable: γ_min[32]
    @constraint(model, ηₘ[33]<=yˢᴳ³_2)                  # Dual variable: γ_max[33,1]
    @constraint(model, ηₘ[33]<=yˢᴳ⁵_1)                  # Dual variable: γ_max[33,2]
    @constraint(model, ηₘ[33]>=yˢᴳ³_2+yˢᴳ⁵_1-1)         # Dual variable: γ_min[33]
    @constraint(model, ηₘ[34]<=yˢᴳ³_2)                  # Dual variable: γ_max[34,1]
    @constraint(model, ηₘ[34]<=yˢᴳ⁵_2)                  # Dual variable: γ_max[34,2]
    @constraint(model, ηₘ[34]>=yˢᴳ³_2+yˢᴳ⁵_2-1)         # Dual variable: γ_min[34]
    @constraint(model, ηₘ[35]<=yˢᴳ³_2)                  # Dual variable: γ_max[35,1]
    @constraint(model, ηₘ[35]<=yˢᴳ²⁷_1)                 # Dual variable: γ_max[35,2]
    @constraint(model, ηₘ[35]>=yˢᴳ³_2+yˢᴳ²⁷_1-1)        # Dual variable: γ_min[35]
    @constraint(model, ηₘ[36]<=yˢᴳ³_2)                  # Dual variable: γ_max[36,1]
    @constraint(model, ηₘ[36]<=yˢᴳ²⁷_2)                 # Dual variable: γ_max[36,2]
    @constraint(model, ηₘ[36]>=yˢᴳ³_2+yˢᴳ²⁷_2-1)        # Dual variable: γ_min[36]
    @constraint(model, ηₘ[37]<=yˢᴳ³_2)                  # Dual variable: γ_max[37,1]
    @constraint(model, ηₘ[37]<=yˢᴳ³⁰_1)                 # Dual variable: γ_max[37,2]
    @constraint(model, ηₘ[37]>=yˢᴳ³_2+yˢᴳ³⁰_1-1)        # Dual variable: γ_min[37]
    @constraint(model, ηₘ[38]<=yˢᴳ³_2)                  # Dual variable: γ_max[38,1]
    @constraint(model, ηₘ[38]<=yˢᴳ³⁰_2)                 # Dual variable: γ_max[38,2]
    @constraint(model, ηₘ[38]>=yˢᴳ³_2+yˢᴳ³⁰_2-1)        # Dual variable: γ_min[38]
    # yˢᴳ⁴_1
    @constraint(model, ηₘ[39]<=yˢᴳ⁴_1)                  # Dual variable: γ_max[39,1]
    @constraint(model, ηₘ[39]<=yˢᴳ⁴_2)                  # Dual variable: γ_max[39,2]
    @constraint(model, ηₘ[39]>=yˢᴳ⁴_1+yˢᴳ⁴_2-1)         # Dual variable: γ_min[39]
    @constraint(model, ηₘ[40]<=yˢᴳ⁴_1)                  # Dual variable: γ_max[40,1]
    @constraint(model, ηₘ[40]<=yˢᴳ⁵_1)                  # Dual variable: γ_max[40,2]
    @constraint(model, ηₘ[40]>=yˢᴳ⁴_1+yˢᴳ⁵_1-1)         # Dual variable: γ_min[40]
    @constraint(model, ηₘ[41]<=yˢᴳ⁴_1)                  # Dual variable: γ_max[41,1]
    @constraint(model, ηₘ[41]<=yˢᴳ⁵_2)                  # Dual variable: γ_max[41,2]
    @constraint(model, ηₘ[41]>=yˢᴳ⁴_1+yˢᴳ⁵_2-1)         # Dual variable: γ_min[41]
    @constraint(model, ηₘ[42]<=yˢᴳ⁴_1)                  # Dual variable: γ_max[42,1]
    @constraint(model, ηₘ[42]<=yˢᴳ²⁷_1)                 # Dual variable: γ_max[42,2]
    @constraint(model, ηₘ[42]>=yˢᴳ⁴_1+yˢᴳ²⁷_1-1)        # Dual variable: γ_min[42]
    @constraint(model, ηₘ[43]<=yˢᴳ⁴_1)                  # Dual variable: γ_max[43,1]
    @constraint(model, ηₘ[43]<=yˢᴳ²⁷_2)                 # Dual variable: γ_max[43,2]
    @constraint(model, ηₘ[43]>=yˢᴳ⁴_1+yˢᴳ²⁷_2-1)        # Dual variable: γ_min[43]
    @constraint(model, ηₘ[44]<=yˢᴳ⁴_1)                  # Dual variable: γ_max[44,1]
    @constraint(model, ηₘ[44]<=yˢᴳ³⁰_1)                 # Dual variable: γ_max[44,2]
    @constraint(model, ηₘ[44]>=yˢᴳ⁴_1+yˢᴳ³⁰_1-1)        # Dual variable: γ_min[44]
    @constraint(model, ηₘ[45]<=yˢᴳ⁴_1)                  # Dual variable: γ_max[45,1]
    @constraint(model, ηₘ[45]<=yˢᴳ³⁰_2)                 # Dual variable: γ_max[45,2]
    @constraint(model, ηₘ[45]>=yˢᴳ⁴_1+yˢᴳ³⁰_2-1)        # Dual variable: γ_min[45]
    # yˢᴳ⁴_2
    @constraint(model, ηₘ[46]<=yˢᴳ⁴_2)                  # Dual variable: γ_max[46,1]
    @constraint(model, ηₘ[46]<=yˢᴳ⁵_1)                  # Dual variable: γ_max[46,2]
    @constraint(model, ηₘ[46]>=yˢᴳ⁴_2+yˢᴳ⁵_1-1)         # Dual variable: γ_min[46]
    @constraint(model, ηₘ[47]<=yˢᴳ⁴_2)                  # Dual variable: γ_max[47,1]
    @constraint(model, ηₘ[47]<=yˢᴳ⁵_2)                  # Dual variable: γ_max[47,2]
    @constraint(model, ηₘ[47]>=yˢᴳ⁴_2+yˢᴳ⁵_2-1)         # Dual variable: γ_min[47]
    @constraint(model, ηₘ[48]<=yˢᴳ⁴_2)                  # Dual variable: γ_max[48,1]
    @constraint(model, ηₘ[48]<=yˢᴳ²⁷_1)                 # Dual variable: γ_max[48,2]
    @constraint(model, ηₘ[48]>=yˢᴳ⁴_2+yˢᴳ²⁷_1-1)        # Dual variable: γ_min[48]
    @constraint(model, ηₘ[49]<=yˢᴳ⁴_2)                  # Dual variable: γ_max[49,1]
    @constraint(model, ηₘ[49]<=yˢᴳ²⁷_2)                 # Dual variable: γ_max[49,2]
    @constraint(model, ηₘ[49]>=yˢᴳ⁴_2+yˢᴳ²⁷_2-1)        # Dual variable: γ_min[49]
    @constraint(model, ηₘ[50]<=yˢᴳ⁴_2)                  # Dual variable: γ_max[50,1]
    @constraint(model, ηₘ[50]<=yˢᴳ³⁰_1)                 # Dual variable: γ_max[50,2]
    @constraint(model, ηₘ[50]>=yˢᴳ⁴_2+yˢᴳ³⁰_1-1)        # Dual variable: γ_min[50]
    @constraint(model, ηₘ[51]<=yˢᴳ⁴_2)                  # Dual variable: γ_max[51,1]
    @constraint(model, ηₘ[51]<=yˢᴳ³⁰_2)                 # Dual variable: γ_max[51,2]
    @constraint(model, ηₘ[51]>=yˢᴳ⁴_2+yˢᴳ³⁰_2-1)        # Dual variable: γ_min[51]
    # yˢᴳ⁵_1
    @constraint(model, ηₘ[52]<=yˢᴳ⁵_1)                  # Dual variable: γ_max[52,1]
    @constraint(model, ηₘ[52]<=yˢᴳ⁵_2)                  # Dual variable: γ_max[52,2]
    @constraint(model, ηₘ[52]>=yˢᴳ⁵_1+yˢᴳ⁵_2-1)         # Dual variable: γ_min[52]
    @constraint(model, ηₘ[53]<=yˢᴳ⁵_1)                  # Dual variable: γ_max[53,1]
    @constraint(model, ηₘ[53]<=yˢᴳ²⁷_1)                 # Dual variable: γ_max[53,2]
    @constraint(model, ηₘ[53]>=yˢᴳ⁵_1+yˢᴳ²⁷_1-1)        # Dual variable: γ_min[53]
    @constraint(model, ηₘ[54]<=yˢᴳ⁵_1)                  # Dual variable: γ_max[54,1]
    @constraint(model, ηₘ[54]<=yˢᴳ²⁷_2)                 # Dual variable: γ_max[54,2]
    @constraint(model, ηₘ[54]>=yˢᴳ⁵_1+yˢᴳ²⁷_2-1)        # Dual variable: γ_min[54]
    @constraint(model, ηₘ[55]<=yˢᴳ⁵_1)                  # Dual variable: γ_max[55,1]
    @constraint(model, ηₘ[55]<=yˢᴳ³⁰_1)                 # Dual variable: γ_max[55,2]
    @constraint(model, ηₘ[55]>=yˢᴳ⁵_1+yˢᴳ³⁰_1-1)        # Dual variable: γ_min[55]
    @constraint(model, ηₘ[56]<=yˢᴳ⁵_1)                  # Dual variable: γ_max[56,1]
    @constraint(model, ηₘ[56]<=yˢᴳ³⁰_2)                 # Dual variable: γ_max[56,2]
    @constraint(model, ηₘ[56]>=yˢᴳ⁵_1+yˢᴳ³⁰_2-1)        # Dual variable: γ_min[56]
    # yˢᴳ⁵_2
    @constraint(model, ηₘ[57]<=yˢᴳ⁵_2)                  # Dual variable: γ_max[57,1]
    @constraint(model, ηₘ[57]<=yˢᴳ²⁷_1)                 # Dual variable: γ_max[57,2]
    @constraint(model, ηₘ[57]>=yˢᴳ⁵_2+yˢᴳ²⁷_1-1)        # Dual variable: γ_min[57]
    @constraint(model, ηₘ[58]<=yˢᴳ⁵_2)                  # Dual variable: γ_max[58,1]
    @constraint(model, ηₘ[58]<=yˢᴳ²⁷_2)                 # Dual variable: γ_max[58,2]
    @constraint(model, ηₘ[58]>=yˢᴳ⁵_2+yˢᴳ²⁷_2-1)        # Dual variable: γ_min[58]
    @constraint(model, ηₘ[59]<=yˢᴳ⁵_2)                  # Dual variable: γ_max[59,1]
    @constraint(model, ηₘ[59]<=yˢᴳ³⁰_1)                 # Dual variable: γ_max[59,2]
    @constraint(model, ηₘ[59]>=yˢᴳ⁵_2+yˢᴳ³⁰_1-1)        # Dual variable: γ_min[59]
    @constraint(model, ηₘ[60]<=yˢᴳ⁵_2)                  # Dual variable: γ_max[60,1]
    @constraint(model, ηₘ[60]<=yˢᴳ³⁰_2)                 # Dual variable: γ_max[60,2]
    @constraint(model, ηₘ[60]>=yˢᴳ⁵_2+yˢᴳ³⁰_2-1)        # Dual variable: γ_min[60]
    # yˢᴳ²⁷_1
    @constraint(model, ηₘ[61]<=yˢᴳ²⁷_1)                 # Dual variable: γ_max[61,1]
    @constraint(model, ηₘ[61]<=yˢᴳ²⁷_2)                 # Dual variable: γ_max[61,2]
    @constraint(model, ηₘ[61]>=yˢᴳ²⁷_1+yˢᴳ²⁷_2-1)       # Dual variable: γ_min[61]
    @constraint(model, ηₘ[62]<=yˢᴳ²⁷_1)                 # Dual variable: γ_max[62,1]
    @constraint(model, ηₘ[62]<=yˢᴳ³⁰_1)                 # Dual variable: γ_max[62,2]
    @constraint(model, ηₘ[62]>=yˢᴳ²⁷_1+yˢᴳ³⁰_1-1)       # Dual variable: γ_min[62]
    @constraint(model, ηₘ[63]<=yˢᴳ²⁷_1)                 # Dual variable: γ_max[63,1]
    @constraint(model, ηₘ[63]<=yˢᴳ³⁰_2)                 # Dual variable: γ_max[63,2]
    @constraint(model, ηₘ[63]>=yˢᴳ²⁷_1+yˢᴳ³⁰_2-1)       # Dual variable: γ_min[63]
    # yˢᴳ²⁷_2
    @constraint(model, ηₘ[64]<=yˢᴳ²⁷_2)                 # Dual variable: γ_max[64,1]
    @constraint(model, ηₘ[64]<=yˢᴳ³⁰_1)                 # Dual variable: γ_max[64,2]
    @constraint(model, ηₘ[64]>=yˢᴳ²⁷_2+yˢᴳ³⁰_1-1)       # Dual variable: γ_min[64]
    @constraint(model, ηₘ[65]<=yˢᴳ²⁷_2)                 # Dual variable: γ_max[65,1]
    @constraint(model, ηₘ[65]<=yˢᴳ³⁰_2)                 # Dual variable: γ_max[65,2]
    @constraint(model, ηₘ[65]>=yˢᴳ²⁷_2+yˢᴳ³⁰_2-1)       # Dual variable: γ_min[65]
    # yˢᴳ³⁰_1
    @constraint(model, ηₘ[66]<=yˢᴳ³⁰_1)                 # Dual variable: γ_max[66,1]
    @constraint(model, ηₘ[66]<=yˢᴳ³⁰_2)                 # Dual variable: γ_max[66,2]
    @constraint(model, ηₘ[66]>=yˢᴳ³⁰_1+yˢᴳ³⁰_2-1)       # Dual variable: γ_min[66]


k=11                                               
        @constraint(model,                  
        K_g[1,k]*yˢᴳ²_1+ K_g[2,k]*yˢᴳ²_2+ 
        K_g[3,k]*yˢᴳ³_1+ K_g[4,k]*yˢᴳ³_2+ 
        K_g[5,k]*yˢᴳ⁴_1+ K_g[6,k]*yˢᴳ⁴_2+
        K_g[7,k]*yˢᴳ⁵_1+ K_g[8,k]*yˢᴳ⁵_2+
        K_g[9,k]*yˢᴳ²⁷_1+ K_g[10,k]*yˢᴳ²⁷_2+ 
        K_g[11,k]*yˢᴳ³⁰_1+ K_g[12,k]*yˢᴳ³⁰_2+

        K_c[1,k]+ K_c[2,k]+ K_c[3,k]+

        K_m[1,k]*ηₘ[1] +K_m[2,k]*ηₘ[2] +K_m[3,k]*ηₘ[3] +K_m[4,k]*ηₘ[4]+
        K_m[5,k]*ηₘ[5] +K_m[6,k]*ηₘ[6] +K_m[7,k]*ηₘ[7] +K_m[8,k]*ηₘ[8]+
        K_m[9,k]*ηₘ[9] +K_m[10,k]*ηₘ[10] +K_m[11,k]*ηₘ[11]+

        K_m[12,k]*ηₘ[12] +K_m[13,k]*ηₘ[13] +K_m[14,k]*ηₘ[14]+
        K_m[15,k]*ηₘ[15] +K_m[16,k]*ηₘ[16] +K_m[17,k]*ηₘ[17] +K_m[18,k]*ηₘ[18]+
        K_m[19,k]*ηₘ[19] +K_m[20,k]*ηₘ[20] +K_m[21,k]*ηₘ[21]+

        K_m[22,k]*ηₘ[22] +K_m[23,k]*ηₘ[23]+
        K_m[24,k]*ηₘ[24] +K_m[25,k]*ηₘ[25] +K_m[26,k]*ηₘ[26] +K_m[27,k]*ηₘ[27]+
        K_m[28,k]*ηₘ[28] +K_m[29,k]*ηₘ[29] +K_m[30,k]*ηₘ[30]+

        K_m[31,k]*ηₘ[31]+
        K_m[32,k]*ηₘ[32] +K_m[33,k]*ηₘ[33] +K_m[34,k]*ηₘ[34] +K_m[35,k]*ηₘ[35]+
        K_m[36,k]*ηₘ[36] +K_m[37,k]*ηₘ[37] +K_m[38,k]*ηₘ[38]+

        K_m[39,k]*ηₘ[39] +K_m[40,k]*ηₘ[40] +K_m[41,k]*ηₘ[41] +K_m[42,k]*ηₘ[42]+
        K_m[43,k]*ηₘ[43] +K_m[44,k]*ηₘ[44] +K_m[45,k]*ηₘ[45]+

        K_m[46,k]*ηₘ[46] +K_m[47,k]*ηₘ[47] +K_m[48,k]*ηₘ[48]+
        K_m[49,k]*ηₘ[49] +K_m[50,k]*ηₘ[50] +K_m[51,k]*ηₘ[51]+

        K_m[52,k]*ηₘ[52] +K_m[53,k]*ηₘ[53]+
        K_m[54,k]*ηₘ[54] +K_m[55,k]*ηₘ[55] +K_m[56,k]*ηₘ[56]+

        K_m[57,k]*ηₘ[57]+
        K_m[58,k]*ηₘ[58] +K_m[59,k]*ηₘ[59] +K_m[60,k]*ηₘ[60]+

        K_m[61,k]*ηₘ[61] +K_m[62,k]*ηₘ[62] +K_m[63,k]*ηₘ[63]+

        K_m[64,k]*ηₘ[64] +K_m[65,k]*ηₘ[65]+

        K_m[66,k]*ηₘ[66] >=Iₗᵢₘ)        # Dual variable: λ_F[1]
                

k=26                                           
        @constraint(model,                      
        K_g[1,k]*yˢᴳ²_1+ K_g[2,k]*yˢᴳ²_2+ 
        K_g[3,k]*yˢᴳ³_1+ K_g[4,k]*yˢᴳ³_2+ 
        K_g[5,k]*yˢᴳ⁴_1+ K_g[6,k]*yˢᴳ⁴_2+
        K_g[7,k]*yˢᴳ⁵_1+ K_g[8,k]*yˢᴳ⁵_2+
        K_g[9,k]*yˢᴳ²⁷_1+ K_g[10,k]*yˢᴳ²⁷_2+ 
        K_g[11,k]*yˢᴳ³⁰_1+ K_g[12,k]*yˢᴳ³⁰_2+

        K_c[1,k]+ K_c[2,k]+ K_c[3,k]+

        K_m[1,k]*ηₘ[1] +K_m[2,k]*ηₘ[2] +K_m[3,k]*ηₘ[3] +K_m[4,k]*ηₘ[4]+
        K_m[5,k]*ηₘ[5] +K_m[6,k]*ηₘ[6] +K_m[7,k]*ηₘ[7] +K_m[8,k]*ηₘ[8]+
        K_m[9,k]*ηₘ[9] +K_m[10,k]*ηₘ[10] +K_m[11,k]*ηₘ[11]+

        K_m[12,k]*ηₘ[12] +K_m[13,k]*ηₘ[13] +K_m[14,k]*ηₘ[14]+
        K_m[15,k]*ηₘ[15] +K_m[16,k]*ηₘ[16] +K_m[17,k]*ηₘ[17] +K_m[18,k]*ηₘ[18]+
        K_m[19,k]*ηₘ[19] +K_m[20,k]*ηₘ[20] +K_m[21,k]*ηₘ[21]+

        K_m[22,k]*ηₘ[22] +K_m[23,k]*ηₘ[23]+
        K_m[24,k]*ηₘ[24] +K_m[25,k]*ηₘ[25] +K_m[26,k]*ηₘ[26] +K_m[27,k]*ηₘ[27]+
        K_m[28,k]*ηₘ[28] +K_m[29,k]*ηₘ[29] +K_m[30,k]*ηₘ[30]+

        K_m[31,k]*ηₘ[31]+
        K_m[32,k]*ηₘ[32] +K_m[33,k]*ηₘ[33] +K_m[34,k]*ηₘ[34] +K_m[35,k]*ηₘ[35]+
        K_m[36,k]*ηₘ[36] +K_m[37,k]*ηₘ[37] +K_m[38,k]*ηₘ[38]+

        K_m[39,k]*ηₘ[39] +K_m[40,k]*ηₘ[40] +K_m[41,k]*ηₘ[41] +K_m[42,k]*ηₘ[42]+
        K_m[43,k]*ηₘ[43] +K_m[44,k]*ηₘ[44] +K_m[45,k]*ηₘ[45]+

        K_m[46,k]*ηₘ[46] +K_m[47,k]*ηₘ[47] +K_m[48,k]*ηₘ[48]+
        K_m[49,k]*ηₘ[49] +K_m[50,k]*ηₘ[50] +K_m[51,k]*ηₘ[51]+

        K_m[52,k]*ηₘ[52] +K_m[53,k]*ηₘ[53]+
        K_m[54,k]*ηₘ[54] +K_m[55,k]*ηₘ[55] +K_m[56,k]*ηₘ[56]+

        K_m[57,k]*ηₘ[57]+
        K_m[58,k]*ηₘ[58] +K_m[59,k]*ηₘ[59] +K_m[60,k]*ηₘ[60]+

        K_m[61,k]*ηₘ[61] +K_m[62,k]*ηₘ[62] +K_m[63,k]*ηₘ[63]+

        K_m[64,k]*ηₘ[64] +K_m[65,k]*ηₘ[65]+

        K_m[66,k]*ηₘ[66] >=Iₗᵢₘ)        # Dual variable: λ_F[2]


k=29                                            
        @constraint(model,                    
        K_g[1,k]*yˢᴳ²_1+ K_g[2,k]*yˢᴳ²_2+ 
        K_g[3,k]*yˢᴳ³_1+ K_g[4,k]*yˢᴳ³_2+ 
        K_g[5,k]*yˢᴳ⁴_1+ K_g[6,k]*yˢᴳ⁴_2+
        K_g[7,k]*yˢᴳ⁵_1+ K_g[8,k]*yˢᴳ⁵_2+
        K_g[9,k]*yˢᴳ²⁷_1+ K_g[10,k]*yˢᴳ²⁷_2+ 
        K_g[11,k]*yˢᴳ³⁰_1+ K_g[12,k]*yˢᴳ³⁰_2+

        K_c[1,k]+ K_c[2,k]+ K_c[3,k]+

        K_m[1,k]*ηₘ[1] +K_m[2,k]*ηₘ[2] +K_m[3,k]*ηₘ[3] +K_m[4,k]*ηₘ[4]+
        K_m[5,k]*ηₘ[5] +K_m[6,k]*ηₘ[6] +K_m[7,k]*ηₘ[7] +K_m[8,k]*ηₘ[8]+
        K_m[9,k]*ηₘ[9] +K_m[10,k]*ηₘ[10] +K_m[11,k]*ηₘ[11]+

        K_m[12,k]*ηₘ[12] +K_m[13,k]*ηₘ[13] +K_m[14,k]*ηₘ[14]+
        K_m[15,k]*ηₘ[15] +K_m[16,k]*ηₘ[16] +K_m[17,k]*ηₘ[17] +K_m[18,k]*ηₘ[18]+
        K_m[19,k]*ηₘ[19] +K_m[20,k]*ηₘ[20] +K_m[21,k]*ηₘ[21]+

        K_m[22,k]*ηₘ[22] +K_m[23,k]*ηₘ[23]+
        K_m[24,k]*ηₘ[24] +K_m[25,k]*ηₘ[25] +K_m[26,k]*ηₘ[26] +K_m[27,k]*ηₘ[27]+
        K_m[28,k]*ηₘ[28] +K_m[29,k]*ηₘ[29] +K_m[30,k]*ηₘ[30]+

        K_m[31,k]*ηₘ[31]+
        K_m[32,k]*ηₘ[32] +K_m[33,k]*ηₘ[33] +K_m[34,k]*ηₘ[34] +K_m[35,k]*ηₘ[35]+
        K_m[36,k]*ηₘ[36] +K_m[37,k]*ηₘ[37] +K_m[38,k]*ηₘ[38]+

        K_m[39,k]*ηₘ[39] +K_m[40,k]*ηₘ[40] +K_m[41,k]*ηₘ[41] +K_m[42,k]*ηₘ[42]+
        K_m[43,k]*ηₘ[43] +K_m[44,k]*ηₘ[44] +K_m[45,k]*ηₘ[45]+

        K_m[46,k]*ηₘ[46] +K_m[47,k]*ηₘ[47] +K_m[48,k]*ηₘ[48]+
        K_m[49,k]*ηₘ[49] +K_m[50,k]*ηₘ[50] +K_m[51,k]*ηₘ[51]+

        K_m[52,k]*ηₘ[52] +K_m[53,k]*ηₘ[53]+
        K_m[54,k]*ηₘ[54] +K_m[55,k]*ηₘ[55] +K_m[56,k]*ηₘ[56]+

        K_m[57,k]*ηₘ[57]+
        K_m[58,k]*ηₘ[58] +K_m[59,k]*ηₘ[59] +K_m[60,k]*ηₘ[60]+

        K_m[61,k]*ηₘ[61] +K_m[62,k]*ηₘ[62] +K_m[63,k]*ηₘ[63]+

        K_m[64,k]*ηₘ[64] +K_m[65,k]*ηₘ[65]+

        K_m[66,k]*ηₘ[66] >=Iₗᵢₘ)                # Dual variable: λ_F[3]


k=30                                            
        @constraint(model,                      
        K_g[1,k]*yˢᴳ²_1+ K_g[2,k]*yˢᴳ²_2+ 
        K_g[3,k]*yˢᴳ³_1+ K_g[4,k]*yˢᴳ³_2+ 
        K_g[5,k]*yˢᴳ⁴_1+ K_g[6,k]*yˢᴳ⁴_2+
        K_g[7,k]*yˢᴳ⁵_1+ K_g[8,k]*yˢᴳ⁵_2+
        K_g[9,k]*yˢᴳ²⁷_1+ K_g[10,k]*yˢᴳ²⁷_2+ 
        K_g[11,k]*yˢᴳ³⁰_1+ K_g[12,k]*yˢᴳ³⁰_2+

        K_c[1,k]+ K_c[2,k]+ K_c[3,k]+

        K_m[1,k]*ηₘ[1] +K_m[2,k]*ηₘ[2] +K_m[3,k]*ηₘ[3] +K_m[4,k]*ηₘ[4]+
        K_m[5,k]*ηₘ[5] +K_m[6,k]*ηₘ[6] +K_m[7,k]*ηₘ[7] +K_m[8,k]*ηₘ[8]+
        K_m[9,k]*ηₘ[9] +K_m[10,k]*ηₘ[10] +K_m[11,k]*ηₘ[11]+

        K_m[12,k]*ηₘ[12] +K_m[13,k]*ηₘ[13] +K_m[14,k]*ηₘ[14]+
        K_m[15,k]*ηₘ[15] +K_m[16,k]*ηₘ[16] +K_m[17,k]*ηₘ[17] +K_m[18,k]*ηₘ[18]+
        K_m[19,k]*ηₘ[19] +K_m[20,k]*ηₘ[20] +K_m[21,k]*ηₘ[21]+

        K_m[22,k]*ηₘ[22] +K_m[23,k]*ηₘ[23]+
        K_m[24,k]*ηₘ[24] +K_m[25,k]*ηₘ[25] +K_m[26,k]*ηₘ[26] +K_m[27,k]*ηₘ[27]+
        K_m[28,k]*ηₘ[28] +K_m[29,k]*ηₘ[29] +K_m[30,k]*ηₘ[30]+

        K_m[31,k]*ηₘ[31]+
        K_m[32,k]*ηₘ[32] +K_m[33,k]*ηₘ[33] +K_m[34,k]*ηₘ[34] +K_m[35,k]*ηₘ[35]+
        K_m[36,k]*ηₘ[36] +K_m[37,k]*ηₘ[37] +K_m[38,k]*ηₘ[38]+

        K_m[39,k]*ηₘ[39] +K_m[40,k]*ηₘ[40] +K_m[41,k]*ηₘ[41] +K_m[42,k]*ηₘ[42]+
        K_m[43,k]*ηₘ[43] +K_m[44,k]*ηₘ[44] +K_m[45,k]*ηₘ[45]+

        K_m[46,k]*ηₘ[46] +K_m[47,k]*ηₘ[47] +K_m[48,k]*ηₘ[48]+
        K_m[49,k]*ηₘ[49] +K_m[50,k]*ηₘ[50] +K_m[51,k]*ηₘ[51]+

        K_m[52,k]*ηₘ[52] +K_m[53,k]*ηₘ[53]+
        K_m[54,k]*ηₘ[54] +K_m[55,k]*ηₘ[55] +K_m[56,k]*ηₘ[56]+

        K_m[57,k]*ηₘ[57]+
        K_m[58,k]*ηₘ[58] +K_m[59,k]*ηₘ[59] +K_m[60,k]*ηₘ[60]+

        K_m[61,k]*ηₘ[61] +K_m[62,k]*ηₘ[62] +K_m[63,k]*ηₘ[63]+

        K_m[64,k]*ηₘ[64] +K_m[65,k]*ηₘ[65]+

        K_m[66,k]*ηₘ[66] >=Iₗᵢₘ)                # Dual variable: λ_F[4]




#-------Define Dual Constraints      
@constraint(model, Oⁿˡ[1]  - ( K_g[1,11]*λ_F[1] + K_g[1,26]*λ_F[2] + K_g[1,29]*λ_F[3] + K_g[1,30]*λ_F[4] ) -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_1 +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_1 +Kˢᵗ[1]*σˢᵗˢᴳ²_1 + ψᵐᵃˣˢᴳ²_1 
                            - sum(γ_max[m,1] for m in 1:11) + sum(γ_min[m] for m in 1:11) >=0 )  

@constraint(model, Oⁿˡ[1]  - ( K_g[2,11]*λ_F[1] + K_g[2,26]*λ_F[2] + K_g[2,29]*λ_F[3] + K_g[2,30]*λ_F[4] ) -Pˢᴳₘₐₓ[1]*μᵐᵃˣˢᴳ²_2 +Pˢᴳₘᵢₙ[1]*μᵐⁱⁿˢᴳ²_2 +Kˢᵗ[1]*σˢᵗˢᴳ²_2 + ψᵐᵃˣˢᴳ²_2 
                            - γ_max[1,2] + γ_min[1] 
                            - sum(γ_max[m,1] for m in 12:21) + sum(γ_min[m] for m in 12:21) >=0 )

@constraint(model, Oⁿˡ[2]  - ( K_g[3,11]*λ_F[1] + K_g[3,26]*λ_F[2] + K_g[3,29]*λ_F[3] + K_g[3,30]*λ_F[4] ) -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_1 +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_1 +Kˢᵗ[2]*σˢᵗˢᴳ³_1 + ψᵐᵃˣˢᴳ³_1 
                            -γ_max[2,2] +γ_min[2] -γ_max[12,2] +γ_min[12] 
                            - sum(γ_max[m,1] for m in 22:30) + sum(γ_min[m] for m in 22:30) >=0 )

@constraint(model, Oⁿˡ[2]  - ( K_g[4,11]*λ_F[1] + K_g[4,26]*λ_F[2] + K_g[4,29]*λ_F[3] + K_g[4,30]*λ_F[4] ) -Pˢᴳₘₐₓ[2]*μᵐᵃˣˢᴳ³_2 +Pˢᴳₘᵢₙ[2]*μᵐⁱⁿˢᴳ³_2 +Kˢᵗ[2]*σˢᵗˢᴳ³_2 + ψᵐᵃˣˢᴳ³_2 
                            -γ_max[3,2] +γ_min[3] -γ_max[13,2] +γ_min[13] -γ_max[22,2] +γ_min[22] 
                            - sum(γ_max[m,1] for m in 31:38) + sum(γ_min[m] for m in 31:38) >=0 )

@constraint(model, Oⁿˡ[3]  - ( K_g[5,11]*λ_F[1] + K_g[5,26]*λ_F[2] + K_g[5,29]*λ_F[3] + K_g[5,30]*λ_F[4] ) -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_1 +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_1 +Kˢᵗ[3]*σˢᵗˢᴳ⁴_1 + ψᵐᵃˣˢᴳ⁴_1 
                            -γ_max[4,2] +γ_min[4] -γ_max[14,2] +γ_min[14] -γ_max[23,2] +γ_min[23] -γ_max[31,2] +γ_min[31] 
                            - sum(γ_max[m,1] for m in 39:45) + sum(γ_min[m] for m in 39:45) >=0 )

@constraint(model, Oⁿˡ[3]  - ( K_g[6,11]*λ_F[1] + K_g[6,26]*λ_F[2] + K_g[6,29]*λ_F[3] + K_g[6,30]*λ_F[4] ) -Pˢᴳₘₐₓ[3]*μᵐᵃˣˢᴳ⁴_2 +Pˢᴳₘᵢₙ[3]*μᵐⁱⁿˢᴳ⁴_2 +Kˢᵗ[3]*σˢᵗˢᴳ⁴_2 + ψᵐᵃˣˢᴳ⁴_2 
                            -γ_max[5,2] +γ_min[5]-γ_max[15,2] +γ_min[15] -γ_max[24,2] +γ_min[24] -γ_max[32,2] +γ_min[32] -γ_max[39,2] +γ_min[39]
                            - sum(γ_max[m,1] for m in 46:51) + sum(γ_min[m] for m in 46:51) >=0 )

@constraint(model, Oⁿˡ[4]  - ( K_g[7,11]*λ_F[1] + K_g[7,26]*λ_F[2] + K_g[7,29]*λ_F[3] + K_g[7,30]*λ_F[4] ) -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_1 +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_1 +Kˢᵗ[4]*σˢᵗˢᴳ⁵_1 + ψᵐᵃˣˢᴳ⁵_1 
                            -γ_max[6,2] +γ_min[6]-γ_max[16,2] +γ_min[16] -γ_max[25,2] +γ_min[25] -γ_max[33,2] +γ_min[33] -γ_max[40,2] +γ_min[40]
                            -γ_max[46,2] +γ_min[46] 
                            - sum(γ_max[m,1] for m in 52:56) + sum(γ_min[m] for m in 52:56) >=0 )

@constraint(model, Oⁿˡ[4]  - ( K_g[8,11]*λ_F[1] + K_g[8,26]*λ_F[2] + K_g[8,29]*λ_F[3] + K_g[8,30]*λ_F[4] ) -Pˢᴳₘₐₓ[4]*μᵐᵃˣˢᴳ⁵_2 +Pˢᴳₘᵢₙ[4]*μᵐⁱⁿˢᴳ⁵_2 +Kˢᵗ[4]*σˢᵗˢᴳ⁵_2 + ψᵐᵃˣˢᴳ⁵_2 
                            -γ_max[7,2] +γ_min[7]-γ_max[17,2] +γ_min[17] -γ_max[26,2] +γ_min[26] -γ_max[34,2] +γ_min[34] -γ_max[41,2] +γ_min[41]
                            -γ_max[47,2] +γ_min[47] -γ_max[52,2] +γ_min[52] 
                            - sum(γ_max[m,1] for m in 57:60) + sum(γ_min[m] for m in 57:60) >=0 )

@constraint(model, Oⁿˡ[5]  - ( K_g[9,11]*λ_F[1] + K_g[9,26]*λ_F[2] + K_g[9,29]*λ_F[3] + K_g[9,30]*λ_F[4] ) -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_1 +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_1 +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_1 + ψᵐᵃˣˢᴳ²⁷_1 
                            -γ_max[8,2] +γ_min[8]-γ_max[18,2] +γ_min[18] -γ_max[27,2] +γ_min[27] -γ_max[35,2] +γ_min[35] -γ_max[42,2] +γ_min[42]
                            -γ_max[48,2] +γ_min[48] -γ_max[53,2] +γ_min[53] -γ_max[57,2] +γ_min[57] 
                            - sum(γ_max[m,1] for m in 61:63) + sum(γ_min[m] for m in 61:63) >=0 )

@constraint(model, Oⁿˡ[5]  - ( K_g[10,11]*λ_F[1] + K_g[10,26]*λ_F[2] + K_g[10,29]*λ_F[3] + K_g[10,30]*λ_F[4] ) -Pˢᴳₘₐₓ[5]*μᵐᵃˣˢᴳ²⁷_2 +Pˢᴳₘᵢₙ[5]*μᵐⁱⁿˢᴳ²⁷_2 +Kˢᵗ[5]*σˢᵗˢᴳ²⁷_2 + ψᵐᵃˣˢᴳ²⁷_2 
                            -γ_max[9,2] +γ_min[9]-γ_max[19,2] +γ_min[19] -γ_max[28,2] +γ_min[28] -γ_max[36,2] +γ_min[36] -γ_max[43,2] +γ_min[43]
                            -γ_max[49,2] +γ_min[49] -γ_max[54,2] +γ_min[54] -γ_max[58,2] +γ_min[58] -γ_max[61,2] +γ_min[61]-γ_max[64,1] +γ_min[64] -γ_max[65,1] +γ_min[65]>=0 )

@constraint(model, Oⁿˡ[6]  - ( K_g[11,11]*λ_F[1] + K_g[11,26]*λ_F[2] + K_g[11,29]*λ_F[3] + K_g[11,30]*λ_F[4] ) -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_1 +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_1 +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_1 + ψᵐᵃˣˢᴳ³⁰_1 
                            -γ_max[10,2] +γ_min[10]-γ_max[20,2] +γ_min[20] -γ_max[29,2] +γ_min[29] -γ_max[37,2] +γ_min[37] -γ_max[44,2] +γ_min[44]
                            -γ_max[50,2] +γ_min[50] -γ_max[55,2] +γ_min[55] -γ_max[59,2] +γ_min[59] -γ_max[62,2] +γ_min[62]-γ_max[64,2] +γ_min[64] -γ_max[66,1] +γ_min[66]>=0 )

@constraint(model, Oⁿˡ[6]  - ( K_g[12,11]*λ_F[1] + K_g[12,26]*λ_F[2] + K_g[12,29]*λ_F[3] + K_g[12,30]*λ_F[4] ) -Pˢᴳₘₐₓ[6]*μᵐᵃˣˢᴳ³⁰_2 +Pˢᴳₘᵢₙ[6]*μᵐⁱⁿˢᴳ³⁰_2 +Kˢᵗ[6]*σˢᵗˢᴳ³⁰_2 + ψᵐᵃˣˢᴳ³⁰_2 
                            -γ_max[11,2] +γ_min[11]-γ_max[21,2] +γ_min[21] -γ_max[30,2] +γ_min[30] -γ_max[38,2] +γ_min[38] -γ_max[45,2] +γ_min[45]
                            -γ_max[51,2] +γ_min[51] -γ_max[56,2] +γ_min[56] -γ_max[60,2] +γ_min[60] -γ_max[63,2] +γ_min[63]-γ_max[65,2] +γ_min[65] -γ_max[66,2] +γ_min[66]>=0 )

@constraint(model, Oᵐ₁[1] - λᴱ +μᵐᵃˣˢᴳ²_1 -μᵐⁱⁿˢᴳ²_1 >=0 )                
@constraint(model, Oᵐ₂[1] - λᴱ +μᵐᵃˣˢᴳ²_2 -μᵐⁱⁿˢᴳ²_2 >=0 )
@constraint(model, Oᵐ₁[2] - λᴱ +μᵐᵃˣˢᴳ³_1 -μᵐⁱⁿˢᴳ³_1 >=0 )                
@constraint(model, Oᵐ₂[2] - λᴱ +μᵐᵃˣˢᴳ³_2 -μᵐⁱⁿˢᴳ³_2 >=0 )
@constraint(model, Oᵐ₁[3] - λᴱ +μᵐᵃˣˢᴳ⁴_1 -μᵐⁱⁿˢᴳ⁴_1 >=0 )
@constraint(model, Oᵐ₂[3] - λᴱ +μᵐᵃˣˢᴳ⁴_2 -μᵐⁱⁿˢᴳ⁴_2 >=0 )
@constraint(model, Oᵐ₁[4] - λᴱ +μᵐᵃˣˢᴳ⁵_1 -μᵐⁱⁿˢᴳ⁵_1 >=0 )
@constraint(model, Oᵐ₂[4] - λᴱ +μᵐᵃˣˢᴳ⁵_2 -μᵐⁱⁿˢᴳ⁵_2 >=0 )
@constraint(model, Oᵐ₁[5] - λᴱ +μᵐᵃˣˢᴳ²⁷_1 -μᵐⁱⁿˢᴳ²⁷_1 >=0 )
@constraint(model, Oᵐ₂[5] - λᴱ +μᵐᵃˣˢᴳ²⁷_2 -μᵐⁱⁿˢᴳ²⁷_2 >=0 )
@constraint(model, Oᵐ₁[6] - λᴱ +μᵐᵃˣˢᴳ³⁰_1 -μᵐⁱⁿˢᴳ³⁰_1 >=0 )
@constraint(model, Oᵐ₂[6] - λᴱ +μᵐᵃˣˢᴳ³⁰_2 -μᵐⁱⁿˢᴳ³⁰_2 >=0 )                

for m in 1:66
    @constraint(model,  -( K_m[m,11]*λ_F[1] + K_m[m,26]*λ_F[2] + K_m[m,29]*λ_F[3] + K_m[m,30]*λ_F[4] )
                                + γ_max[m,1] + γ_max[m,2] - γ_min[m] >=0   )
end

# dual constraint of SG's startup costs Cˢᵗ_g
@constraint(model, σˢᵗˢᴳ²_1<=1)                   
@constraint(model, σˢᵗˢᴳ²_2<=1)
@constraint(model, σˢᵗˢᴳ³_1<=1)
@constraint(model, σˢᵗˢᴳ³_2<=1)
@constraint(model, σˢᵗˢᴳ⁴_1<=1)
@constraint(model, σˢᵗˢᴳ⁴_2<=1)
@constraint(model, σˢᵗˢᴳ⁵_1<=1)
@constraint(model, σˢᵗˢᴳ⁵_2<=1)
@constraint(model, σˢᵗˢᴳ²⁷_1<=1)
@constraint(model, σˢᵗˢᴳ²⁷_2<=1)
@constraint(model, σˢᵗˢᴳ³⁰_1<=1)
@constraint(model, σˢᵗˢᴳ³⁰_2<=1)

# dual constraint of IBR generation P_c
@constraint(model,  -λᴱ +ζᵐᵃˣ¹>=0)
@constraint(model,  -λᴱ +ζᵐᵃˣ²³>=0)
@constraint(model,  -λᴱ +ζᵐᵃˣ²⁶>=0)



#-------Non-negative profit constraints for generators, working for both primal and dual 

M = 2 * Oᵐ₁[6]          # A sufficiently large number
N = 20                   # Number of binary variables for binary expansion of Pˢᴳ variables
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
ΔP[10] = ( Pˢᴳₘₐₓ[5] - Pˢᴳₘᵢₙ[5] ) / ( 2^N - 1)   # Step size for binary expansion of Pˢᴳ²⁷_2
ΔP[11] = ( Pˢᴳₘₐₓ[6] - Pˢᴳₘᵢₙ[6] ) / ( 2^N - 1)   # Step size for binary expansion of Pˢᴳ³⁰_1
ΔP[12] = ( Pˢᴳₘₐₓ[6] - Pˢᴳₘᵢₙ[6] ) / ( 2^N - 1)   # Step size for binary expansion of Pˢᴳ³⁰_2

@variable(model, s[1:12,1:N], Bin)   # Auxiliary binary variables for binary expansion

@constraint(model, Pˢᴳ²_1 == yˢᴳ²_1 * (Pˢᴳₘᵢₙ[1] + ΔP[1] * sum( s[1,n]*2^(n-1) for n in 1:N )  ) )
@constraint(model, Pˢᴳ²_2 == yˢᴳ²_2 * (Pˢᴳₘᵢₙ[1] + ΔP[2] * sum( s[2,n]*2^(n-1) for n in 1:N ) )  )
@constraint(model, Pˢᴳ³_1 == yˢᴳ³_1 * (Pˢᴳₘᵢₙ[2] + ΔP[3] * sum( s[3,n]*2^(n-1) for n in 1:N ) )  )
@constraint(model, Pˢᴳ³_2 == yˢᴳ³_2 * (Pˢᴳₘᵢₙ[2] + ΔP[4] * sum( s[4,n]*2^(n-1) for n in 1:N )  ) )
@constraint(model, Pˢᴳ⁴_1 == yˢᴳ⁴_1 * (Pˢᴳₘᵢₙ[3] + ΔP[5] * sum( s[5,n]*2^(n-1) for n in 1:N ) )  )
@constraint(model, Pˢᴳ⁴_2 == yˢᴳ⁴_2 * (Pˢᴳₘᵢₙ[3] + ΔP[6] * sum( s[6,n]*2^(n-1) for n in 1:N )  ) )
@constraint(model, Pˢᴳ⁵_1 == yˢᴳ⁵_1 * (Pˢᴳₘᵢₙ[4] + ΔP[7] * sum( s[7,n]*2^(n-1) for n in 1:N ) )  )
@constraint(model, Pˢᴳ⁵_2 == yˢᴳ⁵_2 * (Pˢᴳₘᵢₙ[4] + ΔP[8] * sum( s[8,n]*2^(n-1) for n in 1:N )  ) )
@constraint(model, Pˢᴳ²⁷_1 == yˢᴳ²⁷_1 * (Pˢᴳₘᵢₙ[5] + ΔP[9] * sum( s[9,n]*2^(n-1) for n in 1:N )  ) )
@constraint(model, Pˢᴳ²⁷_2 == yˢᴳ²⁷_2 * (Pˢᴳₘᵢₙ[5] + ΔP[10] * sum( s[10,n]*2^(n-1) for n in 1:N ) )  )
@constraint(model, Pˢᴳ³⁰_1 == yˢᴳ³⁰_1 * (Pˢᴳₘᵢₙ[6] + ΔP[11] * sum( s[11,n]*2^(n-1) for n in 1:N ) )  )
@constraint(model, Pˢᴳ³⁰_2 == yˢᴳ³⁰_2 * (Pˢᴳₘᵢₙ[6] + ΔP[12] * sum( s[12,n]*2^(n-1) for n in 1:N )  ) )

@variable(model, λP[1:12] )     # Auxiliary binary variables for binary expansion

@constraint(model, λP[1] == λᴱ * Pˢᴳ²_1  )
@constraint(model, λP[2] == λᴱ * Pˢᴳ²_2  )
@constraint(model, λP[3] == λᴱ * Pˢᴳ³_1  )
@constraint(model, λP[4] == λᴱ * Pˢᴳ³_2  )
@constraint(model, λP[5] == λᴱ * Pˢᴳ⁴_1  )
@constraint(model, λP[6] == λᴱ * Pˢᴳ⁴_2  )
@constraint(model, λP[7] == λᴱ * Pˢᴳ⁵_1  )
@constraint(model, λP[8] == λᴱ * Pˢᴳ⁵_2  )
@constraint(model, λP[9] == λᴱ * Pˢᴳ²⁷_1  )
@constraint(model, λP[10] == λᴱ * Pˢᴳ²⁷_2  )
@constraint(model, λP[11] == λᴱ * Pˢᴳ³⁰_1  )
@constraint(model, λP[12] == λᴱ * Pˢᴳ³⁰_2  )

@constraint(model, λᴱ >= 0  )           # Lower bound for energy price 
@constraint(model, λᴱ <= M  )           # Upper bound for energy price 

@constraint(model,  λ_F .<= 2*( Kˢᵗ[1] + Oⁿˡ[1] ) )               # Upper bound for SCC price of buses

@constraint(model,  λP[1] + ( K_g[1,11]*λ_F[1] + K_g[1,26]*λ_F[2] + K_g[1,29]*λ_F[3] + K_g[1,30]*λ_F[4] ) * yˢᴳ²_1
                        + ( K_m[1,11]*λ_F[1] + K_m[1,26]*λ_F[2] + K_m[1,29]*λ_F[3] + K_m[1,30]*λ_F[4] ) * ηₘ[1]
                        + ( K_m[2,11]*λ_F[1] + K_m[2,26]*λ_F[2] + K_m[2,29]*λ_F[3] + K_m[2,30]*λ_F[4] ) * ηₘ[2]
                        + ( K_m[3,11]*λ_F[1] + K_m[3,26]*λ_F[2] + K_m[3,29]*λ_F[3] + K_m[3,30]*λ_F[4] ) * ηₘ[3]
                        + ( K_m[4,11]*λ_F[1] + K_m[4,26]*λ_F[2] + K_m[4,29]*λ_F[3] + K_m[4,30]*λ_F[4] ) * ηₘ[4]
                        + ( K_m[5,11]*λ_F[1] + K_m[5,26]*λ_F[2] + K_m[5,29]*λ_F[3] + K_m[5,30]*λ_F[4] ) * ηₘ[5]
                        + ( K_m[6,11]*λ_F[1] + K_m[6,26]*λ_F[2] + K_m[6,29]*λ_F[3] + K_m[6,30]*λ_F[4] ) * ηₘ[6]
                        + ( K_m[7,11]*λ_F[1] + K_m[7,26]*λ_F[2] + K_m[7,29]*λ_F[3] + K_m[7,30]*λ_F[4] ) * ηₘ[7]
                        + ( K_m[8,11]*λ_F[1] + K_m[8,26]*λ_F[2] + K_m[8,29]*λ_F[3] + K_m[8,30]*λ_F[4] ) * ηₘ[8]
                        + ( K_m[9,11]*λ_F[1] + K_m[9,26]*λ_F[2] + K_m[9,29]*λ_F[3] + K_m[9,30]*λ_F[4] ) * ηₘ[9]
                        + ( K_m[10,11]*λ_F[1] + K_m[10,26]*λ_F[2] + K_m[10,29]*λ_F[3] + K_m[10,30]*λ_F[4] ) * ηₘ[10]
                        + ( K_m[11,11]*λ_F[1] + K_m[11,26]*λ_F[2] + K_m[11,29]*λ_F[3] + K_m[11,30]*λ_F[4] ) * ηₘ[11]
                        - Oⁿˡ[1]*yˢᴳ²_1 - Oᵐ₁[1]*Pˢᴳ²_1 - Cᵁ²_1 >=0 )

@constraint(model,  λP[2] + ( K_g[2,11]*λ_F[1] + K_g[2,26]*λ_F[2] + K_g[2,29]*λ_F[3] + K_g[2,30]*λ_F[4] ) * yˢᴳ²_2
                        + ( K_m[12,11]*λ_F[1] + K_m[12,26]*λ_F[2] + K_m[12,29]*λ_F[3] + K_m[12,30]*λ_F[4] ) * ηₘ[12]
                        + ( K_m[13,11]*λ_F[1] + K_m[13,26]*λ_F[2] + K_m[13,29]*λ_F[3] + K_m[13,30]*λ_F[4] ) * ηₘ[13]
                        + ( K_m[14,11]*λ_F[1] + K_m[14,26]*λ_F[2] + K_m[14,29]*λ_F[3] + K_m[14,30]*λ_F[4] ) * ηₘ[14]
                        + ( K_m[15,11]*λ_F[1] + K_m[15,26]*λ_F[2] + K_m[15,29]*λ_F[3] + K_m[15,30]*λ_F[4] ) * ηₘ[15]
                        + ( K_m[16,11]*λ_F[1] + K_m[16,26]*λ_F[2] + K_m[16,29]*λ_F[3] + K_m[16,30]*λ_F[4] ) * ηₘ[16]
                        + ( K_m[17,11]*λ_F[1] + K_m[17,26]*λ_F[2] + K_m[17,29]*λ_F[3] + K_m[17,30]*λ_F[4] ) * ηₘ[17]
                        + ( K_m[18,11]*λ_F[1] + K_m[18,26]*λ_F[2] + K_m[18,29]*λ_F[3] + K_m[18,30]*λ_F[4] ) * ηₘ[18]
                        + ( K_m[19,11]*λ_F[1] + K_m[19,26]*λ_F[2] + K_m[19,29]*λ_F[3] + K_m[19,30]*λ_F[4] ) * ηₘ[19]
                        + ( K_m[20,11]*λ_F[1] + K_m[20,26]*λ_F[2] + K_m[20,29]*λ_F[3] + K_m[20,30]*λ_F[4] ) * ηₘ[20]
                        + ( K_m[21,11]*λ_F[1] + K_m[21,26]*λ_F[2] + K_m[21,29]*λ_F[3] + K_m[21,30]*λ_F[4] ) * ηₘ[21]
                        + ( K_m[1,11]*λ_F[1] + K_m[1,26]*λ_F[2] + K_m[1,29]*λ_F[3] + K_m[1,30]*λ_F[4] ) * ηₘ[1]
                        - Oⁿˡ[1]*yˢᴳ²_2 - Oᵐ₂[1]*Pˢᴳ²_2 - Cᵁ²_2 >=0 )

@constraint(model,  λP[3] + ( K_g[3,11]*λ_F[1] + K_g[3,26]*λ_F[2] + K_g[3,29]*λ_F[3] + K_g[3,30]*λ_F[4] ) * yˢᴳ³_1
                        + ( K_m[22,11]*λ_F[1] + K_m[22,26]*λ_F[2] + K_m[22,29]*λ_F[3] + K_m[22,30]*λ_F[4] ) * ηₘ[22]
                        + ( K_m[23,11]*λ_F[1] + K_m[23,26]*λ_F[2] + K_m[23,29]*λ_F[3] + K_m[23,30]*λ_F[4] ) * ηₘ[23]
                        + ( K_m[24,11]*λ_F[1] + K_m[24,26]*λ_F[2] + K_m[24,29]*λ_F[3] + K_m[24,30]*λ_F[4] ) * ηₘ[24]
                        + ( K_m[25,11]*λ_F[1] + K_m[25,26]*λ_F[2] + K_m[25,29]*λ_F[3] + K_m[25,30]*λ_F[4] ) * ηₘ[25]
                        + ( K_m[26,11]*λ_F[1] + K_m[26,26]*λ_F[2] + K_m[26,29]*λ_F[3] + K_m[26,30]*λ_F[4] ) * ηₘ[26]
                        + ( K_m[27,11]*λ_F[1] + K_m[27,26]*λ_F[2] + K_m[27,29]*λ_F[3] + K_m[27,30]*λ_F[4] ) * ηₘ[27]
                        + ( K_m[28,11]*λ_F[1] + K_m[28,26]*λ_F[2] + K_m[28,29]*λ_F[3] + K_m[28,30]*λ_F[4] ) * ηₘ[28]
                        + ( K_m[29,11]*λ_F[1] + K_m[29,26]*λ_F[2] + K_m[29,29]*λ_F[3] + K_m[29,30]*λ_F[4] ) * ηₘ[29]
                        + ( K_m[30,11]*λ_F[1] + K_m[30,26]*λ_F[2] + K_m[30,29]*λ_F[3] + K_m[30,30]*λ_F[4] ) * ηₘ[30]
                        + ( K_m[2,11]*λ_F[1] + K_m[2,26]*λ_F[2] + K_m[2,29]*λ_F[3] + K_m[2,30]*λ_F[4] ) * ηₘ[2]
                        + ( K_m[12,11]*λ_F[1] + K_m[12,26]*λ_F[2] + K_m[12,29]*λ_F[3] + K_m[12,30]*λ_F[4] ) * ηₘ[12]
                        - Oⁿˡ[2]*yˢᴳ³_1 - Oᵐ₁[2]*Pˢᴳ³_1 - Cᵁ³_1 >=0 )
 
@constraint(model,  λP[4] + ( K_g[4,11]*λ_F[1] + K_g[4,26]*λ_F[2] + K_g[4,29]*λ_F[3] + K_g[4,30]*λ_F[4] ) * yˢᴳ³_2
                        + ( K_m[31,11]*λ_F[1] + K_m[31,26]*λ_F[2] + K_m[31,29]*λ_F[3] + K_m[31,30]*λ_F[4] ) * ηₘ[31]
                        + ( K_m[32,11]*λ_F[1] + K_m[32,26]*λ_F[2] + K_m[32,29]*λ_F[3] + K_m[32,30]*λ_F[4] ) * ηₘ[32]
                        + ( K_m[33,11]*λ_F[1] + K_m[33,26]*λ_F[2] + K_m[33,29]*λ_F[3] + K_m[33,30]*λ_F[4] ) * ηₘ[33]
                        + ( K_m[34,11]*λ_F[1] + K_m[34,26]*λ_F[2] + K_m[34,29]*λ_F[3] + K_m[34,30]*λ_F[4] ) * ηₘ[34]
                        + ( K_m[35,11]*λ_F[1] + K_m[35,26]*λ_F[2] + K_m[35,29]*λ_F[3] + K_m[35,30]*λ_F[4] ) * ηₘ[35]
                        + ( K_m[36,11]*λ_F[1] + K_m[36,26]*λ_F[2] + K_m[36,29]*λ_F[3] + K_m[36,30]*λ_F[4] ) * ηₘ[36]
                        + ( K_m[37,11]*λ_F[1] + K_m[37,26]*λ_F[2] + K_m[37,29]*λ_F[3] + K_m[37,30]*λ_F[4] ) * ηₘ[37]
                        + ( K_m[38,11]*λ_F[1] + K_m[38,26]*λ_F[2] + K_m[38,29]*λ_F[3] + K_m[38,30]*λ_F[4] ) * ηₘ[38]
                        + ( K_m[3,11]*λ_F[1] + K_m[3,26]*λ_F[2] + K_m[3,29]*λ_F[3] + K_m[3,30]*λ_F[4] ) * ηₘ[3]
                        + ( K_m[13,11]*λ_F[1] + K_m[13,26]*λ_F[2] + K_m[13,29]*λ_F[3] + K_m[13,30]*λ_F[4] ) * ηₘ[13]
                        + ( K_m[22,11]*λ_F[1] + K_m[22,26]*λ_F[2] + K_m[22,29]*λ_F[3] + K_m[22,30]*λ_F[4] ) * ηₘ[22]
                        - Oⁿˡ[2]*yˢᴳ³_2 - Oᵐ₂[2]*Pˢᴳ³_2 - Cᵁ³_2 >=0 )

@constraint(model,  λP[5] + ( K_g[5,11]*λ_F[1] + K_g[5,26]*λ_F[2] + K_g[5,29]*λ_F[3] + K_g[5,30]*λ_F[4] ) * yˢᴳ⁴_1
                        + ( K_m[39,11]*λ_F[1] + K_m[39,26]*λ_F[2] + K_m[39,29]*λ_F[3] + K_m[39,30]*λ_F[4] ) * ηₘ[39]
                        + ( K_m[40,11]*λ_F[1] + K_m[40,26]*λ_F[2] + K_m[40,29]*λ_F[3] + K_m[40,30]*λ_F[4] ) * ηₘ[40]
                        + ( K_m[41,11]*λ_F[1] + K_m[41,26]*λ_F[2] + K_m[41,29]*λ_F[3] + K_m[41,30]*λ_F[4] ) * ηₘ[41]
                        + ( K_m[42,11]*λ_F[1] + K_m[42,26]*λ_F[2] + K_m[42,29]*λ_F[3] + K_m[42,30]*λ_F[4] ) * ηₘ[42]
                        + ( K_m[43,11]*λ_F[1] + K_m[43,26]*λ_F[2] + K_m[43,29]*λ_F[3] + K_m[43,30]*λ_F[4] ) * ηₘ[43]
                        + ( K_m[44,11]*λ_F[1] + K_m[44,26]*λ_F[2] + K_m[44,29]*λ_F[3] + K_m[44,30]*λ_F[4] ) * ηₘ[44]
                        + ( K_m[45,11]*λ_F[1] + K_m[45,26]*λ_F[2] + K_m[45,29]*λ_F[3] + K_m[45,30]*λ_F[4] ) * ηₘ[45]
                        + ( K_m[4,11]*λ_F[1] + K_m[4,26]*λ_F[2] + K_m[4,29]*λ_F[3] + K_m[4,30]*λ_F[4] ) * ηₘ[4]
                        + ( K_m[14,11]*λ_F[1] + K_m[14,26]*λ_F[2] + K_m[14,29]*λ_F[3] + K_m[14,30]*λ_F[4] ) * ηₘ[14]
                        + ( K_m[23,11]*λ_F[1] + K_m[23,26]*λ_F[2] + K_m[23,29]*λ_F[3] + K_m[23,30]*λ_F[4] ) * ηₘ[23]
                        + ( K_m[31,11]*λ_F[1] + K_m[31,26]*λ_F[2] + K_m[31,29]*λ_F[3] + K_m[31,30]*λ_F[4] ) * ηₘ[31]
                        - Oⁿˡ[3]*yˢᴳ⁴_1 - Oᵐ₁[3]*Pˢᴳ⁴_1 - Cᵁ⁴_1 >=0 )

@constraint(model,  λP[6] + ( K_g[6,11]*λ_F[1] + K_g[6,26]*λ_F[2] + K_g[6,29]*λ_F[3] + K_g[6,30]*λ_F[4] ) * yˢᴳ⁴_2
                        + ( K_m[46,11]*λ_F[1] + K_m[46,26]*λ_F[2] + K_m[46,29]*λ_F[3] + K_m[46,30]*λ_F[4] ) * ηₘ[46]
                        + ( K_m[47,11]*λ_F[1] + K_m[47,26]*λ_F[2] + K_m[47,29]*λ_F[3] + K_m[47,30]*λ_F[4] ) * ηₘ[47]
                        + ( K_m[48,11]*λ_F[1] + K_m[48,26]*λ_F[2] + K_m[48,29]*λ_F[3] + K_m[48,30]*λ_F[4] ) * ηₘ[48]
                        + ( K_m[49,11]*λ_F[1] + K_m[49,26]*λ_F[2] + K_m[49,29]*λ_F[3] + K_m[49,30]*λ_F[4] ) * ηₘ[49]
                        + ( K_m[50,11]*λ_F[1] + K_m[50,26]*λ_F[2] + K_m[50,29]*λ_F[3] + K_m[50,30]*λ_F[4] ) * ηₘ[50]
                        + ( K_m[51,11]*λ_F[1] + K_m[51,26]*λ_F[2] + K_m[51,29]*λ_F[3] + K_m[51,30]*λ_F[4] ) * ηₘ[51]
                        + ( K_m[5,11]*λ_F[1] + K_m[5,26]*λ_F[2] + K_m[5,29]*λ_F[3] + K_m[5,30]*λ_F[4] ) * ηₘ[5]
                        + ( K_m[15,11]*λ_F[1] + K_m[15,26]*λ_F[2] + K_m[15,29]*λ_F[3] + K_m[15,30]*λ_F[4] ) * ηₘ[15]
                        + ( K_m[24,11]*λ_F[1] + K_m[24,26]*λ_F[2] + K_m[24,29]*λ_F[3] + K_m[24,30]*λ_F[4] ) * ηₘ[24]
                        + ( K_m[32,11]*λ_F[1] + K_m[32,26]*λ_F[2] + K_m[32,29]*λ_F[3] + K_m[32,30]*λ_F[4] ) * ηₘ[32]
                        + ( K_m[39,11]*λ_F[1] + K_m[39,26]*λ_F[2] + K_m[39,29]*λ_F[3] + K_m[39,30]*λ_F[4] ) * ηₘ[39]
                        - Oⁿˡ[3]*yˢᴳ⁴_2 - Oᵐ₂[3]*Pˢᴳ⁴_2 - Cᵁ⁴_2 >=0 )

@constraint(model,  λP[7] + ( K_g[7,11]*λ_F[1] + K_g[7,26]*λ_F[2] + K_g[7,29]*λ_F[3] + K_g[7,30]*λ_F[4] ) * yˢᴳ⁵_1
                        + ( K_m[52,11]*λ_F[1] + K_m[52,26]*λ_F[2] + K_m[52,29]*λ_F[3] + K_m[52,30]*λ_F[4] ) * ηₘ[52]
                        + ( K_m[53,11]*λ_F[1] + K_m[53,26]*λ_F[2] + K_m[53,29]*λ_F[3] + K_m[53,30]*λ_F[4] ) * ηₘ[53]
                        + ( K_m[54,11]*λ_F[1] + K_m[54,26]*λ_F[2] + K_m[54,29]*λ_F[3] + K_m[54,30]*λ_F[4] ) * ηₘ[54]
                        + ( K_m[55,11]*λ_F[1] + K_m[55,26]*λ_F[2] + K_m[55,29]*λ_F[3] + K_m[55,30]*λ_F[4] ) * ηₘ[55]
                        + ( K_m[56,11]*λ_F[1] + K_m[56,26]*λ_F[2] + K_m[56,29]*λ_F[3] + K_m[56,30]*λ_F[4] ) * ηₘ[56]
                        + ( K_m[6,11]*λ_F[1] + K_m[6,26]*λ_F[2] + K_m[6,29]*λ_F[3] + K_m[6,30]*λ_F[4] ) * ηₘ[6]
                        + ( K_m[16,11]*λ_F[1] + K_m[16,26]*λ_F[2] + K_m[16,29]*λ_F[3] + K_m[16,30]*λ_F[4] ) * ηₘ[16]
                        + ( K_m[25,11]*λ_F[1] + K_m[25,26]*λ_F[2] + K_m[25,29]*λ_F[3] + K_m[25,30]*λ_F[4] ) * ηₘ[25]
                        + ( K_m[33,11]*λ_F[1] + K_m[33,26]*λ_F[2] + K_m[33,29]*λ_F[3] + K_m[33,30]*λ_F[4] ) * ηₘ[33]
                        + ( K_m[40,11]*λ_F[1] + K_m[40,26]*λ_F[2] + K_m[40,29]*λ_F[3] + K_m[40,30]*λ_F[4] ) * ηₘ[40]
                        + ( K_m[46,11]*λ_F[1] + K_m[46,26]*λ_F[2] + K_m[46,29]*λ_F[3] + K_m[46,30]*λ_F[4] ) * ηₘ[46]
                        - Oⁿˡ[4]*yˢᴳ⁵_1 - Oᵐ₁[4]*Pˢᴳ⁵_1 - Cᵁ⁵_1 >=0 )

@constraint(model,  λP[8] + ( K_g[8,11]*λ_F[1] + K_g[8,26]*λ_F[2] + K_g[8,29]*λ_F[3] + K_g[8,30]*λ_F[4] ) * yˢᴳ⁵_2
                        + ( K_m[57,11]*λ_F[1] + K_m[57,26]*λ_F[2] + K_m[57,29]*λ_F[3] + K_m[57,30]*λ_F[4] ) * ηₘ[57]
                        + ( K_m[58,11]*λ_F[1] + K_m[58,26]*λ_F[2] + K_m[58,29]*λ_F[3] + K_m[58,30]*λ_F[4] ) * ηₘ[58]
                        + ( K_m[59,11]*λ_F[1]   + K_m[59,26]*λ_F[2] + K_m[59,29]*λ_F[3] + K_m[59,30]*λ_F[4] ) * ηₘ[59]
                        + ( K_m[60,11]*λ_F[1]   + K_m[60,26]*λ_F[2] + K_m[60,29]*λ_F[3] + K_m[60,30]*λ_F[4] ) * ηₘ[60]
                        + ( K_m[7,11]*λ_F[1]   + K_m[7,26]*λ_F[2] + K_m[7,29]*λ_F[3] + K_m[7,30]*λ_F[4] ) * ηₘ[7]
                        + ( K_m[17,11]*λ_F[1]   + K_m[17,26]*λ_F[2] + K_m[17,29]*λ_F[3] + K_m[17,30]*λ_F[4] ) * ηₘ[17]
                        + ( K_m[26,11]*λ_F[1]   + K_m[26,26]*λ_F[2] + K_m[26,29]*λ_F[3] + K_m[26,30]*λ_F[4] ) * ηₘ[26]
                        + ( K_m[34,11]*λ_F[1]   + K_m[34,26]*λ_F[2] + K_m[34,29]*λ_F[3] + K_m[34,30]*λ_F[4] ) * ηₘ[34]
                        + ( K_m[41,11]*λ_F[1]   + K_m[41,26]*λ_F[2] + K_m[41,29]*λ_F[3] + K_m[41,30]*λ_F[4] ) * ηₘ[41]
                        + ( K_m[47,11]*λ_F[1]   + K_m[47,26]*λ_F[2] + K_m[47,29]*λ_F[3] + K_m[47,30]*λ_F[4] ) * ηₘ[47]
                        + ( K_m[52,11]*λ_F[1]   + K_m[52,26]*λ_F[2] + K_m[52,29]*λ_F[3] + K_m[52,30]*λ_F[4] ) * ηₘ[52]
                        - Oⁿˡ[4]*yˢᴳ⁵_2 - Oᵐ₂[4]*Pˢᴳ⁵_2 - Cᵁ⁵_2 >=0 )

@constraint(model,  λP[9] + ( K_g[9,11]*λ_F[1] + K_g[9,26]*λ_F[2] + K_g[9,29]*λ_F[3] + K_g[9,30]*λ_F[4] ) * yˢᴳ²⁷_1
                        + ( K_m[61,11]*λ_F[1] + K_m[61,26]*λ_F[2] + K_m[61,29]*λ_F[3] + K_m[61,30]*λ_F[4] ) * ηₘ[61]
                        + ( K_m[62,11]*λ_F[1] + K_m[62,26]*λ_F[2] + K_m[62,29]*λ_F[3] + K_m[62,30]*λ_F[4] ) * ηₘ[62]
                        + ( K_m[63,11]*λ_F[1] + K_m[63,26]*λ_F[2] + K_m[63,29]*λ_F[3] + K_m[63,30]*λ_F[4] ) * ηₘ[63]
                        + ( K_m[8,11]*λ_F[1]   + K_m[8,26]*λ_F[2] + K_m[8,29]*λ_F[3] + K_m[8,30]*λ_F[4] ) * ηₘ[8]
                        + ( K_m[18,11]*λ_F[1]   + K_m[18,26]*λ_F[2] + K_m[18,29]*λ_F[3] + K_m[18,30]*λ_F[4] ) * ηₘ[18]
                        + ( K_m[27,11]*λ_F[1]   + K_m[27,26]*λ_F[2] + K_m[27,29]*λ_F[3] + K_m[27,30]*λ_F[4] ) * ηₘ[27]
                        + ( K_m[35,11]*λ_F[1]   + K_m[35,26]*λ_F[2] + K_m[35,29]*λ_F[3] + K_m[35,30]*λ_F[4] ) * ηₘ[35]
                        + ( K_m[42,11]*λ_F[1]   + K_m[42,26]*λ_F[2] + K_m[42,29]*λ_F[3] + K_m[42,30]*λ_F[4] ) * ηₘ[42]
                        + ( K_m[48,11]*λ_F[1]   + K_m[48,26]*λ_F[2] + K_m[48,29]*λ_F[3] + K_m[48,30]*λ_F[4] ) * ηₘ[48]
                        + ( K_m[53,11]*λ_F[1]   + K_m[53,26]*λ_F[2] + K_m[53,29]*λ_F[3] + K_m[53,30]*λ_F[4] ) * ηₘ[53]
                        + ( K_m[57,11]*λ_F[1]   + K_m[57,26]*λ_F[2] + K_m[57,29]*λ_F[3] + K_m[57,30]*λ_F[4] ) * ηₘ[57]
                        - Oⁿˡ[5]*yˢᴳ²⁷_1 - Oᵐ₁[5]*Pˢᴳ²⁷_1 - Cᵁ²⁷_1 >=0 )

@constraint(model,  λP[10] + ( K_g[10,11]*λ_F[1] + K_g[10,26]*λ_F[2] + K_g[10,29]*λ_F[3] + K_g[10,30]*λ_F[4] ) * yˢᴳ²⁷_2
                        + ( K_m[64,11]*λ_F[1] + K_m[64,26]*λ_F[2] + K_m[64,29]*λ_F[3] + K_m[64,30]*λ_F[4] ) * ηₘ[64]
                        + ( K_m[65,11]*λ_F[1] + K_m[65,26]*λ_F[2] + K_m[65,29]*λ_F[3] + K_m[65,30]*λ_F[4] ) * ηₘ[65]
                        + ( K_m[9,11]*λ_F[1]   + K_m[9,26]*λ_F[2] + K_m[9,29]*λ_F[3] + K_m[9,30]*λ_F[4] ) * ηₘ[9]
                        + ( K_m[19,11]*λ_F[1]   + K_m[19,26]*λ_F[2] + K_m[19,29]*λ_F[3] + K_m[19,30]*λ_F[4] ) * ηₘ[19]
                        + ( K_m[28,11]*λ_F[1]   + K_m[28,26]*λ_F[2] + K_m[28,29]*λ_F[3] + K_m[28,30]*λ_F[4] ) * ηₘ[28]
                        + ( K_m[36,11]*λ_F[1]   + K_m[36,26]*λ_F[2] + K_m[36,29]*λ_F[3] + K_m[36,30]*λ_F[4] ) * ηₘ[36]
                        + ( K_m[43,11]*λ_F[1]   + K_m[43,26]*λ_F[2] + K_m[43,29]*λ_F[3] + K_m[43,30]*λ_F[4] ) * ηₘ[43]
                        + ( K_m[49,11]*λ_F[1]   + K_m[49,26]*λ_F[2] + K_m[49,29]*λ_F[3] + K_m[49,30]*λ_F[4] ) * ηₘ[49]
                        + ( K_m[54,11]*λ_F[1]   + K_m[54,26]*λ_F[2] + K_m[54,29]*λ_F[3] + K_m[54,30]*λ_F[4] ) * ηₘ[54]
                        + ( K_m[58,11]*λ_F[1]   + K_m[58,26]*λ_F[2] + K_m[58,29]*λ_F[3] + K_m[58,30]*λ_F[4] ) * ηₘ[58]
                        + ( K_m[61,11]*λ_F[1]   + K_m[61,26]*λ_F[2] + K_m[61,29]*λ_F[3] + K_m[61,30]*λ_F[4] ) * ηₘ[61]
                        - Oⁿˡ[5]*yˢᴳ²⁷_2 - Oᵐ₂[5]*Pˢᴳ²⁷_2 - Cᵁ²⁷_2 >=0 )

@constraint(model,  λP[11] + ( K_g[11,11]*λ_F[1] + K_g[11,26]*λ_F[2] + K_g[11,29]*λ_F[3] + K_g[11,30]*λ_F[4] ) * yˢᴳ³⁰_1
                        + ( K_m[66,11]*λ_F[1] + K_m[66,26]*λ_F[2] + K_m[66,29]*λ_F[3] + K_m[66,30]*λ_F[4] ) * ηₘ[66]
                        + ( K_m[10,11]*λ_F[1]   + K_m[10,26]*λ_F[2] + K_m[10,29]*λ_F[3] + K_m[10,30]*λ_F[4] ) * ηₘ[10]
                        + ( K_m[20,11]*λ_F[1]   + K_m[20,26]*λ_F[2] + K_m[20,29]*λ_F[3] + K_m[20,30]*λ_F[4] ) * ηₘ[20]
                        + ( K_m[29,11]*λ_F[1]   + K_m[29,26]*λ_F[2] + K_m[29,29]*λ_F[3] + K_m[29,30]*λ_F[4] ) * ηₘ[29]
                        + ( K_m[37,11]*λ_F[1]   + K_m[37,26]*λ_F[2] + K_m[37,29]*λ_F[3] + K_m[37,30]*λ_F[4] ) * ηₘ[37]
                        + ( K_m[44,11]*λ_F[1]   + K_m[44,26]*λ_F[2] + K_m[44,29]*λ_F[3] + K_m[44,30]*λ_F[4] ) * ηₘ[44]
                        + ( K_m[50,11]*λ_F[1]   + K_m[50,26]*λ_F[2] + K_m[50,29]*λ_F[3] + K_m[50,30]*λ_F[4] ) * ηₘ[50]
                        + ( K_m[55,11]*λ_F[1]   + K_m[55,26]*λ_F[2] + K_m[55,29]*λ_F[3] + K_m[55,30]*λ_F[4] ) * ηₘ[55]
                        + ( K_m[59,11]*λ_F[1]   + K_m[59,26]*λ_F[2] + K_m[59,29]*λ_F[3] + K_m[59,30]*λ_F[4] ) * ηₘ[59]
                        + ( K_m[62,11]*λ_F[1]   + K_m[62,26]*λ_F[2] + K_m[62,29]*λ_F[3] + K_m[62,30]*λ_F[4] ) * ηₘ[62]
                        + ( K_m[64,11]*λ_F[1]   + K_m[64,26]*λ_F[2] + K_m[64,29]*λ_F[3] + K_m[64,30]*λ_F[4] ) * ηₘ[64]
                        - Oⁿˡ[6]*yˢᴳ³⁰_1 - Oᵐ₁[6]*Pˢᴳ³⁰_1 - Cᵁ³⁰_1 >=0 )

@constraint(model,  λP[12] + ( K_g[12,11]*λ_F[1] + K_g[12,26]*λ_F[2] + K_g[12,29]*λ_F[3] + K_g[12,30]*λ_F[4] ) * yˢᴳ³⁰_2
                        + ( K_m[66,11]*λ_F[1]   + K_m[66,26]*λ_F[2] + K_m[66,29]*λ_F[3] + K_m[66,30]*λ_F[4] ) * ηₘ[66]
                        + ( K_m[11,11]*λ_F[1]   + K_m[11,26]*λ_F[2] + K_m[11,29]*λ_F[3] + K_m[11,30]*λ_F[4] ) * ηₘ[11]
                        + ( K_m[21,11]*λ_F[1]   + K_m[21,26]*λ_F[2] + K_m[21,29]*λ_F[3] + K_m[21,30]*λ_F[4] ) * ηₘ[21]
                        + ( K_m[30,11]*λ_F[1]   + K_m[30,26]*λ_F[2] + K_m[30,29]*λ_F[3] + K_m[30,30]*λ_F[4] ) * ηₘ[30]
                        + ( K_m[38,11]*λ_F[1]   + K_m[38,26]*λ_F[2] + K_m[38,29]*λ_F[3] + K_m[38,30]*λ_F[4] ) * ηₘ[38]
                        + ( K_m[45,11]*λ_F[1]   + K_m[45,26]*λ_F[2] + K_m[45,29]*λ_F[3] + K_m[45,30]*λ_F[4] ) * ηₘ[45]
                        + ( K_m[51,11]*λ_F[1]   + K_m[51,26]*λ_F[2] + K_m[51,29]*λ_F[3] + K_m[51,30]*λ_F[4] ) * ηₘ[51]
                        + ( K_m[56,11]*λ_F[1]   + K_m[56,26]*λ_F[2] + K_m[56,29]*λ_F[3] + K_m[56,30]*λ_F[4] ) * ηₘ[56]
                        + ( K_m[60,11]*λ_F[1]   + K_m[60,26]*λ_F[2] + K_m[60,29]*λ_F[3] + K_m[60,30]*λ_F[4] ) * ηₘ[60]
                        + ( K_m[63,11]*λ_F[1]   + K_m[63,26]*λ_F[2] + K_m[63,29]*λ_F[3] + K_m[63,30]*λ_F[4] ) * ηₘ[63]
                        + ( K_m[65,11]*λ_F[1]   + K_m[65,26]*λ_F[2] + K_m[65,29]*λ_F[3] + K_m[65,30]*λ_F[4] ) * ηₘ[65]
                        - Oⁿˡ[6]*yˢᴳ³⁰_2 - Oᵐ₂[6]*Pˢᴳ³⁰_2 - Cᵁ³⁰_2 >=0 )




                        #-------Define Objective Functions 
#-Primal obj
@variable(model, obj_Primal)
cost_onoff_Primal=Cᵁ²_1+Cᵁ³_1+Cᵁ⁴_1+Cᵁ⁵_1+Cᵁ²⁷_1+Cᵁ³⁰_1  +Cᵁ²_2+Cᵁ³_2+Cᵁ⁴_2+Cᵁ⁵_2+Cᵁ²⁷_2+Cᵁ³⁰_2      
cost_nl_Primal=Oⁿˡ[1]*(yˢᴳ²_1+yˢᴳ²_2)+Oⁿˡ[2]*(yˢᴳ³_1+yˢᴳ³_2)+Oⁿˡ[3]*(yˢᴳ⁴_1+yˢᴳ⁴_2)+Oⁿˡ[4]*(yˢᴳ⁵_1+yˢᴳ⁵_2)+Oⁿˡ[5]*(yˢᴳ²⁷_1+yˢᴳ²⁷_2)+Oⁿˡ[6]*(yˢᴳ³⁰_1+yˢᴳ³⁰_2)   
cost_gene_Primal=Oᵐ₁[1]*Pˢᴳ²_1+Oᵐ₂[1]*Pˢᴳ²_2 + Oᵐ₁[2]*Pˢᴳ³_1+Oᵐ₂[2]*Pˢᴳ³_2 +Oᵐ₁[3]*Pˢᴳ⁴_1+Oᵐ₂[3]*Pˢᴳ⁴_2+Oᵐ₁[4]*Pˢᴳ⁵_1+Oᵐ₂[4]*Pˢᴳ⁵_2+Oᵐ₁[5]*Pˢᴳ²⁷_1+Oᵐ₂[5]*Pˢᴳ²⁷_2+Oᵐ₁[6]*Pˢᴳ³⁰_1+Oᵐ₂[6]*Pˢᴳ³⁰_2   

@constraint(model, obj_Primal == cost_onoff_Primal +cost_nl_Primal +cost_gene_Primal)


#-Dual obj
@variable(model, obj_Dual)

@constraint(model, obj_Dual == Load_total[index] * λᴱ 
                                + (Iₗᵢₘ- (K_c[1,11]+ K_c[2,11]+ K_c[3,11]))*λ_F[1]
                                + (Iₗᵢₘ- (K_c[1,26]+ K_c[2,26]+ K_c[3,26]))*λ_F[2]
                                + (Iₗᵢₘ- (K_c[1,29]+ K_c[2,29]+ K_c[3,29]))*λ_F[3]
                                + (Iₗᵢₘ- (K_c[1,30]+ K_c[2,30]+ K_c[3,30]))*λ_F[4]
                                -( ζᵐᵃˣ¹*IBG₁ + ζᵐᵃˣ²³*IBG₂₃ + ζᵐᵃˣ²⁶*IBG₂₆ ) 
                                - (ψᵐᵃˣˢᴳ²_1 + ψᵐᵃˣˢᴳ²_2 + ψᵐᵃˣˢᴳ³_1 + ψᵐᵃˣˢᴳ³_2 + ψᵐᵃˣˢᴳ⁴_1 + ψᵐᵃˣˢᴳ⁴_2 + ψᵐᵃˣˢᴳ⁵_1 + ψᵐᵃˣˢᴳ⁵_2 + ψᵐᵃˣˢᴳ²⁷_1 + ψᵐᵃˣˢᴳ²⁷_2 + ψᵐᵃˣˢᴳ³⁰_1 + ψᵐᵃˣˢᴳ³⁰_2 ) 
                                - sum(γ_min)
                                -( yˢᴳ₀[1]*Kˢᵗ[1]*σˢᵗˢᴳ²_1 +yˢᴳ₀[1]*Kˢᵗ[1]*σˢᵗˢᴳ²_2) 
                                -( yˢᴳ₀[2]*Kˢᵗ[2]*σˢᵗˢᴳ³_1 +yˢᴳ₀[2]*Kˢᵗ[2]*σˢᵗˢᴳ³_2) 
                                -( yˢᴳ₀[3]*Kˢᵗ[3]*σˢᵗˢᴳ⁴_1 +yˢᴳ₀[3]*Kˢᵗ[3]*σˢᵗˢᴳ⁴_2) 
                                -( yˢᴳ₀[4]*Kˢᵗ[4]*σˢᵗˢᴳ⁵_1 +yˢᴳ₀[4]*Kˢᵗ[4]*σˢᵗˢᴳ⁵_2) 
                                -( yˢᴳ₀[5]*Kˢᵗ[5]*σˢᵗˢᴳ²⁷_1 +yˢᴳ₀[5]*Kˢᵗ[5]*σˢᵗˢᴳ²⁷_2) 
                                -( yˢᴳ₀[6]*Kˢᵗ[6]*σˢᵗˢᴳ³⁰_1 +yˢᴳ₀[6]*Kˢᵗ[6]*σˢᵗˢᴳ³⁰_2) )


@objective(model, Min, obj_Primal - obj_Dual )  
set_optimizer(model , Gurobi.Optimizer)
@time optimize!(model)
optimize!(model)


#-----------economic metrics for non-strategic one


Energy_price_PD[index] = value(λᴱ)
SCC_price_PD[index] = value(λ_F[2])

end

plot(SCC_price_PD)


Primal_solution_PD[index]=value(obj_Primal)
Dual_solution_PD[index]=value(obj_Dual)
t_record_PD[index]=@elapsed optimize!(model)

(1 .- Dual_solution_PD./Primal_solution_PD) * 100
t_record_PD