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

@variable(model, yˢᴳ²_1[1:T]>=0)      # status of SGs, buses:2,3,4,5,27,30.  Include strategic and non-strategic players
@variable(model, yˢᴳ²_2[1:T]>=0)          
@variable(model, yˢᴳ³_1[1:T]>=0)  
@variable(model, yˢᴳ³_2[1:T]>=0)          
@variable(model, yˢᴳ⁴_1[1:T]>=0)      
@variable(model, yˢᴳ⁴_2[1:T]>=0)      
@variable(model, yˢᴳ⁵_1[1:T]>=0)      
@variable(model, yˢᴳ⁵_2[1:T]>=0)      
@variable(model, yˢᴳ²⁷_1[1:T]>=0)  
@variable(model, yˢᴳ²⁷_2[1:T]>=0)          
@variable(model, yˢᴳ³⁰_1[1:T]>=0)   
@variable(model, yˢᴳ³⁰_2[1:T]>=0)                       


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



#-------Define Primal Constraints

Power_balance=Dict()
for t in 1:T
    Power_balance[t]=@constraint(model, Pˢᴳ²_1[t]+Pˢᴳ²_2[t]+Pˢᴳ³_1[t]+Pˢᴳ³_2[t]+Pˢᴳ⁴_1[t]+Pˢᴳ⁴_2[t]+Pˢᴳ⁵_1[t]+Pˢᴳ⁵_2[t]+Pˢᴳ²⁷_1[t]+Pˢᴳ²⁷_2[t]+Pˢᴳ³⁰_1[t]+Pˢᴳ³⁰_2[t]+
                   Pᴵᴮᴳ¹[t]+Pᴵᴮᴳ²³[t]+Pᴵᴮᴳ²⁶[t]==Load_total[t])     # power balance , dual variable: λᴱₜ
end

Dual_var_relax_bina_21=Dict()
Dual_var_relax_bina_22=Dict()
Dual_var_relax_bina_31=Dict()
Dual_var_relax_bina_32=Dict()
Dual_var_relax_bina_41=Dict()
Dual_var_relax_bina_42=Dict()
Dual_var_relax_bina_51=Dict()
Dual_var_relax_bina_52=Dict()
Dual_var_relax_bina_271=Dict()
Dual_var_relax_bina_272=Dict()
Dual_var_relax_bina_301=Dict()
Dual_var_relax_bina_302=Dict()

for t in 1:T
    Dual_var_relax_bina_21[t]=@constraint(model, yˢᴳ²_1[t]<=1)
    Dual_var_relax_bina_22[t]=@constraint(model, yˢᴳ²_2[t]<=1)
    Dual_var_relax_bina_31[t]=@constraint(model, yˢᴳ³_1[t]<=1)
    Dual_var_relax_bina_32[t]=@constraint(model, yˢᴳ³_2[t]<=1)
    Dual_var_relax_bina_41[t]=@constraint(model, yˢᴳ⁴_1[t]<=1)
    Dual_var_relax_bina_42[t]=@constraint(model, yˢᴳ⁴_2[t]<=1)
    Dual_var_relax_bina_51[t]=@constraint(model, yˢᴳ⁵_1[t]<=1)
    Dual_var_relax_bina_52[t]=@constraint(model, yˢᴳ⁵_2[t]<=1)
    Dual_var_relax_bina_271[t]=@constraint(model, yˢᴳ²⁷_1[t]<=1)
    Dual_var_relax_bina_272[t]=@constraint(model, yˢᴳ²⁷_2[t]<=1)
    Dual_var_relax_bina_301[t]=@constraint(model, yˢᴳ³⁰_1[t]<=1)
    Dual_var_relax_bina_302[t]=@constraint(model, yˢᴳ³⁰_2[t]<=1)
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
    
for t in 1:T
    @constraint(model, Pᴵᴮᴳ¹[t] <= IBG₁[t])        # wind power limit  , dual variable: ζᵐᵃˣₜ
    @constraint(model, Pᴵᴮᴳ²³[t]<= IBG₂₃[t])       
    @constraint(model, Pᴵᴮᴳ²⁶[t]<= IBG₂₆[t])       
                
end

SCL_bus11_dispat=Dict()
k=11   
for t in 1:T                                              # bounds for the SCL of buses  I_₃₀
        @constraint(model, I_₁₁[t]==                     
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+
        K_c[1,k]+ K_c[2,k]+ K_c[3,k])

        SCL_bus11_dispat[t]=@constraint(model, I_₁₁[t]>=Iₗᵢₘ)                  # AS requirement for SCL on bus F=30  , dual variable: λ_F  
end

SCL_bus26_dispat=Dict()
k=26    # for bus 26   
for t in 1:T                                             # bounds for the SCL of buses  I_₂₆   
        @constraint(model, I_₂₆[t]==                     # SCL on bus F=26
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+
        K_c[1,k]+ K_c[2,k]+ K_c[3,k])

        SCL_bus26_dispat[t]=@constraint(model, I_₂₆[t]>=Iₗᵢₘ)                  # AS requirement for SCL on bus F=26  , dual variable: λ_F  
end

SCL_bus29_dispat=Dict()
k=29    
for t in 1:T                                              # bounds for the SCL of buses   I_₂₉  
        @constraint(model, I_₂₉[t]==                      # SCL on bus F=29  
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+
       K_c[1,k]+ K_c[2,k]+ K_c[3,k])

        SCL_bus29_dispat[t]=@constraint(model, I_₂₉[t]>=Iₗᵢₘ)                  # AS requirement for SCL on bus F=29  , dual variable: λ_F  
end

SCL_bus30_dispat=Dict()
k=30    
for t in 1:T                                              # bounds for the SCL of buses  I_₃₀
        @constraint(model, I_₃₀[t]==                     # SCL on bus F=30  
        K_g[1,k]*yˢᴳ²_1[t]+ K_g[2,k]*yˢᴳ²_2[t]+ 
        K_g[3,k]*yˢᴳ³_1[t]+ K_g[4,k]*yˢᴳ³_2[t]+ 
        K_g[5,k]*yˢᴳ⁴_1[t]+ K_g[6,k]*yˢᴳ⁴_2[t]+
        K_g[7,k]*yˢᴳ⁵_1[t]+ K_g[8,k]*yˢᴳ⁵_2[t]+
        K_g[9,k]*yˢᴳ²⁷_1[t]+ K_g[10,k]*yˢᴳ²⁷_2[t]+ 
        K_g[11,k]*yˢᴳ³⁰_1[t]+ K_g[12,k]*yˢᴳ³⁰_2[t]+
        K_c[1,k]+ K_c[2,k]+ K_c[3,k])

        SCL_bus30_dispat[t]=@constraint(model, I_₃₀[t]>=Iₗᵢₘ)                  # AS requirement for SCL on bus F=30  , dual variable: λ_F  
end


#-------Define Objective Functions 
#-Primal obj
@variable(model, cost[1:T])
@variable(model, cost_onoff_Primal[1:T])
@variable(model, cost_nl_Primal[1:T])
@variable(model, cost_gene_Primal[1:T])
@variable(model, cost_IBR_Primal[1:T])
for t in 1:T
    @constraint(model, cost_onoff_Primal[t]==Cᵁ²_1[t]+Cᴰ²_1[t]+Cᵁ³_1[t]+Cᴰ³_1[t]+Cᵁ⁴_1[t]+Cᴰ⁴_1[t]+Cᵁ⁵_1[t]+Cᴰ⁵_1[t]+Cᵁ²⁷_1[t]+Cᴰ²⁷_1[t]+Cᵁ³⁰_1[t]+Cᴰ³⁰_1[t]  +Cᵁ²_2[t]+Cᴰ²_2[t]+Cᵁ³_2[t]+Cᴰ³_2[t]+Cᵁ⁴_2[t]+Cᴰ⁴_2[t]+Cᵁ⁵_2[t]+Cᴰ⁵_2[t]+Cᵁ²⁷_2[t]+Cᴰ²⁷_2[t]+Cᵁ³⁰_2[t]+Cᴰ³⁰_2[t])       
    @constraint(model, cost_nl_Primal[t]==Oⁿˡ[1].*(yˢᴳ²_1[t]+yˢᴳ²_2[t])+Oⁿˡ[2].*(yˢᴳ³_1[t]+yˢᴳ³_2[t])+Oⁿˡ[3].*(yˢᴳ⁴_1[t]+yˢᴳ⁴_2[t])+Oⁿˡ[4].*(yˢᴳ⁵_1[t]+yˢᴳ⁵_2[t])+Oⁿˡ[5].*(yˢᴳ²⁷_1[t]+yˢᴳ²⁷_2[t])+Oⁿˡ[6].*(yˢᴳ³⁰_1[t]+yˢᴳ³⁰_2[t])  )  
    @constraint(model, cost_gene_Primal[t]==Oᵐ₁[1].*Pˢᴳ²_1[t]+Oᵐ₂[1].*Pˢᴳ²_2[t] + Oᵐ₁[2].*Pˢᴳ³_1[t]+Oᵐ₂[2].*Pˢᴳ³_2[t] +Oᵐ₁[3].*Pˢᴳ⁴_1[t]+Oᵐ₂[3].*Pˢᴳ⁴_2[t]+Oᵐ₁[4].*Pˢᴳ⁵_1[t]+Oᵐ₂[4].*Pˢᴳ⁵_2[t]+Oᵐ₁[5].*Pˢᴳ²⁷_1[t]+Oᵐ₂[5].*Pˢᴳ²⁷_2[t]+Oᵐ₁[6].*Pˢᴳ³⁰_1[t]+Oᵐ₂[6].*Pˢᴳ³⁰_2[t]   )
    #@constraint(model, cost_IBR_Primal[t]==Oᴱ_c[1].*Pᴵᴮᴳ¹[t] +Oᴱ_c[2].*Pᴵᴮᴳ²³[t] +Oᴱ_c[3].*Pᴵᴮᴳ²⁶[t])
    @constraint(model, cost[t]==cost_onoff_Primal[t]+cost_nl_Primal[t]+cost_gene_Primal[t])
end
obj_Primal=sum(cost)

@objective(model, Min, obj_Primal)  # single-level objective function
#-------Solve and Output Results
set_optimizer(model , Gurobi.Optimizer)
optimize!(model)


energy_prices_MAC=zeros(1,T)
SCL_prices_bus11_MAC=zeros(1,T)
SCL_prices_bus26_MAC=zeros(1,T)
SCL_prices_bus29_MAC=zeros(1,T)
SCL_prices_bus30_MAC=zeros(1,T)

Dual_var_relax_bina_2_1=zeros(1,T)
Dual_var_relax_bina_2_2=zeros(1,T)
Dual_var_relax_bina_3_1=zeros(1,T)
Dual_var_relax_bina_3_2=zeros(1,T)
Dual_var_relax_bina_4_1=zeros(1,T)
Dual_var_relax_bina_4_2=zeros(1,T)
Dual_var_relax_bina_5_1=zeros(1,T)
Dual_var_relax_bina_5_2=zeros(1,T)
Dual_var_relax_bina_27_1=zeros(1,T)
Dual_var_relax_bina_27_2=zeros(1,T)
Dual_var_relax_bina_30_1=zeros(1,T)
Dual_var_relax_bina_30_2=zeros(1,T)

for t in 1:T
energy_prices_MAC[t]=dual(Power_balance[t])
SCL_prices_bus11_MAC[t]=dual(SCL_bus11_dispat[t])
SCL_prices_bus26_MAC[t]=dual(SCL_bus26_dispat[t])
SCL_prices_bus29_MAC[t]=dual(SCL_bus29_dispat[t])
SCL_prices_bus30_MAC[t]=dual(SCL_bus30_dispat[t])
Dual_var_relax_bina_2_1[t]=dual(Dual_var_relax_bina_21[t])
Dual_var_relax_bina_2_2[t]=dual(Dual_var_relax_bina_22[t])
Dual_var_relax_bina_3_1[t]=dual(Dual_var_relax_bina_31[t])
Dual_var_relax_bina_3_2[t]=dual(Dual_var_relax_bina_32[t])
Dual_var_relax_bina_4_1[t]=dual(Dual_var_relax_bina_41[t])
Dual_var_relax_bina_4_2[t]=dual(Dual_var_relax_bina_42[t])
Dual_var_relax_bina_5_1[t]=dual(Dual_var_relax_bina_51[t])
Dual_var_relax_bina_5_2[t]=dual(Dual_var_relax_bina_52[t])
Dual_var_relax_bina_27_1[t]=dual(Dual_var_relax_bina_271[t])
Dual_var_relax_bina_27_2[t]=dual(Dual_var_relax_bina_272[t])
Dual_var_relax_bina_30_1[t]=dual(Dual_var_relax_bina_301[t])
Dual_var_relax_bina_30_2[t]=dual(Dual_var_relax_bina_302[t])
end


plot(ψᵐᵃˣˢᴳ²_1)
plot!(-Dual_var_relax_bina_2_1')
plot(ψᵐᵃˣˢᴳ²_2)
plot!(-Dual_var_relax_bina_2_2')
plot(ψᵐᵃˣˢᴳ³_1)
plot!(-Dual_var_relax_bina_3_1')
plot(ψᵐᵃˣˢᴳ³_2)
plot!(-Dual_var_relax_bina_3_2')
plot(ψᵐᵃˣˢᴳ⁴_1)
plot!(-Dual_var_relax_bina_4_1')
plot(ψᵐᵃˣˢᴳ⁴_2)
plot!(-Dual_var_relax_bina_4_2')
plot(ψᵐᵃˣˢᴳ³⁰_1)
plot!(-Dual_var_relax_bina_30_1')
plot(ψᵐᵃˣˢᴳ³⁰_2)
plot!(-Dual_var_relax_bina_30_2')
plot(ψᵐᵃˣˢᴳ²⁷_1)
plot!(-Dual_var_relax_bina_27_1')
plot(ψᵐᵃˣˢᴳ²⁷_2)
plot!(-Dual_var_relax_bina_27_2')
plot(ψᵐᵃˣˢᴳ⁵_1)
plot!(-Dual_var_relax_bina_5_1')
plot(ψᵐᵃˣˢᴳ⁵_2)
plot!(-Dual_var_relax_bina_5_2')


matwrite("dual_var_relax_uc_21.mat", Dict("dual_var_relax_uc_21" => -Dual_var_relax_bina_2_1))
matwrite("dual_var_relax_uc_22.mat", Dict("dual_var_relax_uc_22" => -Dual_var_relax_bina_2_2))
matwrite("dual_var_relax_uc_31.mat", Dict("dual_var_relax_uc_31" => -Dual_var_relax_bina_3_1))
matwrite("dual_var_relax_uc_32.mat", Dict("dual_var_relax_uc_32" => -Dual_var_relax_bina_3_2))
matwrite("dual_var_relax_uc_41.mat", Dict("dual_var_relax_uc_41" => -Dual_var_relax_bina_4_1))
matwrite("dual_var_relax_uc_42.mat", Dict("dual_var_relax_uc_42" => -Dual_var_relax_bina_4_2))
matwrite("dual_var_relax_uc_51.mat", Dict("dual_var_relax_uc_51" => -Dual_var_relax_bina_5_1))
matwrite("dual_var_relax_uc_52.mat", Dict("dual_var_relax_uc_52" => -Dual_var_relax_bina_5_2))
matwrite("dual_var_relax_uc_271.mat", Dict("dual_var_relax_uc_271" => -Dual_var_relax_bina_27_1))
matwrite("dual_var_relax_uc_272.mat", Dict("dual_var_relax_uc_272" => -Dual_var_relax_bina_27_2))
matwrite("dual_var_relax_uc_301.mat", Dict("dual_var_relax_uc_301" => -Dual_var_relax_bina_30_1))
matwrite("dual_var_relax_uc_302.mat", Dict("dual_var_relax_uc_302" => -Dual_var_relax_bina_30_2))

matwrite("y_relax_21.mat", Dict("y_relax_21" => yˢᴳ²_1))
matwrite("y_relax_22.mat", Dict("y_relax_22" => yˢᴳ²_2))
matwrite("y_relax_31.mat", Dict("y_relax_31" => yˢᴳ³_1))
matwrite("y_relax_32.mat", Dict("y_relax_32" => yˢᴳ³_2))
matwrite("y_relax_41.mat", Dict("y_relax_41" => yˢᴳ⁴_1))
matwrite("y_relax_42.mat", Dict("y_relax_42" => yˢᴳ⁴_2))
matwrite("y_relax_51.mat", Dict("y_relax_51" => yˢᴳ⁵_1))
matwrite("y_relax_52.mat", Dict("y_relax_52" => yˢᴳ⁵_2))
matwrite("y_relax_271.mat", Dict("y_relax_271" => yˢᴳ²⁷_1))
matwrite("y_relax_272.mat", Dict("y_relax_272" => yˢᴳ²⁷_2))
matwrite("y_relax_301.mat", Dict("y_relax_301" => yˢᴳ³⁰_1))
matwrite("y_relax_302.mat", Dict("y_relax_302" => yˢᴳ³⁰_2))




plot(energy_prices_MAC')
plot!(λᴱ)

plot(SCL_prices_bus11_MAC')
plot!(SCL_prices_bus26_MAC')
plot!(SCL_prices_bus29_MAC')
plot!(SCL_prices_bus30_MAC')


matwrite("energy_prices_disp_MAC.mat", Dict("energy_prices_disp_MAC" => energy_prices_MAC))
matwrite("SCL_prices_bus11_disp_MAC.mat", Dict("SCL_prices_bus11_disp_MAC" => SCL_prices_bus11_MAC))
matwrite("SCL_prices_bus26_disp_MAC.mat", Dict("SCL_prices_bus26_disp_MAC" => SCL_prices_bus26_MAC))
matwrite("SCL_prices_bus29_disp_MAC.mat", Dict("SCL_prices_bus29_disp_MAC" => SCL_prices_bus29_MAC))
matwrite("SCL_prices_bus30_disp_MAC.mat", Dict("SCL_prices_bus30_disp_MAC" => SCL_prices_bus30_MAC))



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

    K_c[1,k]+ K_c[2,k]+ K_c[3,k]
end
I_min[k]=minimum(I_scc[k,:])
end
