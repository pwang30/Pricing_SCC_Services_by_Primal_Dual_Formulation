# Author: Peng Wang       from Technical University of Madrid (UPM)
# Supervisor: Luis Badesa

# Pricing voltage stability
# 12.Apr.2026


#------------------------------------------------------------------------------
#--------Price voltage stability with restricted method------------------------
#------------------------------------------------------------------------------



import Pkg
using JuMP,Gurobi, CSV,DataFrames,LinearAlgebra, XLSX, IterTools, DelimitedFiles, Plots, DataStructures, Dualization
include("dataset_gene.jl")
include("admittance_matrix_calculation.jl") 
include("offline_trainning.jl")
# SGs_ buses:2_3_4_5_27_30    VSG_ buses:1_     IBGs_ buses:23_24

data_bus = CSV.read("IEEE30_Bus_Data.csv", DataFrame)
Bus_Pd = CSV.read("Bus_Pd_24h.csv", DataFrame)
Bus_Qd = CSV.read("Bus_Qd_24h.csv", DataFrame)



#-----------------------------------Representation and Approximation of VOLTAGE STABILITY Constraints  &  Visualization-----------------------------------
nᵥ= []              # The interval is evenly divided into nᵥ parts
for i in 0:0.001:1
    push!(nᵥ,i)
end 
IBG=[23,24]         # GFL.   The location (Bus) of IBGs.
Gᵥ=[1]              # FM.    The location (Bus) of Virtual Synchronous Generators (VSGs).
zᵟ_c, matrix_ω =dataset_gene(nᵥ,Gᵥ,IBG)                                                             # data set generation      +sum(K_c_m[2_:] .*matrix_ω[i_14:91])                
K_c_gc, K_c_gv, K_c_m, MAPE_z_23_1, MAPE_z_24_1 = offline_trainning(zᵟ_c, matrix_ω)  # offline_trainning



#-----------------------------------Define Parameters for Optimization-----------------------------------  
T=24                                    # number of periods

Pᴰ=  sum( Matrix(Bus_Pd[:,(2:25)]  )  , dims=1)*1.5           # system demand of active power at each bus
Qᴰ=  sum( Matrix(Bus_Qd[:,(2:25)]  )  , dims=1)*1.5            # system demand of reactive power at each bus

Pᴰ=     Pᴰ[10]         # system demand of active power at each bus
Qᴰ=     Qᴰ[10]            # system demand of reactive power at each bus

Pˢᴳₘₐₓ= [72.4, 68.8, 63.4, 60.5, 56.9, 51.3] ./1.2
Pˢᴳₘᵢₙ= Pˢᴳₘₐₓ*0.3                      # ACtive min generation of SGs_  buses:2_3_4_5_27_30
Qˢᴳₘₐₓ= Pˢᴳₘₐₓ*0.6                          # REACtive max generation of SGs_  buses:2_3_4_5_27_30
Qˢᴳₘᵢₙ= -Pˢᴳₘₐₓ*0.6                      # REACtive min generation of SGs_  buses:2_3_4_5_27_30

Pⱽˢᴳₘₐₓ=[80]                           # ACtive max generation of VSGs_ buses:1
Pⱽˢᴳₘᵢₙ=[0]                             # ACtive min generation of VSGs_  buses:1
Qⱽˢᴳₘₐₓ= Pⱽˢᴳₘₐₓ *0.6                         # REACtive max generation of SGs_  buses:2_3_4_5_27_30
Qⱽˢᴳₘᵢₙ= -Pⱽˢᴳₘₐₓ*0.6  

Pᴵᴮᴳₘₐₓ= [80, 80]                      # ACtive max generation of IBGs_ buses:23_24
Pᴵᴮᴳₘᵢₙ=  [0, 0]                        # ACtive min generation of IBGs_ buses:23_24

Qᴵᴮᴳₘₐₓ= Pᴵᴮᴳₘₐₓ*0.6
Qᴵᴮᴳₘᵢₙ= -Pᴵᴮᴳₘₐₓ*0.6

Sᵐᵃˣ_gc=Pˢᴳₘₐₓ*1.2
Sᵐᵃˣ_gv=Pⱽˢᴳₘₐₓ*1.2
Sᵐᵃˣ_c=Pᴵᴮᴳₘₐₓ*1.2
S_Base=20

α_VSG=0.9
α_IBG=1
ratio=1

cˢᵗ=[125, 120, 115, 110, 105, 100]              #  Startup cost of SGs                     SGs_ buses:2_3_4_5_27_30
cᵐ=[6.2, 7.07, 8.47, 9.51, 10.16, 11.81]                   #  Marginal generation cost of SGs   in    SGs_ buses:2_3_4_5_27_30
cⁿˡ=cᵐ*10 

yˢᴳ²_0 = 1
yˢᴳ³_0 = 1
yˢᴳ⁴_0 = 0
yˢᴳ⁵_0 = 0
yˢᴳ²⁷_0 = 0
yˢᴳ³⁰_0 = 0


#-------------------------------------------------------
#-------------------Define primal model-------------------
#-------------------------------------------------------

model= Model()                     

@variable(model, yˢᴳ² >= 0)    # ,Bin  >= 0
@variable(model, yˢᴳ³ >= 0)            
@variable(model, yˢᴳ⁴ >= 0)           
@variable(model, yˢᴳ⁵ >= 0)           
@variable(model, yˢᴳ²⁷ >= 0)            
@variable(model, yˢᴳ³⁰ >= 0)  

@constraint(model, yˢᴳ² <= 1)      #  ψˢᴳ²_ᵐᵃˣ 
@constraint(model, yˢᴳ³ <= 1)      #  ψˢᴳ³_ᵐᵃˣ
@constraint(model, yˢᴳ⁴ <= 1)      #  ψˢᴳ⁴_ᵐᵃˣ
@constraint(model, yˢᴳ⁵ <= 1)      #  ψˢᴳ⁵_ᵐᵃˣ
@constraint(model, yˢᴳ²⁷ <= 1)     #  ψˢᴳ²⁷_ᵐᵃˣ
@constraint(model, yˢᴳ³⁰ <= 1)     #  ψˢᴳ³⁰_ᵐᵃˣ

@variable(model, Cᵁ²>=0)                                                
@variable(model, Cᵁ³>=0)                            
@variable(model, Cᵁ⁴>=0)                                
@variable(model, Cᵁ⁵>=0)                                
@variable(model, Cᵁ²⁷>=0)                               
@variable(model, Cᵁ³⁰>=0)                 

@variable(model, Pᴳ[1:9]) 
@variable(model, Qᴳ[1:9])  

@constraint(model, Cᵁ²>=(yˢᴳ²-yˢᴳ²_0)*cˢᵗ[1])     #  σˢᴳ²_ˢᵗ
@constraint(model, Cᵁ³>=(yˢᴳ³-yˢᴳ³_0)*cˢᵗ[2])     #  σˢᴳ³_ˢᵗ
@constraint(model, Cᵁ⁴>=(yˢᴳ⁴-yˢᴳ⁴_0)*cˢᵗ[3])     #  σˢᴳ⁴_ˢᵗ
@constraint(model, Cᵁ⁵>=(yˢᴳ⁵-yˢᴳ⁵_0)*cˢᵗ[4])     #  σˢᴳ⁵_ˢᵗ
@constraint(model, Cᵁ²⁷>=(yˢᴳ²⁷-yˢᴳ²⁷_0)*cˢᵗ[5])  #  σˢᴳ²⁷_ˢᵗ
@constraint(model, Cᵁ³⁰>=(yˢᴳ³⁰-yˢᴳ³⁰_0)*cˢᵗ[6])  #  σˢᴳ³⁰_ˢᵗ

@constraint(model, Pᴳ[1] <= yˢᴳ²*Pˢᴳₘₐₓ[1])    #  τᵖ_ᵘᵇ_ˢᴳ²     
@constraint(model, yˢᴳ²*Pˢᴳₘᵢₙ[1] <= Pᴳ[1])     #  τᵖ_ˡᵇ_ˢᴳ² 
@constraint(model, Pᴳ[2] <= yˢᴳ³*Pˢᴳₘₐₓ[2])    #  τᵖ_ᵘᵇ_ˢᴳ³
@constraint(model, yˢᴳ³*Pˢᴳₘᵢₙ[2] <= Pᴳ[2])     #  τᵖ_ˡᵇ_ˢᴳ³ 
@constraint(model, Pᴳ[3] <= yˢᴳ⁴*Pˢᴳₘₐₓ[3])     #  τᵖ_ᵘᵇ_ˢᴳ⁴
@constraint(model, yˢᴳ⁴*Pˢᴳₘᵢₙ[3] <= Pᴳ[3])     #  τᵖ_ˡᵇ_ˢᴳ⁴
@constraint(model, Pᴳ[4] <= yˢᴳ⁵*Pˢᴳₘₐₓ[4])     #  τᵖ_ᵘᵇ_ˢᴳ⁵
@constraint(model, yˢᴳ⁵*Pˢᴳₘᵢₙ[4] <= Pᴳ[4])     #  τᵖ_ˡᵇ_ˢᴳ⁵
@constraint(model, Pᴳ[5] <= yˢᴳ²⁷*Pˢᴳₘₐₓ[5])   #  τᵖ_ᵘᵇ_ˢᴳ²⁷ 
@constraint(model, yˢᴳ²⁷*Pˢᴳₘᵢₙ[5] <=Pᴳ[5])    #  τᵖ_ˡᵇ_ˢᴳ²⁷
@constraint(model, Pᴳ[6] <= yˢᴳ³⁰*Pˢᴳₘₐₓ[6])   #  τᵖ_ᵘᵇ_ˢᴳ³⁰     
@constraint(model, yˢᴳ³⁰*Pˢᴳₘᵢₙ[6] <= Pᴳ[6])    #  τᵖ_ˡᵇ_ˢᴳ³⁰

@constraint(model, Qᴳ[1] <= yˢᴳ²*Qˢᴳₘₐₓ[1])   # τᵠ_ᵘᵇ_ˢᴳ²
@constraint(model, yˢᴳ²*Qˢᴳₘᵢₙ[1] <= Qᴳ[1])   # τᵠ_ˡᵇ_ˢᴳ²    
@constraint(model, Qᴳ[2] <= yˢᴳ³*Qˢᴳₘₐₓ[2])   # τᵠ_ᵘᵇ_ˢᴳ³      
@constraint(model, yˢᴳ³*Qˢᴳₘᵢₙ[2] <= Qᴳ[2])   # τᵠ_ˡᵇ_ˢᴳ³        
@constraint(model, Qᴳ[3] <= yˢᴳ⁴*Qˢᴳₘₐₓ[3])   # τᵠ_ᵘᵇ_ˢᴳ⁴       
@constraint(model, yˢᴳ⁴*Qˢᴳₘᵢₙ[3] <= Qᴳ[3])   # τᵠ_ˡᵇ_ˢᴳ⁴   
@constraint(model, Qᴳ[4] <= yˢᴳ⁵*Qˢᴳₘₐₓ[4])   # τᵠ_ᵘᵇ_ˢᴳ⁵       
@constraint(model, yˢᴳ⁵*Qˢᴳₘᵢₙ[4] <= Qᴳ[4])    # τᵠ_ˡᵇ_ˢᴳ⁵
@constraint(model, Qᴳ[5] <= yˢᴳ²⁷*Qˢᴳₘₐₓ[5])  # τᵠ_ᵘᵇ_ˢᴳ²⁷       
@constraint(model, yˢᴳ²⁷*Qˢᴳₘᵢₙ[5] <=Qᴳ[5])   # τᵠ_ˡᵇ_ˢᴳ²⁷
@constraint(model, Qᴳ[6] <= yˢᴳ³⁰*Qˢᴳₘₐₓ[6])  # τᵠ_ᵘᵇ_ˢᴳ³⁰      
@constraint(model, yˢᴳ³⁰*Qˢᴳₘᵢₙ[6] <= Qᴳ[6])  # τᵠ_ˡᵇ_ˢᴳ³⁰      

@constraint(model, Pᴳ[7] <= α_VSG*Pⱽˢᴳₘₐₓ[1])      # τᵖ_ᵘᵇ_VSG_1                
@constraint(model, 0 <= Pᴳ[7])       
@constraint(model, Pᴳ[8] <= α_IBG*Pᴵᴮᴳₘₐₓ[1])     # τᵖ_ᵘᵇ_IBG_23   
@constraint(model, 0 <= Pᴳ[8])         
@constraint(model, Pᴳ[9] <= α_IBG*Pᴵᴮᴳₘₐₓ[2])     # τᵖ_ᵘᵇ_IBG_24  
@constraint(model, 0 <= Pᴳ[9])        

@constraint(model, Qᴳ[7] <= Qⱽˢᴳₘₐₓ[1])    # τᵠ_ᵘᵇ_VSG_1   
@constraint(model, Qⱽˢᴳₘᵢₙ[1] <= Qᴳ[7])    # τᵠ_ˡᵇ_VSG_1
@constraint(model, Qᴳ[8] <= ratio*Qᴵᴮᴳₘₐₓ[1])    # τᵠ_ᵘᵇ_IBG_23   
@constraint(model, ratio*Qᴵᴮᴳₘᵢₙ[1] <= Qᴳ[8])    # τᵠ_ˡᵇ_IBG_23
@constraint(model, Qᴳ[9] <= ratio*Qᴵᴮᴳₘₐₓ[2])    # τᵠ_ᵘᵇ_IBG_24   
@constraint(model, ratio*Qᴵᴮᴳₘᵢₙ[2] <= Qᴳ[9])    # τᵠ_ˡᵇ_IBG_24

@constraint(model, [ Sᵐᵃˣ_gc[1], Pᴳ[1], Qᴳ[1] ] in SecondOrderCone()  )   #  υ_1_ˢᴳ²，  υ_2_ˢᴳ²，  υ_3_ˢᴳ²
@constraint(model, [ Sᵐᵃˣ_gc[2], Pᴳ[2], Qᴳ[2] ] in SecondOrderCone()  )   #  υ_1_ˢᴳ³，  υ_2_ˢᴳ³，  υ_3_ˢᴳ³
@constraint(model, [ Sᵐᵃˣ_gc[3], Pᴳ[3], Qᴳ[3] ] in SecondOrderCone()  )   #  υ_1_ˢᴳ⁴，  υ_2_ˢᴳ⁴，  υ_3_ˢᴳ⁴
@constraint(model, [ Sᵐᵃˣ_gc[4], Pᴳ[4], Qᴳ[4] ] in SecondOrderCone()  )   #  υ_1_ˢᴳ⁵，  υ_2_ˢᴳ⁵，  υ_3_ˢᴳ⁵
@constraint(model, [ Sᵐᵃˣ_gc[5], Pᴳ[5], Qᴳ[5] ] in SecondOrderCone()  )  #  υ_1_ˢᴳ²⁷，  υ_2_ˢᴳ²⁷，  υ_3_ˢᴳ²⁷
@constraint(model, [ Sᵐᵃˣ_gc[6], Pᴳ[6], Qᴳ[6] ] in SecondOrderCone()  )  #  υ_1_ˢᴳ³⁰，  υ_2_ˢᴳ³⁰，  υ_3_ˢᴳ³⁰
@constraint(model, [ Sᵐᵃˣ_gv[1], Pᴳ[7], Qᴳ[7] ] in SecondOrderCone()  )    #  υ_1_VSG_1，  υ_2_VSG_1，  υ_3_VSG_1
@constraint(model, [ Sᵐᵃˣ_c[1], Pᴳ[8], Qᴳ[8] ] in SecondOrderCone()  )    #  υ_1_IBG_23，  υ_2_IBG_23，  υ_3_IBG_23
@constraint(model, [ Sᵐᵃˣ_c[2], Pᴳ[9], Qᴳ[9] ] in SecondOrderCone()  )    #  υ_1_IBG_24，  υ_2_IBG_24，  υ_3_IBG_24

@constraint(model, sum(Pᴳ) == Pᴰ  )    #   λᴱ
@constraint(model, sum(Qᴳ) == Qᴰ )    #    ϕ



#--------------------------Voltage stability constraints--------------------------

@variable(model, z_23)
@variable(model, z_24)

@variable(model, ηₘ_1[1:15] >= 0)       
@variable(model, ηₘ_2[1:6])            

    # 15+6
    @constraint(model, ηₘ_1[1]>=yˢᴳ²+yˢᴳ³-1)    # γ_ᵐⁱⁿ[1]
    @constraint(model, ηₘ_1[1]<=yˢᴳ²)              # γ_ᵐᵃˣ[1,1]
    @constraint(model, ηₘ_1[1]<=yˢᴳ³)              # γ_ᵐᵃˣ[2,1]
    @constraint(model, ηₘ_1[2]>=yˢᴳ²+yˢᴳ⁴-1)    # γ_ᵐⁱⁿ[2]
    @constraint(model, ηₘ_1[2]<=yˢᴳ²)              # γ_ᵐᵃˣ[1,2] 
    @constraint(model, ηₘ_1[2]<=yˢᴳ⁴)              # γ_ᵐᵃˣ[2,2]
    @constraint(model, ηₘ_1[3]>=yˢᴳ²+yˢᴳ⁵-1)    # γ_ᵐⁱⁿ[3]
    @constraint(model, ηₘ_1[3]<=yˢᴳ²)              # γ_ᵐᵃˣ[1,3]
    @constraint(model, ηₘ_1[3]<=yˢᴳ⁵)              # γ_ᵐᵃˣ[2,3]
    @constraint(model, ηₘ_1[4]>=yˢᴳ²+yˢᴳ²⁷-1)   # γ_ᵐⁱⁿ[4]
    @constraint(model, ηₘ_1[4]<=yˢᴳ²)              # γ_ᵐᵃˣ[1,4]
    @constraint(model, ηₘ_1[4]<=yˢᴳ²⁷)             # γ_ᵐᵃˣ[2,4]
    @constraint(model, ηₘ_1[5]>=yˢᴳ²+yˢᴳ³⁰-1)   # γ_ᵐⁱⁿ[5]
    @constraint(model, ηₘ_1[5]<=yˢᴳ²)              # γ_ᵐᵃˣ[1,5]
    @constraint(model, ηₘ_1[5]<=yˢᴳ³⁰)             # γ_ᵐᵃˣ[2,5]

    @constraint(model, ηₘ_1[6]>=yˢᴳ³+yˢᴳ⁴-1)   # γ_ᵐⁱⁿ[6]
    @constraint(model, ηₘ_1[6]<=yˢᴳ³)             # γ_ᵐᵃˣ[1,6]
    @constraint(model, ηₘ_1[6]<=yˢᴳ⁴)             # γ_ᵐᵃˣ[2,6]
    @constraint(model, ηₘ_1[7]>=yˢᴳ³+yˢᴳ⁵-1)   # γ_ᵐⁱⁿ[7]
    @constraint(model, ηₘ_1[7]<=yˢᴳ³)             # γ_ᵐᵃˣ[1,7]
    @constraint(model, ηₘ_1[7]<=yˢᴳ⁵)             # γ_ᵐᵃˣ[2,7]
    @constraint(model, ηₘ_1[8]>=yˢᴳ³+yˢᴳ²⁷-1)  # γ_ᵐⁱⁿ[8] 
    @constraint(model, ηₘ_1[8]<=yˢᴳ³)             # γ_ᵐᵃˣ[1,8]
    @constraint(model, ηₘ_1[8]<=yˢᴳ²⁷)            # γ_ᵐᵃˣ[2,8]
    @constraint(model, ηₘ_1[9]>=yˢᴳ³+yˢᴳ³⁰-1)  # γ_ᵐⁱⁿ[9] 
    @constraint(model, ηₘ_1[9]<=yˢᴳ³)             # γ_ᵐᵃˣ[1,9]
    @constraint(model, ηₘ_1[9]<=yˢᴳ³⁰)            # γ_ᵐᵃˣ[2,9]

    @constraint(model, ηₘ_1[10]>=yˢᴳ⁴+yˢᴳ⁵-1)   # γ_ᵐⁱⁿ[10]
    @constraint(model, ηₘ_1[10]<=yˢᴳ⁴)             # γ_ᵐᵃˣ[1,10]
    @constraint(model, ηₘ_1[10]<=yˢᴳ⁵)             # γ_ᵐᵃˣ[2,10]
    @constraint(model, ηₘ_1[11]>=yˢᴳ⁴+yˢᴳ²⁷-1)  # γ_ᵐⁱⁿ[11] 
    @constraint(model, ηₘ_1[11]<=yˢᴳ⁴)             # γ_ᵐᵃˣ[1,11]
    @constraint(model, ηₘ_1[11]<=yˢᴳ²⁷)            # γ_ᵐᵃˣ[2,11]
    @constraint(model, ηₘ_1[12]>=yˢᴳ⁴+yˢᴳ³⁰-1)  # γ_ᵐⁱⁿ[12] 
    @constraint(model, ηₘ_1[12]<=yˢᴳ⁴)             # γ_ᵐᵃˣ[1,12]
    @constraint(model, ηₘ_1[12]<=yˢᴳ³⁰)            # γ_ᵐᵃˣ[2,12]

    @constraint(model, ηₘ_1[13]>=yˢᴳ⁵+yˢᴳ²⁷-1)  # γ_ᵐⁱⁿ[13] 
    @constraint(model, ηₘ_1[13]<=yˢᴳ⁵)             # γ_ᵐᵃˣ[1,13]
    @constraint(model, ηₘ_1[13]<=yˢᴳ²⁷)            # γ_ᵐᵃˣ[2,13]       
    @constraint(model, ηₘ_1[14]>=yˢᴳ⁵+yˢᴳ³⁰-1)  # γ_ᵐⁱⁿ[14]  
    @constraint(model, ηₘ_1[14]<=yˢᴳ⁵)             # γ_ᵐᵃˣ[1,14]
    @constraint(model, ηₘ_1[14]<=yˢᴳ³⁰)            # γ_ᵐᵃˣ[2,14]

    @constraint(model, ηₘ_1[15]>=yˢᴳ²⁷+yˢᴳ³⁰-1)  # γ_ᵐⁱⁿ[15] 
    @constraint(model, ηₘ_1[15]<=yˢᴳ²⁷)             # γ_ᵐᵃˣ[1,15]
    @constraint(model, ηₘ_1[15]<=yˢᴳ³⁰)             # γ_ᵐᵃˣ[2,15]

    @constraint(model, ηₘ_2[1]==yˢᴳ²*α_VSG)        # γ[1]
    @constraint(model, ηₘ_2[2]==yˢᴳ³*α_VSG)        # γ[2]
    @constraint(model, ηₘ_2[3]==yˢᴳ⁴*α_VSG)        # γ[3]
    @constraint(model, ηₘ_2[4]==yˢᴳ⁵*α_VSG)        # γ[4]
    @constraint(model, ηₘ_2[5]==yˢᴳ²⁷*α_VSG)       # γ[5]
    @constraint(model, ηₘ_2[6]==yˢᴳ³⁰*α_VSG)       # γ[6]

    @constraint(model, z_23 ==  ( ( K_c_gc[1, 1] *yˢᴳ² +K_c_gc[1, 2] *yˢᴳ³ +K_c_gc[1, 3] *yˢᴳ⁴ 
    +K_c_gc[1, 4] *yˢᴳ⁵ +K_c_gc[1, 5] *yˢᴳ²⁷ +K_c_gc[1, 6] *yˢᴳ³⁰ )   +α_VSG*K_c_gv[1, 1]
    + sum(K_c_m[1, 1:15] .*ηₘ_1) +  sum(K_c_m[1, 16:end] .*ηₘ_2   ) ) )     #   ξ_gf_23

    @constraint(model, z_24 == ( ( K_c_gc[2, 1] *yˢᴳ² +K_c_gc[2, 2] *yˢᴳ³ +K_c_gc[2, 3] *yˢᴳ⁴ 
    +K_c_gc[2, 4] *yˢᴳ⁵ +K_c_gc[2, 5] *yˢᴳ²⁷ +K_c_gc[2, 6] *yˢᴳ³⁰ )   +α_VSG*K_c_gv[2, 1]
    + sum(K_c_m[2, 1:15] .*ηₘ_1) +  sum(K_c_m[2, 16:end] .*ηₘ_2   ) ) )     #   ξ_gf_24

    @constraint(model, [Qᴳ[8] + 1/2*z_23*S_Base,  Pᴳ[8],  Qᴳ[8]] in SecondOrderCone()  )     # λ_1_23,   λ_2_23,   μ_23
    @constraint(model, [Qᴳ[9] + 1/2*z_24*S_Base,  Pᴳ[9],  Qᴳ[9]] in SecondOrderCone()  )     # λ_1_24,   λ_2_24,   μ_24


#--------------------------------
#  Model solving
#--------------------------------

cost_onoff_Primal=sum(Cᵁ²)+sum(Cᵁ³)+sum(Cᵁ⁴)+sum(Cᵁ⁵)+sum(Cᵁ²⁷)+sum(Cᵁ³⁰)
cost_nl_Primal=sum(cⁿˡ[1].*(yˢᴳ²))+sum(cⁿˡ[2].*(yˢᴳ³))+sum(cⁿˡ[3].*(yˢᴳ⁴))+sum(cⁿˡ[4].*(yˢᴳ⁵))+sum(cⁿˡ[5].*(yˢᴳ²⁷))+sum(cⁿˡ[6].*(yˢᴳ³⁰))    
cost_gene_Primal=sum(cᵐ[1].*Pᴳ[1])+ sum(cᵐ[2].*Pᴳ[2])+sum(cᵐ[3].*Pᴳ[3])+sum(cᵐ[4].*Pᴳ[4])+sum(cᵐ[5].*Pᴳ[5])+sum(cᵐ[6].*Pᴳ[6])   
obj=cost_onoff_Primal +cost_nl_Primal +cost_gene_Primal 

@objective(model, Min,  obj)  # single-level objective function
#-------Solve and Output Results
set_optimizer(model,  Gurobi.Optimizer)
#set_optimizer_attribute(model, "QCPDual",  1)
optimize!(model)


dual_model=dualize(model)
set_optimizer(dual_model,  Gurobi.Optimizer)
#set_optimizer_attribute(dual_model, "QCPDual",  1)
optimize!(dual_model)

# 将完整模型信息写入文件
open("dual_model_info.txt", "w") do io
    # 保存目标函数
    println(io, "=== 目标函数 ===")
    println(io, objective_sense(dual_model))
    println(io, objective_function(dual_model))
    
    # 保存所有约束
    println(io, "\n=== 所有约束条件 ===")
    for (F, S) in list_of_constraint_types(dual_model)
        println(io, "\n--- 约束类型: $F in $S ---")
        for con in all_constraints(dual_model, F, S)
            println(io, con)
        end
    end
end

println("已保存到 dual_model_info.txt")


#-------------------------------------------------------
#-------------------Define dual model-------------------
#-------------------------------------------------------

model= Model()                     

@variable(model, ψˢᴳ²_ᵐᵃˣ >= 0)    
@variable(model, ψˢᴳ³_ᵐᵃˣ >= 0)
@variable(model, ψˢᴳ⁴_ᵐᵃˣ >= 0)
@variable(model, ψˢᴳ⁵_ᵐᵃˣ >= 0)
@variable(model, ψˢᴳ²⁷_ᵐᵃˣ >= 0)
@variable(model, ψˢᴳ³⁰_ᵐᵃˣ >= 0)   
@variable(model, σˢᴳ²_ˢᵗ >= 0)
@variable(model, σˢᴳ³_ˢᵗ >= 0)
@variable(model, σˢᴳ⁴_ˢᵗ >= 0)
@variable(model, σˢᴳ⁵_ˢᵗ >= 0)
@variable(model, σˢᴳ²⁷_ˢᵗ >= 0)
@variable(model, σˢᴳ³⁰_ˢᵗ >= 0)
@variable(model, τᵖ_ᵘᵇ_ˢᴳ² >= 0)
@variable(model, τᵖ_ˡᵇ_ˢᴳ² >= 0)
@variable(model, τᵖ_ᵘᵇ_ˢᴳ³ >= 0)
@variable(model, τᵖ_ˡᵇ_ˢᴳ³ >= 0)
@variable(model, τᵖ_ᵘᵇ_ˢᴳ⁴ >= 0)
@variable(model, τᵖ_ˡᵇ_ˢᴳ⁴ >= 0)
@variable(model, τᵖ_ᵘᵇ_ˢᴳ⁵ >= 0)
@variable(model, τᵖ_ˡᵇ_ˢᴳ⁵ >= 0)
@variable(model, τᵖ_ᵘᵇ_ˢᴳ²⁷ >= 0)
@variable(model, τᵖ_ˡᵇ_ˢᴳ²⁷ >= 0)
@variable(model, τᵖ_ᵘᵇ_ˢᴳ³⁰ >= 0)
@variable(model, τᵖ_ˡᵇ_ˢᴳ³⁰ >= 0)
@variable(model, τᵠ_ᵘᵇ_ˢᴳ² >= 0)
@variable(model, τᵠ_ˡᵇ_ˢᴳ² >= 0)
@variable(model, τᵠ_ᵘᵇ_ˢᴳ³ >= 0)
@variable(model, τᵠ_ˡᵇ_ˢᴳ³ >= 0)
@variable(model, τᵠ_ᵘᵇ_ˢᴳ⁴ >= 0)
@variable(model, τᵠ_ˡᵇ_ˢᴳ⁴ >= 0)
@variable(model, τᵠ_ᵘᵇ_ˢᴳ⁵ >= 0)
@variable(model, τᵠ_ˡᵇ_ˢᴳ⁵ >= 0)
@variable(model, τᵠ_ᵘᵇ_ˢᴳ²⁷ >= 0)
@variable(model, τᵠ_ˡᵇ_ˢᴳ²⁷ >= 0)
@variable(model, τᵠ_ᵘᵇ_ˢᴳ³⁰ >= 0)
@variable(model, τᵠ_ˡᵇ_ˢᴳ³⁰ >= 0)
@variable(model, τᵖ_ᵘᵇ_VSG_1 >= 0)
@variable(model, τᵖ_ᵘᵇ_IBG_23 >= 0)
@variable(model, τᵖ_ᵘᵇ_IBG_24 >= 0)
@variable(model, τᵠ_ᵘᵇ_VSG_1 >= 0)
@variable(model, τᵠ_ˡᵇ_VSG_1 >= 0)
@variable(model, τᵠ_ᵘᵇ_IBG_23 >= 0)
@variable(model, τᵠ_ˡᵇ_IBG_23 >= 0)
@variable(model, τᵠ_ᵘᵇ_IBG_24 >= 0)
@variable(model, τᵠ_ˡᵇ_IBG_24 >= 0)
@variable(model, υ_1_ˢᴳ²)    
@variable(model, υ_2_ˢᴳ²)
@variable(model, υ_3_ˢᴳ² >= 0)
@variable(model, υ_1_ˢᴳ³)
@variable(model, υ_2_ˢᴳ³)
@variable(model, υ_3_ˢᴳ³ >= 0)
@variable(model, υ_1_ˢᴳ⁴)
@variable(model, υ_2_ˢᴳ⁴)
@variable(model, υ_3_ˢᴳ⁴ >= 0)
@variable(model, υ_1_ˢᴳ⁵)
@variable(model, υ_2_ˢᴳ⁵)
@variable(model, υ_3_ˢᴳ⁵ >= 0)
@variable(model, υ_1_ˢᴳ²⁷)
@variable(model, υ_2_ˢᴳ²⁷)
@variable(model, υ_3_ˢᴳ²⁷ >= 0)
@variable(model, υ_1_ˢᴳ³⁰)
@variable(model, υ_2_ˢᴳ³⁰)
@variable(model, υ_3_ˢᴳ³⁰ >= 0)
@variable(model, υ_1_VSG_1)
@variable(model, υ_2_VSG_1)
@variable(model, υ_3_VSG_1 >= 0)
@variable(model, υ_1_IBG_23)
@variable(model, υ_2_IBG_23)
@variable(model, υ_3_IBG_23 >= 0)
@variable(model, υ_1_IBG_24)
@variable(model, υ_2_IBG_24)
@variable(model, υ_3_IBG_24 >= 0)
@variable(model, λᴱ )
@variable(model, ϕ )
    
@variable(model, γ_ᵐⁱⁿ[1:15] >= 0)
@variable(model, γ_ᵐᵃˣ[1:2,1:15] >= 0)
@variable(model, γ[1:6] >= 0)

@variable(model, ξ_gf_23 )
@variable(model, ξ_gf_24 )
@variable(model, λ_1_23 )
@variable(model, λ_2_23 )
@variable(model, μ_23 >= 0)
@variable(model, λ_1_24 )
@variable(model, λ_2_24 )
@variable(model, μ_24 >= 0)


#--------------------------------
#  Dual constraints
#--------------------------------

# ============================================================
# Cᵁ²...Cᵁ³⁰
@constraint(model, 1 - σˢᴳ²_ˢᵗ >= 0)
@constraint(model, 1 - σˢᴳ³_ˢᵗ >= 0)
@constraint(model, 1 - σˢᴳ⁴_ˢᵗ >= 0)
@constraint(model, 1 - σˢᴳ⁵_ˢᵗ >= 0)
@constraint(model, 1 - σˢᴳ²⁷_ˢᵗ >= 0)
@constraint(model, 1 - σˢᴳ³⁰_ˢᵗ >= 0)

# ============================================================
# yˢᴳ²...yˢᴳ³⁰
@constraint(model, cⁿˡ[1] + ψˢᴳ²_ᵐᵃˣ - Pˢᴳₘₐₓ[1]*τᵖ_ᵘᵇ_ˢᴳ² + Pˢᴳₘᵢₙ[1]*τᵖ_ˡᵇ_ˢᴳ² 
                    - Qˢᴳₘₐₓ[1]*τᵠ_ᵘᵇ_ˢᴳ² + Qˢᴳₘᵢₙ[1]*τᵠ_ˡᵇ_ˢᴳ² 
                    + K_c_gc[1, 1]*ξ_gf_23 + K_c_gc[2, 1]*ξ_gf_24 + α_VSG*γ[1]
                    + cˢᵗ[1]*σˢᴳ²_ˢᵗ 
                    + sum(γ_ᵐⁱⁿ[1:5]) - sum(γ_ᵐᵃˣ[1,1:5]) >= 0 )
@constraint(model, cⁿˡ[2] + ψˢᴳ³_ᵐᵃˣ - Pˢᴳₘₐₓ[2]*τᵖ_ᵘᵇ_ˢᴳ³ + Pˢᴳₘᵢₙ[2]*τᵖ_ˡᵇ_ˢᴳ³ 
                    - Qˢᴳₘₐₓ[2]*τᵠ_ᵘᵇ_ˢᴳ³ + Qˢᴳₘᵢₙ[2]*τᵠ_ˡᵇ_ˢᴳ³ 
                    + K_c_gc[1, 2]*ξ_gf_23 + K_c_gc[2, 2]*ξ_gf_24 + α_VSG*γ[2]
                    + cˢᵗ[2]*σˢᴳ³_ˢᵗ 
                    + sum(γ_ᵐⁱⁿ[6:9]) - sum(γ_ᵐᵃˣ[1,6:9]) + γ_ᵐⁱⁿ[1] - γ_ᵐᵃˣ[2,1] >= 0 )
@constraint(model, cⁿˡ[3] + ψˢᴳ⁴_ᵐᵃˣ - Pˢᴳₘₐₓ[3]*τᵖ_ᵘᵇ_ˢᴳ⁴ + Pˢᴳₘᵢₙ[3]*τᵖ_ˡᵇ_ˢᴳ⁴ 
                    - Qˢᴳₘₐₓ[3]*τᵠ_ᵘᵇ_ˢᴳ⁴ + Qˢᴳₘᵢₙ[3]*τᵠ_ˡᵇ_ˢᴳ⁴ 
                    + K_c_gc[1, 3]*ξ_gf_23 + K_c_gc[2, 3]*ξ_gf_24 + α_VSG*γ[3]
                    + cˢᵗ[3]*σˢᴳ⁴_ˢᵗ 
                    + sum(γ_ᵐⁱⁿ[10:12]) - sum(γ_ᵐᵃˣ[1,10:12]) + γ_ᵐⁱⁿ[2] + γ_ᵐⁱⁿ[6] - γ_ᵐᵃˣ[2,2] - γ_ᵐᵃˣ[2,6] >= 0 )
@constraint(model, cⁿˡ[4] + ψˢᴳ⁵_ᵐᵃˣ - Pˢᴳₘₐₓ[4]*τᵖ_ᵘᵇ_ˢᴳ⁵ + Pˢᴳₘᵢₙ[4]*τᵖ_ˡᵇ_ˢᴳ⁵ 
                    - Qˢᴳₘₐₓ[4]*τᵠ_ᵘᵇ_ˢᴳ⁵ + Qˢᴳₘᵢₙ[4]*τᵠ_ˡᵇ_ˢᴳ⁵ 
                    + K_c_gc[1, 4]*ξ_gf_23 + K_c_gc[2, 4]*ξ_gf_24 + α_VSG*γ[4]
                    + cˢᵗ[4]*σˢᴳ⁵_ˢᵗ 
                    + sum(γ_ᵐⁱⁿ[13:14]) - sum(γ_ᵐᵃˣ[1,13:14]) + γ_ᵐⁱⁿ[3] + γ_ᵐⁱⁿ[7] + γ_ᵐⁱⁿ[10] - γ_ᵐᵃˣ[2,3] - γ_ᵐᵃˣ[2,7] - γ_ᵐᵃˣ[2,10] >= 0 )
@constraint(model, cⁿˡ[5] + ψˢᴳ²⁷_ᵐᵃˣ - Pˢᴳₘₐₓ[5]*τᵖ_ᵘᵇ_ˢᴳ²⁷ + Pˢᴳₘᵢₙ[5]*τᵖ_ˡᵇ_ˢᴳ²⁷ 
                    - Qˢᴳₘₐₓ[5]*τᵠ_ᵘᵇ_ˢᴳ²⁷ + Qˢᴳₘᵢₙ[5]*τᵠ_ˡᵇ_ˢᴳ²⁷ 
                    + K_c_gc[1, 5]*ξ_gf_23 + K_c_gc[2, 5]*ξ_gf_24 + α_VSG*γ[5]
                    + cˢᵗ[5]*σˢᴳ²⁷_ˢᵗ 
                    + γ_ᵐⁱⁿ[15] + γ_ᵐⁱⁿ[4] + γ_ᵐⁱⁿ[8] + γ_ᵐⁱⁿ[11] + γ_ᵐⁱⁿ[13]   
                    - γ_ᵐᵃˣ[1,15] - γ_ᵐᵃˣ[2,4] - γ_ᵐᵃˣ[2,8] - γ_ᵐᵃˣ[2,11] - γ_ᵐᵃˣ[2,13] >= 0 )
@constraint(model, cⁿˡ[6] + ψˢᴳ³⁰_ᵐᵃˣ - Pˢᴳₘₐₓ[6]*τᵖ_ᵘᵇ_ˢᴳ³⁰ + Pˢᴳₘᵢₙ[6]*τᵖ_ˡᵇ_ˢᴳ³⁰ 
                    - Qˢᴳₘₐₓ[6]*τᵠ_ᵘᵇ_ˢᴳ³⁰ + Qˢᴳₘᵢₙ[6]*τᵠ_ˡᵇ_ˢᴳ³⁰ 
                    + K_c_gc[1, 6]*ξ_gf_23 + K_c_gc[2, 6]*ξ_gf_24 + α_VSG*γ[6]
                    + cˢᵗ[6]*σˢᴳ³⁰_ˢᵗ 
                    + γ_ᵐⁱⁿ[15] + γ_ᵐⁱⁿ[5] + γ_ᵐⁱⁿ[9] + γ_ᵐⁱⁿ[12] + γ_ᵐⁱⁿ[14]   
                    - γ_ᵐᵃˣ[2,15] - γ_ᵐᵃˣ[2,5] - γ_ᵐᵃˣ[2,9] - γ_ᵐᵃˣ[2,12] - γ_ᵐᵃˣ[2,14] >= 0 )

# ============================================================
# ηₘ_1[1:15]
for k in 1:15
    @constraint(model,
        K_c_m[1,k]*ξ_gf_23 + K_c_m[2,k]*ξ_gf_24 + γ_ᵐᵃˣ[1,k] + γ_ᵐᵃˣ[2,k] - γ_ᵐⁱⁿ[k] >= 0)
end

# ============================================================
# ηₘ_2[1:6]
for j in 1:6
    @constraint(model,
        K_c_m[1, 15+j]*ξ_gf_23 + K_c_m[2, 15+j]*ξ_gf_24 - γ[j] >= 0)
end

# ============================================================
# z_23, z_24
@constraint(model,  0.5*S_Base*μ_23 - ξ_gf_23 >= 0)    # SOC sign
@constraint(model,  0.5*S_Base*μ_24 - ξ_gf_24 >= 0)

# ============================================================
# Pᴳ[1]...Pᴳ[6]
@constraint(model, cᵐ[1] + τᵖ_ᵘᵇ_ˢᴳ² - τᵖ_ˡᵇ_ˢᴳ² + υ_1_ˢᴳ² - λᴱ >= 0)  # SOC sign
@constraint(model, cᵐ[2] + τᵖ_ᵘᵇ_ˢᴳ³ - τᵖ_ˡᵇ_ˢᴳ³ + υ_1_ˢᴳ³ - λᴱ >= 0)
@constraint(model, cᵐ[3] + τᵖ_ᵘᵇ_ˢᴳ⁴ - τᵖ_ˡᵇ_ˢᴳ⁴ + υ_1_ˢᴳ⁴ - λᴱ >= 0)
@constraint(model, cᵐ[4] + τᵖ_ᵘᵇ_ˢᴳ⁵ - τᵖ_ˡᵇ_ˢᴳ⁵ + υ_1_ˢᴳ⁵ - λᴱ >= 0)
@constraint(model, cᵐ[5] + τᵖ_ᵘᵇ_ˢᴳ²⁷ - τᵖ_ˡᵇ_ˢᴳ²⁷ + υ_1_ˢᴳ²⁷ - λᴱ >= 0)
@constraint(model, cᵐ[6] + τᵖ_ᵘᵇ_ˢᴳ³⁰ - τᵖ_ˡᵇ_ˢᴳ³⁰ + υ_1_ˢᴳ³⁰ - λᴱ >= 0)

# ============================================================
# Pᴳ[7] VSG_1，Pᴳ[8] IBG_23，Pᴳ[9] IBG_24
@constraint(model, τᵖ_ᵘᵇ_VSG_1 + υ_1_VSG_1 - λᴱ >= 0)   # SOC sign

@constraint(model, τᵖ_ᵘᵇ_IBG_23 + υ_1_IBG_23 + λ_1_23 - λᴱ >= 0)
@constraint(model, τᵖ_ᵘᵇ_IBG_24 + υ_1_IBG_24 + λ_1_24 - λᴱ >= 0)

# ============================================================
# Qᴳ[1]...Qᴳ[6] 
@constraint(model, τᵠ_ᵘᵇ_ˢᴳ² - τᵠ_ˡᵇ_ˢᴳ² + υ_2_ˢᴳ² - ϕ >= 0)  # SOC sign
@constraint(model, τᵠ_ᵘᵇ_ˢᴳ³ - τᵠ_ˡᵇ_ˢᴳ³ + υ_2_ˢᴳ³ - ϕ >= 0)
@constraint(model, τᵠ_ᵘᵇ_ˢᴳ⁴ - τᵠ_ˡᵇ_ˢᴳ⁴ + υ_2_ˢᴳ⁴ - ϕ >= 0)
@constraint(model, τᵠ_ᵘᵇ_ˢᴳ⁵ - τᵠ_ˡᵇ_ˢᴳ⁵ + υ_2_ˢᴳ⁵ - ϕ >= 0)
@constraint(model, τᵠ_ᵘᵇ_ˢᴳ²⁷ - τᵠ_ˡᵇ_ˢᴳ²⁷ + υ_2_ˢᴳ²⁷ - ϕ >= 0)
@constraint(model, τᵠ_ᵘᵇ_ˢᴳ³⁰ - τᵠ_ˡᵇ_ˢᴳ³⁰ + υ_2_ˢᴳ³⁰ - ϕ >= 0)

# ============================================================
# Qᴳ[7] VSG_1，Qᴳ[8] IBG_23，Qᴳ[9] IBG_24
@constraint(model, τᵠ_ᵘᵇ_VSG_1 - τᵠ_ˡᵇ_VSG_1 + υ_2_VSG_1 - ϕ >= 0)  # SOC sign

@constraint(model, τᵠ_ᵘᵇ_IBG_23 - τᵠ_ˡᵇ_IBG_23 + υ_2_IBG_23
                   + λ_2_23 - μ_23 - ϕ >= 0)
@constraint(model, τᵠ_ᵘᵇ_IBG_24 - τᵠ_ˡᵇ_IBG_24 + υ_2_IBG_24
                   + λ_2_24 - μ_24 - ϕ >= 0)

# ============================================================
# SOC
# ============================================================
@constraint(model, [υ_3_ˢᴳ², υ_1_ˢᴳ², υ_2_ˢᴳ²] in SecondOrderCone())
@constraint(model, [υ_3_ˢᴳ³, υ_1_ˢᴳ³, υ_2_ˢᴳ³] in SecondOrderCone())
@constraint(model, [υ_3_ˢᴳ⁴, υ_1_ˢᴳ⁴, υ_2_ˢᴳ⁴] in SecondOrderCone())
@constraint(model, [υ_3_ˢᴳ⁵, υ_1_ˢᴳ⁵, υ_2_ˢᴳ⁵] in SecondOrderCone())
@constraint(model, [υ_3_ˢᴳ²⁷, υ_1_ˢᴳ²⁷, υ_2_ˢᴳ²⁷] in SecondOrderCone())
@constraint(model, [υ_3_ˢᴳ³⁰, υ_1_ˢᴳ³⁰, υ_2_ˢᴳ³⁰] in SecondOrderCone())
@constraint(model, [υ_3_VSG_1, υ_1_VSG_1, υ_2_VSG_1] in SecondOrderCone())
@constraint(model, [υ_3_IBG_23, υ_1_IBG_23, υ_2_IBG_23] in SecondOrderCone())
@constraint(model, [υ_3_IBG_24, υ_1_IBG_24, υ_2_IBG_24] in SecondOrderCone())

@constraint(model, [μ_23, λ_1_23, λ_2_23] in SecondOrderCone())
@constraint(model, [μ_24, λ_1_24, λ_2_24] in SecondOrderCone())

# ============================================================
# Max obj
@objective(model, Max,

    λᴱ * Pᴰ + ϕ * Qᴰ

    - sum(γ_ᵐⁱⁿ[1,:])

    - ψˢᴳ²_ᵐᵃˣ - ψˢᴳ³_ᵐᵃˣ - ψˢᴳ⁴_ᵐᵃˣ
    - ψˢᴳ⁵_ᵐᵃˣ - ψˢᴳ²⁷_ᵐᵃˣ - ψˢᴳ³⁰_ᵐᵃˣ

    - σˢᴳ²_ˢᵗ  * cˢᵗ[1]*yˢᴳ²_0
    - σˢᴳ³_ˢᵗ  * cˢᵗ[2]*yˢᴳ³_0
    - σˢᴳ⁴_ˢᵗ  * cˢᵗ[3]*yˢᴳ⁴_0
    - σˢᴳ⁵_ˢᵗ  * cˢᵗ[4]*yˢᴳ⁵_0
    - σˢᴳ²⁷_ˢᵗ * cˢᵗ[5]*yˢᴳ²⁷_0
    - σˢᴳ³⁰_ˢᵗ * cˢᵗ[6]*yˢᴳ³⁰_0

    - υ_3_ˢᴳ²*Sᵐᵃˣ_gc[1]  - υ_3_ˢᴳ³*Sᵐᵃˣ_gc[2]  - υ_3_ˢᴳ⁴*Sᵐᵃˣ_gc[3]
    - υ_3_ˢᴳ⁵*Sᵐᵃˣ_gc[4]  - υ_3_ˢᴳ²⁷*Sᵐᵃˣ_gc[5] - υ_3_ˢᴳ³⁰*Sᵐᵃˣ_gc[6]
    - υ_3_VSG_1*Sᵐᵃˣ_gv[1] - υ_3_IBG_23*Sᵐᵃˣ_c[1] - υ_3_IBG_24*Sᵐᵃˣ_c[2]

    + ξ_gf_23 * α_VSG*K_c_gv[1,1]
    + ξ_gf_24 * α_VSG*K_c_gv[2,1]

    - α_VSG*Pⱽˢᴳₘₐₓ[1]*τᵖ_ᵘᵇ_VSG_1 - α_IBG*Pᴵᴮᴳₘₐₓ[1]*τᵖ_ᵘᵇ_IBG_23 - α_IBG*Pᴵᴮᴳₘₐₓ[2]*τᵖ_ᵘᵇ_IBG_24
    - Qⱽˢᴳₘₐₓ[1]*τᵠ_ᵘᵇ_VSG_1 + Qⱽˢᴳₘᵢₙ[1]*τᵠ_ˡᵇ_VSG_1 - ratio*Qᴵᴮᴳₘₐₓ[1]*τᵠ_ᵘᵇ_IBG_23 + ratio*Qᴵᴮᴳₘᵢₙ[1]*τᵠ_ˡᵇ_IBG_23
    - ratio*Qᴵᴮᴳₘₐₓ[2]*τᵠ_ᵘᵇ_IBG_24 + ratio*Qᴵᴮᴳₘᵢₙ[2]*τᵠ_ˡᵇ_IBG_24
)

set_optimizer(model, Gurobi.Optimizer)
optimize!(model)


println(model)









#--------------------------------
#  Model solving
#--------------------------------

cost_onoff_Primal=sum(Cᵁ²)+sum(Cᵁ³)+sum(Cᵁ⁴)+sum(Cᵁ⁵)+sum(Cᵁ²⁷)+sum(Cᵁ³⁰)
cost_nl_Primal=sum(cⁿˡ[1].*(yˢᴳ²))+sum(cⁿˡ[2].*(yˢᴳ³))+sum(cⁿˡ[3].*(yˢᴳ⁴))+sum(cⁿˡ[4].*(yˢᴳ⁵))+sum(cⁿˡ[5].*(yˢᴳ²⁷))+sum(cⁿˡ[6].*(yˢᴳ³⁰))    
cost_gene_Primal=sum(cᵐ[1].*Pᴳ[2])+ sum(cᵐ[2].*Pᴳ[3])+sum(cᵐ[3].*Pᴳ[4])+sum(cᵐ[4].*Pᴳ[5])+sum(cᵐ[5].*Pᴳ[27])+sum(cᵐ[6].*Pᴳ[30])   
obj=cost_onoff_Primal +cost_nl_Primal +cost_gene_Primal 

@objective(model, Min, obj)  # single-level objective function
#-------Solve and Output Results
set_optimizer(model, Gurobi.Optimizer)
set_optimizer_attribute(model, "QCPDual", 1)
optimize!(model)
