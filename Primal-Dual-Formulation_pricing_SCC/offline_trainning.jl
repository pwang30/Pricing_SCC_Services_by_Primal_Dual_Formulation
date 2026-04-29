function offline_trainning(zᵟ_c, matrix_ω)

    a=6
    b=15
#-----------------------------------Dataset Classification
    K_c_gc=zeros(2,a)   # define the linearized coffecoents for SGs
    K_c_gv=zeros(2,1)    # define the linearized coffecoents for VSGs
    K_c_m=zeros(2,a+b)   # define the linearized coffecoents for interactions of each two (V)SGs.   78=66+12


    #=
#-----------------------------------Define Offline-Training Model.  z₂₃²⁴
        Training_z₂₃²⁴= Model()
        #  @variable(Training_z₂₃²⁴, K_z₂₃²⁴[1:3])  # shape control
        @variable(Training_z₂₃²⁴, sh_1[1])
        @variable(Training_z₂₃²⁴, K_23_gc[1:a])  # buses of SGs are 2,3,4,5,27,30.
        @variable(Training_z₂₃²⁴, K_23_gv[1:1])   # buses of VSGs are 1,23,24.
        @variable(Training_z₂₃²⁴, K_23_m[1:a+b])   # pairs of (V)SGs
        @variable(Training_z₂₃²⁴, z_23_24[1:size(zᵟ_c,1)])   
        @variable(Training_z₂₃²⁴, penalty_z₂₃²⁴[1:size(zᵟ_c,1)])

        for i in 1:size(zᵟ_c,1)
            #@constraint(Training_z₂₃²⁴, z_23_24[i]==K_z₂₃²⁴[1]*(sum(K_23_gc .*matrix_ω[i,1:a]) +K_z₂₃²⁴[2]*sum(K_23_gv *matrix_ω[i,a+1]) +K_z₂₃²⁴[3]*sum(K_23_m .*matrix_ω[i,a+2:end])))
             @constraint(Training_z₂₃²⁴, z_23_24[i]==sh_1[1]-(sum(K_23_gc .*matrix_ω[i,1:a]) +sum(K_23_gv *matrix_ω[i,a+1]) +sum(K_23_m .*matrix_ω[i,a+2:end])))
           
            
            @constraint(Training_z₂₃²⁴, penalty_z₂₃²⁴[i]==z_23_24[i] -zᵟ_c[i,1])
        end

        @objective(Training_z₂₃²⁴, Min, sum(penalty_z₂₃²⁴.^2))  

        set_optimizer(Training_z₂₃²⁴, Gurobi.Optimizer)
        optimize!(Training_z₂₃²⁴)
    


#-----------------------------------Define Offline-Training Model.  z₂₄²³
        Training_z₂₄²³= Model()
        #  @variable(Training_z₂₄²³, K_z₂₄²³[1:3])  # shape control
        @variable(Training_z₂₄²³, sh_2[1])
        @variable(Training_z₂₄²³, K_24_gc[1:a])  # buses of SGs are 2,3,4,5,27,30.
        @variable(Training_z₂₄²³, K_24_gv[1:1])   # buses of VSGs are 1,23,24.
        @variable(Training_z₂₄²³, K_24_m[1:a+b])   # pairs of (V)SGs
        @variable(Training_z₂₄²³, z_24_23[1:size(zᵟ_c,1)])   
        @variable(Training_z₂₄²³, penalty_z₂₄²³[1:size(zᵟ_c,1)])

        for i in 1:size(zᵟ_c,1)
          #    @constraint(Training_z₂₄²³, z_24_23[i]==K_z₂₄²³[1]*(sum(K_24_gc .*matrix_ω[i,1:a]) +K_z₂₄²³[2]*sum(K_24_gv *matrix_ω[i,a+1]) +K_z₂₄²³[3]*sum(K_24_m .*matrix_ω[i,a+2:end])))
           @constraint(Training_z₂₄²³, z_24_23[i]==sh_2[1]-(sum(K_24_gc .*matrix_ω[i,1:a]) +sum(K_24_gv *matrix_ω[i,a+1]) +sum(K_24_m .*matrix_ω[i,a+2:end])))
          @constraint(Training_z₂₄²³, penalty_z₂₄²³[i]==z_24_23[i] -zᵟ_c[i,2])
        end

        @objective(Training_z₂₄²³, Min, sum(penalty_z₂₄²³.^2))

        set_optimizer(Training_z₂₄²³, Gurobi.Optimizer)
        optimize!(Training_z₂₄²³)


  =#

#-----------------------------------Define Offline-Training Model.  z₂₃¹
        Training_z₂₃¹= Model()

        @variable(Training_z₂₃¹, K_23_1_gc[1:a])  # buses of SGs are 2,3,4,5,27,30.
        @variable(Training_z₂₃¹, K_23_1_gv[1:1])   # buses of VSGs are 1,23,24.
        @variable(Training_z₂₃¹, K_23_1_m[1:a+b])   # pairs of (V)SGs
        @variable(Training_z₂₃¹, z_23_1[1:size(zᵟ_c,1)])   
        @variable(Training_z₂₃¹, penalty_z₂₃¹[1:size(zᵟ_c,1)])

        for i in 1:size(zᵟ_c,1)
            @constraint(Training_z₂₃¹, z_23_1[i]==(sum(K_23_1_gc .*matrix_ω[i,1:a]) +sum(K_23_1_gv *matrix_ω[i,a+1]) +sum(K_23_1_m .*matrix_ω[i,a+2:end])))
            @constraint(Training_z₂₃¹, penalty_z₂₃¹[i]==z_23_1[i] -zᵟ_c[i,1])
        end

        @objective(Training_z₂₃¹, Min, sum(penalty_z₂₃¹.^2))

        set_optimizer(Training_z₂₃¹, Gurobi.Optimizer)
        optimize!(Training_z₂₃¹)



#-----------------------------------Define Offline-Training Model.  z₂₄¹
        Training_z₂₄¹= Model()

        @variable(Training_z₂₄¹, K_24_1_gc[1:a])  # buses of SGs are 2,3,4,5,27,30.
        @variable(Training_z₂₄¹, K_24_1_gv[1:1])   # buses of VSGs are 1,23,26.
        @variable(Training_z₂₄¹, K_24_1_m[1:a+b])   # pairs of SGs
        @variable(Training_z₂₄¹, z_24_1[1:size(zᵟ_c,1)])   
        @variable(Training_z₂₄¹, penalty_z₂₄¹[1:size(zᵟ_c,1)])

        for i in 1:size(zᵟ_c,1)
            @constraint(Training_z₂₄¹, z_24_1[i]==(sum(K_24_1_gc .*matrix_ω[i,1:a]) +sum(K_24_1_gv *matrix_ω[i,a+1]) +sum(K_24_1_m .*matrix_ω[i,a+2:end])))
            @constraint(Training_z₂₄¹, penalty_z₂₄¹[i]==z_24_1[i] -zᵟ_c[i,2])
        end

        @objective(Training_z₂₄¹, Min, sum(penalty_z₂₄¹.^2))

        set_optimizer(Training_z₂₄¹, Gurobi.Optimizer)
        optimize!(Training_z₂₄¹)


K_23_1_gc = value.(K_23_1_gc)
K_24_1_gc = value.(K_24_1_gc)
K_23_1_gv = value.(K_23_1_gv)
K_24_1_gv = value.(K_24_1_gv)
K_23_1_m = value.(K_23_1_m)
K_24_1_m = value.(K_24_1_m)
#sh_1 = value.(sh_1[1])
#sh_2 = value.(sh_2[1])
#shape_control = zeros(2,3)
#shape_control[1,:] = value.(K_z₂₃²⁴)
#shape_control[2,:] = value.(K_z₂₄²³)
K_c_gc[1,:]=K_23_1_gc
K_c_gc[2,:]=K_24_1_gc

K_c_gv[1,:]=K_23_1_gv
K_c_gv[2,:]=K_24_1_gv

K_c_m[1,:]=K_23_1_m
K_c_m[2,:]=K_24_1_m

z_23_1 = value.(z_23_1)
z_24_1 = value.(z_24_1)
# MAPE calculation
MAPE_z_23_1 = sum(abs.(z_23_1 - zᵟ_c[:,1]) ./ abs.(zᵟ_c[:,1])) / size(zᵟ_c,1)
MAPE_z_24_1 = sum(abs.(z_24_1 - zᵟ_c[:,2]) ./ abs.(zᵟ_c[:,2])) / size(zᵟ_c,1)

    return  K_c_gc, K_c_gv, K_c_m, MAPE_z_23_1, MAPE_z_24_1 #shape_control , sh_1, sh_2
end
