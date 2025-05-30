function offline_trainning(I_SCC_all_buses_scenarios, matrix_ω, Iₗᵢₘ, v)

#-----------------------------------Dataset Classification
    K_g=zeros(12,30)   # define the linearized coffecoents for SGs
    K_c=zeros(3,30)    # define the linearized coffecoents for IBGs
    K_m=zeros(66,30)  # define the linearized coffecoents for pairs of SGs

    N_Ω1_tp1=0  # define the number of type-I & II errors 
    N_Ω2_tp1=0
    N_Ω3_tp1=0
    N_Ω1_tp2=0
    N_Ω2_tp2=0
    N_Ω3_tp2=0

    err_Ω1_tp1=0  # define the error of type-I & II
    err_Ω2_tp1=0
    err_Ω3_tp1=0
    err_Ω1_tp2=0
    err_Ω2_tp2=0
    err_Ω3_tp2=0
    

    for k in 1:size(I_SCC_all_buses_scenarios,2)
    
        I_scc_Ω1 = []     # initialize the classification matrix
        I_scc_Ω2 = []
        I_scc_Ω3 = []
        matrix_ω_Ω1 = []
        matrix_ω_Ω2 = []
        matrix_ω_Ω3 = []
    
        for i in 1:size(I_SCC_all_buses_scenarios,1)  # traverse I_scc and matrix_ω for classification
            if I_SCC_all_buses_scenarios[i,k] < Iₗᵢₘ
                push!(I_scc_Ω1, I_SCC_all_buses_scenarios[i,k])
                push!(matrix_ω_Ω1, matrix_ω[i,:])
            elseif Iₗᵢₘ <= I_SCC_all_buses_scenarios[i,k] < Iₗᵢₘ + v
                push!(I_scc_Ω2, I_SCC_all_buses_scenarios[i,k])
                push!(matrix_ω_Ω2, matrix_ω[i,:])
            elseif I_SCC_all_buses_scenarios[i,k] >= Iₗᵢₘ + v
                push!(I_scc_Ω3, I_SCC_all_buses_scenarios[i,k])
                push!(matrix_ω_Ω3, matrix_ω[i,:])
            end
        end
    
        I_scc_Ω1 = hcat(I_scc_Ω1...)'               # convert the classification results into a matrix
        I_scc_Ω2 = hcat(I_scc_Ω2...)'
        I_scc_Ω3 = hcat(I_scc_Ω3...)'
        matrix_ω_Ω1 = hcat(matrix_ω_Ω1...)'
        matrix_ω_Ω2 = hcat(matrix_ω_Ω2...)'
        matrix_ω_Ω3 = hcat(matrix_ω_Ω3...)'
    
    
#-----------------------------------Define Offline-Training Model
        model_ot= Model()
    
        @variable(model_ot, k_fg[1:12])  # buses of SGs are 2,3,4,5,27,30.
        @variable(model_ot, k_fc[1:3])  # buses of IBG are 1,23,26. 
        @variable(model_ot, k_fm[1:66]) # pairs of SGs
    
        @variable(model_ot, I_FL_ω_Ω1[1:size(matrix_ω_Ω1,1)])  
        @variable(model_ot, I_FL_ω_Ω2[1:size(matrix_ω_Ω2,1)])  
        @variable(model_ot, I_FL_ω_Ω3[1:size(matrix_ω_Ω3,1)])  
    
        for i in 1:size(matrix_ω_Ω1,1)
            @constraint(model_ot, I_FL_ω_Ω1[i]==sum(k_fg.* matrix_ω_Ω1[i,1:12])+ sum(k_fc.* matrix_ω_Ω1[i,13:15])+ sum(k_fm.* matrix_ω_Ω1[i,16:81]))
            @constraint(model_ot, I_FL_ω_Ω1[i]<=Iₗᵢₘ ) 
        end
    
        for i in 1:size(matrix_ω_Ω3,1)
            @constraint(model_ot, I_FL_ω_Ω3[i]==sum(k_fg.* matrix_ω_Ω3[i,1:12])+ sum(k_fc.* matrix_ω_Ω3[i,13:15])+ sum(k_fm.* matrix_ω_Ω3[i,16:81]))
            @constraint(model_ot, I_FL_ω_Ω3[i]>=Iₗᵢₘ ) 
        end
    
        @variable(model_ot, penalty_one_time_item[1:size(matrix_ω_Ω2,1)])
        for i in 1:size(matrix_ω_Ω2,1)
            @constraint(model_ot, I_FL_ω_Ω2[i]==sum(k_fg.* matrix_ω_Ω2[i,1:12])+ sum(k_fc.* matrix_ω_Ω2[i,13:15])+ sum(k_fm.* matrix_ω_Ω2[i,16:81]))
            @constraint(model_ot, penalty_one_time_item[i]==I_scc_Ω2[i]- I_FL_ω_Ω2[i])
        end
    
    
        @objective(model_ot, Min, sum(penalty_one_time_item.^2))  


#-----------------------------------Solve and Output Results
        set_optimizer(model_ot , Gurobi.Optimizer)
        optimize!(model_ot)
    

        k_fg=JuMP.value.(k_fg)
        k_fc=JuMP.value.(k_fc)
        k_fm=JuMP.value.(k_fm)
 
        K_g[:,k]=k_fg'
        K_c[:,k]=k_fc'
        K_m[:,k]=k_fm'
    
        I_FL_ω_Ω1=JuMP.value.(I_FL_ω_Ω1)
        I_FL_ω_Ω2=JuMP.value.(I_FL_ω_Ω2)
        I_FL_ω_Ω3=JuMP.value.(I_FL_ω_Ω3)


#-----------------------------------Calculate errors for type-1
        for  i in 1:size(I_FL_ω_Ω1,2) # error calculation
            if I_FL_ω_Ω1[i] >= Iₗᵢₘ &&  I_scc_Ω1[i] < Iₗᵢₘ
                N_Ω1_tp1=N_Ω1_tp1+1
                err_Ω1_tp1= err_Ω1_tp1+(I_FL_ω_Ω1[i] - I_scc_Ω1[i])/I_scc_Ω1[i]
            end
        end
    
        for  i in 1:size(I_FL_ω_Ω2,2) # error calculation
            if I_FL_ω_Ω2[i] >= Iₗᵢₘ &&  I_scc_Ω2[i] < Iₗᵢₘ
                N_Ω2_tp1=N_Ω2_tp1+1
                err_Ω2_tp1= err_Ω2_tp1+(I_FL_ω_Ω2[i] - I_scc_Ω2[i])/I_scc_Ω2[i]
            end
        end
    
        for  i in 1:size(I_FL_ω_Ω3,2) # error calculation
            if I_FL_ω_Ω3[i] >= Iₗᵢₘ &&  I_scc_Ω3[i] < Iₗᵢₘ
                N_Ω3_tp1=N_Ω3_tp1+1
                err_Ω3_tp1= err_Ω3_tp1+(I_FL_ω_Ω3[i] - I_scc_Ω3[i])/I_scc_Ω3[i]
            end
        end
    

#-----------------------------------Calculate errors for type-2
    
        for  i in 1:size(I_FL_ω_Ω1,2) # error calculation
            if I_FL_ω_Ω1[i] < Iₗᵢₘ &&  I_scc_Ω1[i] >= Iₗᵢₘ
                N_Ω1_tp2=N_Ω1_tp2+1
                err_Ω1_tp2= err_Ω1_tp2+(I_FL_ω_Ω1[i] - I_scc_Ω1[i])/I_scc_Ω1[i]
            end
        end

        for  i in 1:size(I_FL_ω_Ω2,2) # error calculation
            if I_FL_ω_Ω2[i] < Iₗᵢₘ &&  I_scc_Ω2[i] >= Iₗᵢₘ
                N_Ω2_tp2=N_Ω2_tp2+1
                err_Ω2_tp2= err_Ω2_tp2+(I_FL_ω_Ω2[i] - I_scc_Ω2[i])/I_scc_Ω2[i]
            end
        end

        for  i in 1:size(I_FL_ω_Ω3,2) # error calculation
            if I_FL_ω_Ω3[i] < Iₗᵢₘ &&  I_scc_Ω3[i] >= Iₗᵢₘ
                N_Ω3_tp2=N_Ω3_tp2+1
                err_Ω3_tp2= err_Ω3_tp2+(I_FL_ω_Ω3[i] - I_scc_Ω3[i])/I_scc_Ω3[i]
            end
        end

    end


    N_type_1=N_Ω1_tp1+N_Ω2_tp1+N_Ω3_tp1    
    err_type_1=(err_Ω1_tp1+err_Ω2_tp1+err_Ω3_tp1)/N_type_1

    N_type_2=N_Ω1_tp2+N_Ω2_tp2+N_Ω3_tp2    
    err_type_2=(err_Ω1_tp2+err_Ω2_tp2+err_Ω3_tp2)/N_type_2

    return  K_g, K_c, K_m, N_type_1, N_type_2, err_type_1, err_type_2
end
