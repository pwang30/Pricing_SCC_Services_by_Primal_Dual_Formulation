function cal_SCL_contribution(UC_SGs, λ_SCL)  

#-----------------------------------SGs reactance
    T=size(UC_SGs, 2)                     # number of time steps
    numnodes=30                           # number of nodes
    #            Bus Number	     x1	      x2
        SGpara=[
                   2	       0.0846	0.0812	
                   3	       0.0799	0.0774	
                   4	       0.0758	0.0735	
                   5	       0.0731	0.0713	
                   27	       0.0519	0.0457	
                   30	       0.0523	0.0469 ]  # buses where SGs are located
    


#------------Calculate IMPEDANCE MATRIX--------------------------
    λ_SCL_ᴳ²_1 =zeros(4,T)
    λ_SCL_ᴳ²_2 =zeros(4,T)
    λ_SCL_ᴳ³_1 =zeros(4,T)
    λ_SCL_ᴳ³_2 =zeros(4,T)
    λ_SCL_ᴳ⁴_1 =zeros(4,T)
    λ_SCL_ᴳ⁴_2 =zeros(4,T)
    λ_SCL_ᴳ⁵_1 =zeros(4,T)
    λ_SCL_ᴳ⁵_2 =zeros(4,T)
    λ_SCL_ᴳ²⁷_1 =zeros(4,T)
    λ_SCL_ᴳ²⁷_2 =zeros(4,T)    
    λ_SCL_ᴳ³⁰_1 =zeros(4,T)
    λ_SCL_ᴳ³⁰_2 =zeros(4,T)

    Yₗᵢₙₑ= admittance_matrix_calculation(numnodes)      # calculate the ADMITTANCE MATRIX of the network
    Y_SGs = zeros(size(SGpara,1), size(SGpara,2)-1)    # define ADMITTANCE MATRIX of the SGs, buses:2,3,4,5,27,30
        for k in 1:size(SGpara,1)                      # calculate the ADMITTANCE MATRIX of the SGs
            for j in 2:size(SGpara,2)
                Y_SGs[k, j-1] = 1/SGpara[k, j]/10        
            end           
        end 
    Y_SGs_with_status = zeros(numnodes, numnodes)     # define the ADMITTANCE MATRIX for SGs status
    Y_total = zeros(numnodes, numnodes)               # define the total ADMITTANCE MATRIX
    Z = zeros(numnodes, numnodes)                     # define the impedance matrix

    for t in 1:T
        status_SGs = UC_SGs[:,t]                    # status of SGs
        Y_SGs_with_status .= 0                      # reset Y_SGs_with_status matrix
        
        index = 1
            for i in 1:length(SGpara[:,2:end])
                Y_SGs_with_status[Int(SGpara[Int(ceil(i/2)), 1]), Int(SGpara[Int(ceil(i/2)), 1])] = Y_SGs_with_status[Int(SGpara[Int(ceil(i/2)), 1]), Int(SGpara[Int(ceil(i/2)), 1])] +Y_SGs[Int(ceil(i/2)),index] *status_SGs[i]  # status of SGs
                    index = index+1
                    if index > size(Y_SGs, 2)
                        index = 1
                    end
            end

        Y_total .= Yₗᵢₙₑ + Y_SGs_with_status      # calculate the total ADMITTANCE MATRIX
        Z .= inv(Y_total)                        # calculate the IMPEDANCE MATRIX

        λ_SCL_ᴳ²_1[1, t] =   (Z[11,2]/Z[11,11]) *λ_SCL[1, t] 
        λ_SCL_ᴳ²_1[2, t] =   (Z[26,2]/Z[26,26]) *λ_SCL[2, t] 
        λ_SCL_ᴳ²_1[3, t] =   (Z[29,2]/Z[29,29]) *λ_SCL[3, t]
        λ_SCL_ᴳ²_1[4, t] =   (Z[30,2]/Z[30,30]) *λ_SCL[4, t]
        λ_SCL_ᴳ²_2[1, t] =   (Z[11,2]/Z[11,11]) *λ_SCL[1, t] 
        λ_SCL_ᴳ²_2[2, t] =   (Z[26,2]/Z[26,26]) *λ_SCL[2, t] 
        λ_SCL_ᴳ²_2[3, t] =   (Z[29,2]/Z[29,29]) *λ_SCL[3, t]
        λ_SCL_ᴳ²_2[4, t] =   (Z[30,2]/Z[30,30]) *λ_SCL[4, t]
        λ_SCL_ᴳ³_1[1, t] =   (Z[11,3]/Z[11,11]) *λ_SCL[1, t]
        λ_SCL_ᴳ³_1[2, t] =   (Z[26,3]/Z[26,26]) *λ_SCL[2, t]
        λ_SCL_ᴳ³_1[3, t] =   (Z[29,3]/Z[29,29]) *λ_SCL[3, t]
        λ_SCL_ᴳ³_1[4, t] =   (Z[30,3]/Z[30,30]) *λ_SCL[4, t]
        λ_SCL_ᴳ³_2[1, t] =   (Z[11,3]/Z[11,11]) *λ_SCL[1, t]
        λ_SCL_ᴳ³_2[2, t] =   (Z[26,3]/Z[26,26]) *λ_SCL[2, t]
        λ_SCL_ᴳ³_2[3, t] =   (Z[29,3]/Z[29,29]) *λ_SCL[3, t]
        λ_SCL_ᴳ³_2[4, t] =   (Z[30,3]/Z[30,30]) *λ_SCL[4, t]
        λ_SCL_ᴳ⁴_1[1, t] =   (Z[11,4]/Z[11,11]) *λ_SCL[1, t]
        λ_SCL_ᴳ⁴_1[2, t] =   (Z[26,4]/Z[26,26]) *λ_SCL[2, t]
        λ_SCL_ᴳ⁴_1[3, t] =   (Z[29,4]/Z[29,29]) *λ_SCL[3, t]
        λ_SCL_ᴳ⁴_1[4, t] =   (Z[30,4]/Z[30,30]) *λ_SCL[4, t]
        λ_SCL_ᴳ⁴_2[1, t] =   (Z[11,4]/Z[11,11]) *λ_SCL[1, t]
        λ_SCL_ᴳ⁴_2[2, t] =   (Z[26,4]/Z[26,26]) *λ_SCL[2, t]
        λ_SCL_ᴳ⁴_2[3, t] =   (Z[29,4]/Z[29,29]) *λ_SCL[3, t]
        λ_SCL_ᴳ⁴_2[4, t] =   (Z[30,4]/Z[30,30]) *λ_SCL[4, t]
        λ_SCL_ᴳ⁵_1[1, t] =   (Z[11,5]/Z[11,11]) *λ_SCL[1, t]
        λ_SCL_ᴳ⁵_1[2, t] =   (Z[26,5]/Z[26,26]) *λ_SCL[2, t]
        λ_SCL_ᴳ⁵_1[3, t] =   (Z[29,5]/Z[29,29]) *λ_SCL[3, t]
        λ_SCL_ᴳ⁵_1[4, t] =   (Z[30,5]/Z[30,30]) *λ_SCL[4, t]
        λ_SCL_ᴳ⁵_2[1, t] =   (Z[11,5]/Z[11,11]) *λ_SCL[1, t]
        λ_SCL_ᴳ⁵_2[2, t] =   (Z[26,5]/Z[26,26]) *λ_SCL[2, t]
        λ_SCL_ᴳ⁵_2[3, t] =   (Z[29,5]/Z[29,29]) *λ_SCL[3, t]
        λ_SCL_ᴳ⁵_2[4, t] =   (Z[30,5]/Z[30,30]) *λ_SCL[4, t]
        λ_SCL_ᴳ²⁷_1[1, t] = (Z[11,27]/Z[11,11]) *λ_SCL[1, t]
        λ_SCL_ᴳ²⁷_1[2, t] = (Z[26,27]/Z[26,26]) *λ_SCL[2, t]
        λ_SCL_ᴳ²⁷_1[3, t] = (Z[29,27]/Z[29,29]) *λ_SCL[3, t]
        λ_SCL_ᴳ²⁷_1[4, t] = (Z[30,27]/Z[30,30]) *λ_SCL[4, t]
        λ_SCL_ᴳ²⁷_2[1, t] = (Z[11,27]/Z[11,11]) *λ_SCL[1, t]
        λ_SCL_ᴳ²⁷_2[2, t] = (Z[26,27]/Z[26,26]) *λ_SCL[2, t]
        λ_SCL_ᴳ²⁷_2[3, t] = (Z[29,27]/Z[29,29]) *λ_SCL[3, t]
        λ_SCL_ᴳ²⁷_2[4, t] = (Z[30,27]/Z[30,30]) *λ_SCL[4, t]
        λ_SCL_ᴳ³⁰_1[1, t] = (Z[11,30]/Z[11,11]) *λ_SCL[1, t]
        λ_SCL_ᴳ³⁰_1[2, t] = (Z[26,30]/Z[26,26]) *λ_SCL[2, t]
        λ_SCL_ᴳ³⁰_1[3, t] = (Z[29,30]/Z[29,29]) *λ_SCL[3, t]
        λ_SCL_ᴳ³⁰_1[4, t] = (Z[30,30]/Z[30,30]) *λ_SCL[4, t]
        λ_SCL_ᴳ³⁰_2[1, t] = (Z[11,30]/Z[11,11]) *λ_SCL[1, t]
        λ_SCL_ᴳ³⁰_2[2, t] = (Z[26,30]/Z[26,26]) *λ_SCL[2, t]
        λ_SCL_ᴳ³⁰_2[3, t] = (Z[29,30]/Z[29,29]) *λ_SCL[3, t]
        λ_SCL_ᴳ³⁰_2[4, t] = (Z[30,30]/Z[30,30]) *λ_SCL[4, t]      
    end

    
    return λ_SCL_ᴳ²_1, λ_SCL_ᴳ²_2, λ_SCL_ᴳ³_1, λ_SCL_ᴳ³_2, λ_SCL_ᴳ⁴_1, λ_SCL_ᴳ⁴_2, 
    λ_SCL_ᴳ⁵_1, λ_SCL_ᴳ⁵_2, λ_SCL_ᴳ²⁷_1, λ_SCL_ᴳ²⁷_2, λ_SCL_ᴳ³⁰_1, λ_SCL_ᴳ³⁰_2
    
    
end
