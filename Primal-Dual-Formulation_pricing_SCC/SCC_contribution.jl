function SCC_contribution(UC_SG)  

#-----------------------------------Calculation of SCC constraints----------------------------------
    numnodes=30                         # number of nodes
    num_IBG=3                           # number of nodes where IBGs are located
    #            Bus Number	     x1	      x2
        SGpara=[
                   2	       0.0846	0.0812	
                   3	       0.0799	0.0774	
                   4	       0.0758	0.0735	
                   5	       0.0731	0.0713	
                   27	       0.0519	0.0457	
                   30	       0.0523	0.0469 ]  # buses where SGs are located

    Y_SGs = zeros(size(SGpara,1), size(SGpara,2)-1)    # define ADMITTANCE MATRIX of the SGs, buses:2,3,4,5,27,30
    I_SGs = zeros(size(SGpara,1), size(SGpara,2)-1)    # define I_SGs of the SGs, buses:2,3,4,5,27,30
    
    Yₗᵢₙₑ=admittance_matrix_calculation(numnodes)   # calculate the ADMITTANCE MATRIX of the network

    for k in 1:size(SGpara,1)                      # calculate the ADMITTANCE MATRIX of the SGs
        for j in 2:size(SGpara,2)
            Y_SGs[k, j-1] = 1/SGpara[k, j]/10        
        end           
    end 
    
    E_sg=0.95
    I_SGs=Y_SGs.*E_sg        # calculate the I_SGs of the SGs, buses:2,3,4,5,27,30
    

#------------
    
     Weighted_contribution_to_11 = zeros(12, 24)
     Weighted_contribution_to_26 = zeros(12, 24)
     Weighted_contribution_to_29 = zeros(12, 24)
     Weighted_contribution_to_30 = zeros(12, 24)

for t in 1:24

    Y_SGs_with_status = zeros(numnodes, numnodes)     # define the ADMITTANCE MATRIX for SGs status
    Y_total = zeros(numnodes, numnodes)        # define the total ADMITTANCE MATRIX
    Z = zeros(numnodes, numnodes)              # define the impedance matrix
    
    Y_SGs_with_status[2,2]= Y_SGs[1,1]*UC_SG[t,1] + Y_SGs[1,2]*UC_SG[t,2]
    Y_SGs_with_status[3,3]= Y_SGs[2,1]*UC_SG[t,3] + Y_SGs[2,2]*UC_SG[t,4]
    Y_SGs_with_status[4,4]= Y_SGs[3,1]*UC_SG[t,5] + Y_SGs[3,2]*UC_SG[t,6]
    Y_SGs_with_status[5,5]= Y_SGs[4,1]*UC_SG[t,7] + Y_SGs[4,2]*UC_SG[t,8]
    Y_SGs_with_status[27,27]= Y_SGs[5,1]*UC_SG[t,9] + Y_SGs[5,2]*UC_SG[t,10]
    Y_SGs_with_status[30,30]= Y_SGs[6,1]*UC_SG[t,11] + Y_SGs[6,2]*UC_SG[t,12]

    Y_total .= Yₗᵢₙₑ + Y_SGs_with_status      # calculate the total ADMITTANCE MATRIX
    Z .= inv(Y_total)                        # calculate the IMPEDANCE MATRIX
    

    Weighted_contribution_to_11[1,t] = Z[2,11]/Z[11,11] *I_SGs[1,1] *UC_SG[t,1]
    Weighted_contribution_to_11[2,t] = Z[2,11]/Z[11,11] *I_SGs[1,2] *UC_SG[t,2]
    Weighted_contribution_to_11[3,t] = Z[3,11]/Z[11,11] *I_SGs[2,1] *UC_SG[t,3]
    Weighted_contribution_to_11[4,t] = Z[3,11]/Z[11,11] *I_SGs[2,2] *UC_SG[t,4]
    Weighted_contribution_to_11[5,t] = Z[4,11]/Z[11,11] *I_SGs[3,1] *UC_SG[t,5]
    Weighted_contribution_to_11[6,t] = Z[4,11]/Z[11,11] *I_SGs[3,2] *UC_SG[t,6]
    Weighted_contribution_to_11[7,t] = Z[5,11]/Z[11,11] *I_SGs[4,1] *UC_SG[t,7]
    Weighted_contribution_to_11[8,t] = Z[5,11]/Z[11,11] *I_SGs[4,2] *UC_SG[t,8]
    Weighted_contribution_to_11[9,t] = Z[27,11]/Z[11,11] *I_SGs[5,1] *UC_SG[t,9]
    Weighted_contribution_to_11[10,t] = Z[27,11]/Z[11,11] *I_SGs[5,2] *UC_SG[t,10]
    Weighted_contribution_to_11[11,t] = Z[30,11]/Z[11,11] *I_SGs[6,1] *UC_SG[t,11]
    Weighted_contribution_to_11[12,t] = Z[30,11]/Z[11,11] *I_SGs[6,2] *UC_SG[t,12]


    Weighted_contribution_to_26[1,t] = Z[2,26]/Z[26,26] *I_SGs[1,1] *UC_SG[t,1]
    Weighted_contribution_to_26[2,t] = Z[2,26]/Z[26,26] *I_SGs[1,2] *UC_SG[t,2]
    Weighted_contribution_to_26[3,t] = Z[3,26]/Z[26,26] *I_SGs[2,1] *UC_SG[t,3]
    Weighted_contribution_to_26[4,t] = Z[3,26]/Z[26,26] *I_SGs[2,2] *UC_SG[t,4]
    Weighted_contribution_to_26[5,t] = Z[4,26]/Z[26,26] *I_SGs[3,1] *UC_SG[t,5]
    Weighted_contribution_to_26[6,t] = Z[4,26]/Z[26,26] *I_SGs[3,2] *UC_SG[t,6]
    Weighted_contribution_to_26[7,t] = Z[5,26]/Z[26,26] *I_SGs[4,1] *UC_SG[t,7]
    Weighted_contribution_to_26[8,t] = Z[5,26]/Z[26,26] *I_SGs[4,2] *UC_SG[t,8]
    Weighted_contribution_to_26[9,t] = Z[27,26]/Z[26,26] *I_SGs[5,1] *UC_SG[t,9]
    Weighted_contribution_to_26[10,t] = Z[27,26]/Z[26,26] *I_SGs[5,2] *UC_SG[t,10]
    Weighted_contribution_to_26[11,t] = Z[30,26]/Z[26,26] *I_SGs[6,1] *UC_SG[t,11]
    Weighted_contribution_to_26[12,t] = Z[30,26]/Z[26,26] *I_SGs[6,2] *UC_SG[t,12]

    Weighted_contribution_to_29[1,t] = Z[2,29]/Z[29,29] *I_SGs[1,1] *UC_SG[t,1]
    Weighted_contribution_to_29[2,t] = Z[2,29]/Z[29,29] *I_SGs[1,2] *UC_SG[t,2]
    Weighted_contribution_to_29[3,t] = Z[3,29]/Z[29,29] *I_SGs[2,1] *UC_SG[t,3]
    Weighted_contribution_to_29[4,t] = Z[3,29]/Z[29,29] *I_SGs[2,2] *UC_SG[t,4]
    Weighted_contribution_to_29[5,t] = Z[4,29]/Z[29,29]  *I_SGs[3,1] *UC_SG[t,5]
    Weighted_contribution_to_29[6,t] = Z[4,29]/Z[29,29] *I_SGs[3,2] *UC_SG[t,6]
    Weighted_contribution_to_29[7,t] = Z[5,29]/Z[29,29] *I_SGs[4,1] *UC_SG[t,7]
    Weighted_contribution_to_29[8,t] = Z[5,29]/Z[29,29] *I_SGs[4,2] *UC_SG[t,8]
    Weighted_contribution_to_29[9,t] = Z[27,29]/Z[29,29] *I_SGs[5,1] *UC_SG[t,9]
    Weighted_contribution_to_29[10,t] = Z[27,29]/Z[29,29]    *I_SGs[5,2] *UC_SG[t,10]
    Weighted_contribution_to_29[11,t] = Z[30,29]/Z[29,29] *I_SGs[6,1] *UC_SG[t,11]
    Weighted_contribution_to_29[12,t] = Z[30,29]/Z[29,29]  *I_SGs[6,2] *UC_SG[t,12]

    Weighted_contribution_to_30[1,t] = Z[2,30]/Z[30,30] *I_SGs[1,1] *UC_SG[t,1]
    Weighted_contribution_to_30[2,t] = Z[2,30]/Z[30,30] *I_SGs[1,2] *UC_SG[t,2]
    Weighted_contribution_to_30[3,t] = Z[3,30]/Z[30,30] *I_SGs[2,1] *UC_SG[t,3]
    Weighted_contribution_to_30[4,t] = Z[3,30]/Z[30,30] *I_SGs[2,2] *UC_SG[t,4]
    Weighted_contribution_to_30[5,t] = Z[4,30]/Z[30,30] *I_SGs[3,1] *UC_SG[t,5]
    Weighted_contribution_to_30[6,t] = Z[4,30]/Z[30,30] *I_SGs[3,2] *UC_SG[t,6]
    Weighted_contribution_to_30[7,t] = Z[5,30]/Z[30,30] *I_SGs[4,1] *UC_SG[t,7]
    Weighted_contribution_to_30[8,t] = Z[5,30]/Z[30,30] *I_SGs[4,2] *UC_SG[t,8]
    Weighted_contribution_to_30[9,t] = Z[27,30]/Z[30,30] *I_SGs[5,1] *UC_SG[t,9]
    Weighted_contribution_to_30[10,t] = Z[27,30]/Z[30,30] *I_SGs[5,2] *UC_SG[t,10]
    Weighted_contribution_to_30[11,t] = Z[30,30]/Z[30,30] *I_SGs[6,1] *UC_SG[t,11]
    Weighted_contribution_to_30[12,t] = Z[30,30]/Z[30,30] *I_SGs[6,2] *UC_SG[t,12]

end
    return Weighted_contribution_to_11, Weighted_contribution_to_26, Weighted_contribution_to_29, Weighted_contribution_to_30
    
    
end
