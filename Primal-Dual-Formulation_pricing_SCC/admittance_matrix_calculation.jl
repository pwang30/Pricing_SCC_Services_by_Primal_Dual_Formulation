function admittance_matrix_calculation(numnodes)
    df = DataFrame(CSV.File( "/Users/kl/Desktop/SCL_pricing_PD/Linespara.csv" )) 
    linepara=df[:,:]
    
    Yₗᵢₙₑ= zeros(numnodes, numnodes)         # define admittance matrix of the transmission lines
    branch_num= size(linepara, 1)           # number of branches
    for k in 1:branch_num                   #  calculate the ADMITTANCE MATRIX of the transmission lines
        i = linepara[k, 1]                  # bus from
        j = linepara[k, 2]                  # bus to
        Yₗᵢₙₑ[i, j] = -1/linepara[k, 4]*3    # off-diagonal elements
        Yₗᵢₙₑ[j, i] = Yₗᵢₙₑ[i, j]             # symmetry
    end
    for k in 1:numnodes
        Yₗᵢₙₑ[k, k] = -sum(Yₗᵢₙₑ[k, :])       # diagonal elements 
    end  

    return Yₗᵢₙₑ

end