# How to price SCC via primal-dual formulation

**PLEASE NOTE, the main models and methodologies are in the listed papers here. Fully understanding these works is the foundation of our work.**
- Short-Circuit Current (SCC) models refer to:
  1. Chu, Zhongda, and Fei Teng. ["Short circuit current constrained UC in high IBG-penetrated power systems." IEEE Transactions on Power Systems 36.4 (2021): 3776-3785.](https://ieeexplore.ieee.org/abstract/document/9329077)
  2. Chu, Zhongda, Jingyi Wu, and Fei Teng. ["Pricing of short circuit current in high IBR-penetrated system." Electric Power Systems Research 235 (2024): 110690.](https://www.sciencedirect.com/science/article/pii/S0378779624005765)
- Primal-Dual formulation for addressing UC issues refer to:
  1. Ye, Yujian, et al. ["Incorporating non-convex operating characteristics into bi-level optimization electricity market models." IEEE Transactions on Power Systems 35.1 (2019): 163-176.](https://ieeexplore.ieee.org/abstract/document/8746573)
- Data used in this work and relevant work/data please refer to our previous work:
  1. Wang, Peng, and Luis Badesa. ["Imperfect Competition in Markets for Short-Circuit Current Services." arXiv preprint arXiv:2508.09425 (2025)](https://arxiv.org/pdf/2508.09425).


**GUIDANCE abot how to use the code of our work**

The work is mainly made of two parts:
1. Modelling SCC constraints.
2. Modelling of primal-dual formulation.

We try to guide you to understand our logistics of coding, once you fully understand, then analyze any power systems you want.
- For the code of SCC modelling, please refer to the files named "_admittance_matrix_calculation.jl_", "_dataset_gene.jl_" and "_offline_trainning.jl_".

  1. "_admittance_matrix_calculation.jl_" calculates the impedance of transmission lines of the system, easy to follow.

  2. "_dataset_gene.jl_" generates the data for classification, i.e., the offline trainning process. The subfunction "_admittance_matrix_calculation.jl_" is called here to obtain the transmission line admittance matrix which is combined with the generators' admittance matrix. In "_dataset_gene.jl_", code from line 79-86 is the equation of actual, exact SCC representation. The remainder of code is generating all possible UC status pairs of generators. The matrix "I_SCC_all_buses_scenarios" are the SCC corresponding to "matrix_ω" (storing UC status and capacity factor of IBR, comprising all possible scenarios).

  3. "_offline_trainning.jl_" is the trainning process, with inputting parameters from above subfunctions.

- For the code of primal-dual modelling, please refer to the file named "_Primal-Dual-Formulation_pricing_SCC.jl_".
- For the code of dispatchable pricing, please refer to the file named "_dispatchable_pricing.jl_".
- The file named "_SCC_contribution.jl_" is to compute the SCC revenue by units' weighted contributions, which serve for the P-D method and dispatchable method.
- For the code of restricted pricing, please refer to the files named "_restricted_pricing_optimal.jl_" and "_restricted_pricing_preset.jl_". The former one is the first stage of this method, created to find the optimal UC decisions, while the latter one is for the second stage, i.e., pricing.
----

If you find something helpful or use this code for your own work, please cite this paper:
<ol>
      Wang, Peng and Luis Badesa. "Pricing Short-Circuit Current via a Primal-Dual Formulation for Preserving Integrality Constraints." arXiv preprint arXiv:2510.05293 (2025).
</ol>
      <br>
      
<ol> 
@article{wang2025pricing, <br>
  title={Pricing Short-Circuit Current via a Primal-Dual Formulation for Preserving Integrality Constraints}, <br>
  author={Wang, Peng and Badesa, Luis}, <br>
  journal={arXiv preprint arXiv:2510.05293}, <br>
  year={2025} <br>
}
  
----

This work was supported by MICIU/AEI/10.13039/501100011033 and ERDF/EU under grant PID2023-150401OA-C22, as well as by the Madrid Government (Comunidad de Madrid-Spain) under the Multiannual Agreement 2023-2026 with Universidad Politécnica de Madrid, ``Line A - Emerging PIs''. The work of Peng Wang was also supported by China Scholarship Council under grant 202408500065.


<figure style="display:inline-block; margin:10px; text-align:center;">
   <img src="./logos/MICIU+Cofinanciado+AEI.jpg" style="width:600px; height:140px; object-fit:contain; display:block;">
</figure>
<figure style="display:inline-block; margin:10px; text-align:center;">
   <img src="./logos/Logo_CM.png" style="width:200px; height:160px; object-fit:contain; display:block;">
</figure>
