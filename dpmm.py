# -*- coding: utf-8 -*-


"""
Dirichlet process mixture model (for N observations y_1, ..., y_N)
    1) generate a distribution G ~ DP(G_0, α)
    2) generate parameters θ_1, ..., θ_N ~ G
    [1+2) <=> (with B_1, ..., B_N a measurable partition of the set for which 
        G_0 is a finite measure, G(B_i) = Θ_i:)
       generate G(B_1), ..., G(B_N) ~ Dirichlet(αG_0(B_1), ..., αG_0(B_N)]
    3) generate each datapoint y_i ~ F(θ_i)
Now, an alternative is:
    1) generate a vector β ~ Stick(1, α) (<=> GEM(1, α))
    2) generate cluster assignments c_i ~ Categorical(β) (gives K clusters)
    3) generate parameters Φ_1, ...,Φ_K ~ G_0
    4) generate each datapoint y_i ~ F(Φ_{c_i})
    for instance F is a Gaussian and Φ_c = (mean_c, var_c)
Another one is:
    1) generate cluster assignments c_1, ..., c_N ~ CRP(N, α) (K clusters)
    2) generate parameters Φ_1, ...,Φ_K ~ G_0
    3) generate each datapoint y_i ~ F(Φ_{c_i})
"""

#def normal_distrib(

#def dirichlet_process_mixture_model(base_distrib, alpha):

