using LinearAlgebra, MKL
using QuantumInformation
using Convex, SCS
using Statistics
using Plots
using StatsPlots
using LaTeXStrings

#########################################################

function random_channel(y, r)
    g = GinibreEnsemble{2}(y,y)
    A = [rand(g) for _ = 1:r]
    Q = (sum(As' * As for As in A))^(-1/2)
    return [As * Q for As in A]
end

function standardization_of_E(E)
    y = size(E[1])[1]
    r = length(E)
    tr_list = [norm(E[i]) for i=1:r]
    tr_sort = sortperm(tr_list, rev=true)
    return E[tr_sort]
end

function def_r(y, x)
    return Integer(ceil(sqrt(y/(x-1)))) - 1
end

function get_F_from_E(EE, r)
    y = size(EE[1])[1]
    F = [Array{ComplexF64}(I(y))]
    r_bis = min(length(EE), r - 1)
    for i=1:r_bis
        append!(F, [EE[i]])
    end
    return F
end

function find_s0(F, eps = 1e-07)
    r = length(F)
    y = size(F[1])[1]

    s = Int(ceil(y / r)) - 1
    Gs = rand(GinibreEnsemble{2}(y, s))
    Pis = Gs * Gs'
    Xs = sum(F[i]' * Pis * F[i] for i=1:r)
    s += 1

    while s <= y
        vs = randn(y) + im * randn(y)
        temps = [F[i]' * vs for i=1:r]
        Gs = hcat(Gs, vs)
        Xs += sum(z * z' for z in temps)
        if rank(Xs, eps) == y
            break
        end
        s += 1
    end
    return (Gs, s)
end

function find_w0(F, Gs, eps =  1e-05)
    y, s = size(Gs)
    tempFdagGs = [Fs' * Gs for Fs in F]
    Ws = zeros(ComplexF64, y, s)
    for i = 1:s
        Z = sum(FFs * (I(s) - ket(i, s) * bra(i, s)) * FFs' for FFs in tempFdagGs)
        wi = (I(y) - Z * pinv(Z, eps)) * Gs[:, i]
        Ws[:, i] = wi
    end
    if sum(norm(Gs' * Fs * Ws - diagm(diag(Gs' * Fs * Ws)) ) for Fs in F ) < eps && rank(Gs' * Ws, eps) == s
        return Ws
    end
end

function define_M(F, Gs, Ws)
    r = length(F)
    y, s = size(Gs)
    M = zeros(ComplexF64, s, r)
    for i = 1:r
        Di = diag(Gs' * F[i] * Ws)
        M[:, i] = Di
    end
    return M
end

function create_group(M, x, eps = 1e-05)
    s = size(M)[1]
    MM = M'
    A = Set(1:s)
    group = []

    while length(group) < x
        temp = [pop!(A)]
        temp_rank = 1
        temp_matrix = MM[:, temp[1]]

        for idx in A
            if rank(hcat(temp_matrix, MM[:, idx]), eps) > temp_rank
                append!(temp, [idx])
                delete!(A, idx)
                temp_rank += 1
                temp_matrix = hcat(temp_matrix, MM[:, idx])
            end
        end
        
        append!(group, [temp])
    end

   return group
end

function define_S(Ws, group, eps = 1e-05)
    x = length(group)
    y = size(Ws)[1]
    S = zeros(ComplexF64, y, x)
    for i = 1:x
        S[:, i] = sum(Ws[:, j] for j in group[i])
    end
    if rank(S, eps) == x
        return S * (S' * S)^(-1/2)
    end
end

function define_deterministic_R_sdp(E, S, eps = 1e-05)
    y, x = size(S)
    
    Pi_ortogonal = I(y) - S*S'
    F = [Pi_ortogonal * Es * S for Es in E]
    JF = sum(res(Fs) * res(Fs)' for Fs in F)  
  
    Q = ComplexVariable(y*x, y*x)
    constraints = [Q in :SDP]
    constraints += [partialtrace(Q, 2, [y, x]) == I(y)]
    t = real(sum(Q[i]*conj(JF[i]) for i=1:(x^2*y^2)))

    problem = maximize(t, constraints)
    solve!(problem, Convex.MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => eps, "eps_rel" => eps); silent_solver = true)
    if string(problem.status) == "OPTIMAL"
        temp = Q.value
        
        temp = sqrt(temp*temp')
        partial_temp = (partialtrace(temp, 2, [y, x]))^(-1/2)
        temp = (partial_temp ⊗ I(x)) * temp * (partial_temp ⊗ I(x))           

        eigen_temp = eigen(temp)
        R = [Array(S')]
        for i=1:(x*y)
            append!(R, [sqrt(eigen_temp.values[i]) * unres(eigen_temp.vectors[:, i], x)' * Pi_ortogonal])
        end
        
        if norm(I(y) - sum(Rs'* Rs for Rs in R)) < eps
            return R
        end
    end    
end

function define_deterministic_R(E, S, eps = 1e-05)
    y, x = size(S)
    
    Pi_ortogonal = I(y) - S*S'
    F = [Pi_ortogonal * Es * S for Es in E]
    F1_inv_sqrt = sqrt( pinv( sum(Fs*Fs' for Fs in F), eps) )
    
    R = [Array(S')]
    
    for Fs in F
        append!(R, [Array(Fs' * F1_inv_sqrt)])
    end
    
    return R
end

function define_probabilistic_R(E, S, p = 3/4, q = 1e-03, eps = 1e-07)
    y, x = size(S)
    F = [sqrt(1-p) * S]
    for Es in E
        append!(F, [sqrt(p) * Es * S])
    end
    JF = sum(res(Fs)*res(Fs)' for Fs in F)
    Pis = res(S) * res(S)' / x + (I(y) - S*S') ⊗ I(x)
    F1_inv_sqrt = sqrt(pinv(Pis * (partialtrace(JF, 2, [y, x]) ⊗ I(x)) * Pis', eps))

    Choi = Hermitian(F1_inv_sqrt * JF * F1_inv_sqrt)
    Choi_eigen = eigen(Choi)

    v0 = Choi_eigen.vectors[:, x*y]
    v0 = Pis * F1_inv_sqrt * v0
    if abs(res(S)' * v0) < q
        v0 = (1-q) * v0 + q * res(S)
    end
    R = unres(v0, x)'
    R = R / maximum(svdvals(R))
    return [R]    
end

function _info_RES(R, E, S)
    RS = [Rs * S for Rs in R]
    RES = [Rs * Es * S for Rs in R for Es in E]
    
    fuJRS = sum(abs2(tr(RSs)) for RSs in RS)
    trJRS = real(sum(tr(RSs * RSs') for RSs in RS))
    
    fuJRES = sum(abs2(tr(RESs)) for RESs in RES)
    trJRES = real(sum(tr(RESs * RESs') for RESs in RES))
    
    return (fuJRS, trJRS, fuJRES, trJRES)
end

function result_REpS(T_info_RES, x, p)
    fuJRS, trJRS, fuJRES, trJRES = T_info_RES
    return (( (1-p) * trJRS + p * trJRES ) / x, ((1-p) * fuJRS + p * fuJRES) / (x * (1-p) * trJRS + x * p * trJRES)  )
end

function reference_fuRES(E, x)
    y = size(E[1])[1]
    S = zeros(ComplexF64, y, x)
    for i=1:x
        S[i,i] = 1
    end
    P = I(y) - S*S'

    Phi1 = [S' * Es * S for Es in E]
    Phi2 = [P * Es * S for Es in E]

    return sum(abs2(tr(Phi1s)) for Phi1s in Phi1) + (real(sum(tr(Phi2s' * Phi2s) for Phi2s in Phi2)) / x)
end

function generate_data(y, x, rankE, k, q1, q2, q3, sdp = 1, n_points = 20)
    list_x_p = Array(0:(1/n_points):1)
    
    list_y_pavg3 = zeros(Float64, length(list_x_p), k)
    list_y_pavg6 = zeros(Float64, length(list_x_p), k)
    list_y_pavg9 = zeros(Float64, length(list_x_p), k)

    list_y_c_favg3 = zeros(Float64, length(list_x_p), k)
    list_y_c_favg6 = zeros(Float64, length(list_x_p), k)
    list_y_c_favg9 = zeros(Float64, length(list_x_p), k)
    
    list_y_favg = zeros(Float64, length(list_x_p), k)
    if sdp == 1
        list_y_favg_sdp = zeros(Float64, length(list_x_p), k)
    end

    list_y_ref = zeros(Float64, length(list_x_p), k)

    for l=1:k
        E = random_channel(y, rankE)
        EE = standardization_of_E(E)
        r = def_r(y, x)
        F = get_F_from_E(EE, r)

        zal = 0
        prop_eps = 1
        S = zeros(ComplexF64, y, x)
        while zal == 0
            Gs = find_s0(F, 1e-07*prop_eps)[1]
            Ws = find_w0(F, Gs)
            if typeof(Ws) != Nothing
                zal = 1
                M = define_M(F, Gs, Ws)
                group = create_group(M, x)
                S = define_S(Ws, group)
            else
                prop_eps *= 2
            end
        end
        
        R_deterministic = define_deterministic_R(E, S)
        if sdp == 1
            R_deterministic_sdp = define_deterministic_R_sdp(E, S)
        end
        R_probabilistic3 = define_probabilistic_R(E, S, q1)
        R_probabilistic6 = define_probabilistic_R(E, S, q2)
        R_probabilistic9 = define_probabilistic_R(E, S, q3)

        Phi1 = _info_RES(R_deterministic, E, S)
        if sdp == 1
            Phi1_sdp = _info_RES(R_deterministic_sdp, E, S)
        end
        Phi03 = _info_RES(R_probabilistic3, E, S)
        Phi06 = _info_RES(R_probabilistic6, E, S)
        Phi09 = _info_RES(R_probabilistic9, E, S)
        
        ref_fu = reference_fuRES(E, x)
        
        for i=1:length(list_x_p)
            p = list_x_p[i]
            
            res1 = result_REpS(Phi1, x, p)
            if sdp == 1
                res1_sdp = result_REpS(Phi1_sdp, x, p)
            end
            res03 = result_REpS(Phi03, x, p)
            res06 = result_REpS(Phi06, x, p)
            res09 = result_REpS(Phi09, x, p)

            list_y_pavg3[i,l], list_y_c_favg3[i, l] = res03
            list_y_pavg6[i,l], list_y_c_favg6[i, l] = res06
            list_y_pavg9[i,l], list_y_c_favg9[i, l] = res09
                        
            list_y_favg[i,l] = res1[2]
            if sdp == 1
                list_y_favg_sdp[i,l] = res1_sdp[2]
            end
            
            list_y_ref[i, l] = (1-p) + (p * ref_fu / (x^2))
        end
    end
    
    drawf = errorline(list_x_p, list_y_c_favg3, errorstyle=:ribbon, errortype=:percentile, label=latexstring("\\mathcal{R}_{0, $(q1)}"), 
    groupcolor = :black, secondarycolor=:matched, yformatter = x -> latexstring("$(round(100*x)/100)"), xformatter = x -> latexstring("$(round(100*x)/100)"))
    errorline!(list_x_p, list_y_c_favg6, errorstyle=:ribbon, errortype=:percentile, label=latexstring("\\mathcal{R}_{0, $(q2)}"), groupcolor = :green, secondarycolor=:matched)
    errorline!(list_x_p, list_y_c_favg9, errorstyle=:ribbon, errortype=:percentile, label=latexstring("\\mathcal{R}_{0, $(q3)}"), groupcolor = :red, secondarycolor=:matched)
    if sdp == 1
        errorline!(list_x_p, list_y_favg_sdp, errorstyle=:ribbon, errortype=:percentile, label=L"\mathcal{R}_{1, \mathtt{sdp}}", groupcolor = :brown, secondarycolor=:matched)
    end
    errorline!(list_x_p, list_y_favg, errorstyle=:ribbon, errortype=:percentile, label=L"\mathcal{R}_{1}", groupcolor = :blue, secondarycolor=:matched)
    
    plot!(list_x_p, map(i -> mean(list_y_ref[i, :]), 1:length(list_x_p)), label=latexstring("\\mathcal{R}_{*}"), line=(:dash, :gray))
    
    xlabel!(drawf, L"p")
    title!(drawf, latexstring("\\mathtt{Fidelity \\; \\; \\;function \\; \\; \\; for \\; \\; \\;  }(y, x, r, k) = ($(y),$(x),$(rankE),$(k))"))
    ylabel!(drawf, L"F_{\mathtt{|avg}}(\mathcal{R} \mathcal{E}_p \mathcal{S} )")
    savefig(drawf, "$(y)_$(x)_$(rankE)_$(k)_fid.pdf")

    drawp = errorline(list_x_p, list_y_pavg3, errorstyle=:ribbon, errortype=:percentile, label=latexstring("\\mathcal{R}_{0, $(q1)}"), groupcolor = :black, secondarycolor=:matched, 
    yformatter = x -> latexstring("$(round(1000*x)/1000)"), xformatter = x -> latexstring("$(round(100*x)/100)"))
    errorline!(list_x_p, list_y_pavg6, errorstyle=:ribbon, errortype=:percentile, label=latexstring("\\mathcal{R}_{0, $(q2)}"), groupcolor = :green, secondarycolor=:matched)
    errorline!(list_x_p, list_y_pavg9, errorstyle=:ribbon, errortype=:percentile, label=latexstring("\\mathcal{R}_{0, $(q3)}"), groupcolor = :red, secondarycolor=:matched)
    
    xlabel!(drawp, L"p")
    title!(drawp, latexstring("\\mathtt{Probability \\; \\; \\; of \\; \\; \\; success \\; \\; \\;  }(y, x, r, k) = ($(y),$(x),$(rankE),$(k))"))
    ylabel!(drawp, L"p_{\mathtt{avg}}(\mathcal{R} \mathcal{E}_p \mathcal{S} )")
    savefig(drawp, "$(y)_$(x)_$(rankE)_$(k)_prob.pdf")
end

#########################################################

# Test for random channels:
# y - dimension of Y
# x - dimension of X
# rankE - expected rank of E
# k - the number of sampled random channels
# q1, q2, q3 - parameters of probabilistic decoding operations
# sdp - 1 = add deterministic decoding operation calculated via sdp
# n_points - the number of points

generate_data(4, 2, 2, 10, 0.1, 0.4, 0.8, 1, 100)
