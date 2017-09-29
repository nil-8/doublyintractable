using LightGraphs

function count_edges(g::LightGraphs.Graph)
  return ne(g)
end

function count_edges(g::LightGraphs.Graph, e::LightGraphs.Edge)
  return 1
end

function count_3cycles(g::LightGraphs.Graph)
  # reference: Manber, U. (1989). Introduction to algorithms: a creative approach, p. 326
  # complexity: O(M(nv(g))), where M(n) is cost of matrix multiplication of n x n matrix

  A = LightGraphs.adjacency_matrix(g)
  B = A*A
  count = 0
  for e in LightGraphs.edges(g)
    count += B[src(e),dst(e)]
  end
  return Int(count/3)
end

function count_3cycles(g::LightGraphs.Graph, e::LightGraphs.Edge)
  # counts the number of 3-cycles which contain e (after adding e to g if necessary)
  # NOTE: e is not required to belong to g
  u = src(e)
  v = dst(e)
  length(LightGraphs.common_neighbors(g,u,v))
end

function count_kstars(g::LightGraphs.Graph, k::Integer)
  return sum([binomial(d,k) for d in degree(g)])
end

function count_2stars(g::LightGraphs.Graph)
  return count_kstars(g, 2)
end

function count_2stars(g::LightGraphs.Graph, e::LightGraphs.Edge)
  # counts the number of 2-stars which contain e (after adding e to g if necessary)

  u = src(e)
  v = dst(e)
  n1 = degree(g,u)
  n2 = degree(g,v)
  if e in edges(g) # the edge between u and v shouldn't be counted as 2-star.
    return n1 + n2 - 2
  else
    return n1 + n2
  end
end

function RandomEdge(nv, n_samples = 1)
  # NOTE: is it overkill to use erdos_renyi?
  randomEdges = Array{Edge}(n_samples)
  for i=1:n_samples
    v1 = rand(1:nv)
    v2 = rand(1:nv-1)
    if v2 >= v1
      v2 += 1
    end
    randomEdges[i] = Edge(v1,v2)
  end
  if n_samples == 1
    return randomEdges[1]
  else
    return randomEdges
  end
end

function lt(g::LightGraphs.Graph, h::LightGraphs.Graph)
  for e in edges(g)
    if ~has_edge(h,e)
      return false
    end
  end
  return true
end

function GlauberStep!(g::LightGraphs.Graph, theta::Vector{Float64}, s::Vector{Function}; e::LightGraphs.Edge=Edge(0,0), U=-1.0)

  if e == Edge(0,0) || U < 0
    n = nv(g)
    e = RandomEdge(n)
    U = rand()
  end

  q = exp(sum(theta .* [s_k(g,e) for s_k in s]))
  p_include = q / (1+q)
  if U <= p_include
      add_edge!(g, e)
  else
    rem_edge!(g,e)
  end
  return g
end

function Glauber!(g::LightGraphs.Graph, theta::Vector{Float64}, s::Vector{Function}, n_iter::Integer)

  randomEdges = RandomEdge(nv(g),n_iter)
  randomUniforms = rand(n_iter)

  for i in 1:n_iter
    GlauberStep!(g, theta, s, e = randomEdges[i], U = randomUniforms[i])
  end
  return g
end

function ExchangeAlgorithm(s::Vector{Function}, Y_data, priorVariance::Float64, proposalVariance::Float64, Theta0, n_iter::Int)

  dim = length(Theta0)
  Theta = Array{Float64,2}(dim,n_iter + 1)
  Theta[:,1] = Theta0
  Theta_current = Theta0
  U = rand(n_iter)
  Z = sqrt(proposalVariance) * randn(dim,n_iter)
  n = nv(Y_data)
  s_data = [f(Y_data) for f in s]

  for i=1:n_iter
    Theta_proposed = Theta_current + Z[:,i]

    # hacky way to prevent negative theta_2: prior restricted to theta_2 > 0
    if (Theta_proposed[2] > 0)
      Y_proposed = ProppWilsonButts(n, s, Theta_proposed)
      s_proposed = [f(Y_proposed) for f in s]
      mh_numerator = exp(dot(Theta_proposed,s_data)) * exp(-norm(Theta_proposed)^2/(2*priorVariance)) * exp(dot(Theta_current,s_proposed))
      mh_denominator = exp(dot(Theta_current,s_data)) * exp(-norm(Theta_current)^2/(2*priorVariance)) * exp(dot(Theta_proposed,s_proposed))
      if U[i] <= mh_numerator/mh_denominator
        Theta_current = Theta_proposed
      end
    end
    Theta[:,i+1] = Theta_current
    # println("Iteration ", i, ": theta = ", Theta_current)
  end

  return Theta
end

function ProppWilsonButts(n::Integer, s::Vector{Function}, theta::Vector{Float64})

  n_steps = div(n * (n-1), 2) # identical to number of edges
  # distance = zeros(n_steps +1)
  coalesced = false
  current_iteration = 1
  while ~coalesced
    x_lower = Graph(n) # empty graph
    x_upper = CompleteGraph(n)
    x_test = LightGraphs.erdos_renyi(n, 0.5)
    if current_iteration == 1
      E = RandomEdge(n, n_steps)
      U = rand(n_steps)
    else
      E = vcat(RandomEdge(n, n_steps - div(n_steps,2)), E)
      U = vcat(rand(n_steps - div(n_steps,2)), U)
    end
    for i=1:n_steps
      delta_lower = [s_k(x_lower, E[i]) for s_k in s]
      delta_upper = [s_k(x_upper, E[i]) for s_k in s]
      exponent_upper = 0
      exponent_lower = 0
      for k=1:length(theta) # invariant: exponent_upper >= exponent_lower
        if theta[k] > 0
          exponent_upper += theta[k] * delta_upper[k]
          exponent_lower += theta[k] * delta_lower[k]
        else
          exponent_upper += theta[k] * delta_lower[k]
          exponent_lower += theta[k] * delta_upper[k]
        end
      end
      p_upper = 1/(1 + exp(-exponent_upper))
      p_lower = 1/(1 + exp(-exponent_lower))
      if p_upper < p_lower
        println("p_upper < p_lower")
        throw(ErrorException("p_upper < p_lower"))
      end
      if U[i] > p_upper
        rem_edge!(x_upper,E[i])
      else
        add_edge!(x_upper,E[i])
      end
      if U[i] > p_lower
        rem_edge!(x_lower,E[i])
      else
        add_edge!(x_lower,E[i])
      end
      GlauberStep!(x_test, theta, s, e=E[i], U=U[i])
    end
    coalesced = (x_lower == x_upper)
    if coalesced
      if (x_lower != x_test)
        throw(ErrorException("Test chain not equal to coalesced chains."))
      end
      println("Propp-Wilson-Butts coalescence after ", current_iteration, " iterations.")
      # print(".")
      return x_upper, current_iteration
    end
    n_steps = 2 * n_steps
    current_iteration += 1
  end
end

function EnumerateTuples(n::Integer, q::Integer)
    # enumerate tuples of distinct integers in the range 1, ..., n
  if (q == 1)
    return Vector{Integer}(1:n)
  else
    vecsize = binomial(n,q)
    result = Array{Integer}(vecsize, q)
    current_index = 1
    for k = q:n
      n_subvectors = binomial(k-1,q-1)
      result[current_index:current_index+n_subvectors-1,:] = hcat(repmat([k],n_subvectors, 1), EnumerateTuples(k-1,q-1))
      current_index += n_subvectors
    end
    return result
  end
end

function TupleCycles(tuple)
  n = length(tuple)
  current_tuple = copy(tuple)
  result = Array{Integer}(n, n)
  for i=1:n
    result[i,:] = current_tuple
    if i < n
      current_tuple = vcat(current_tuple[2:n],current_tuple[1])
    end
  end
  return result
end


function EnumeratePstars(n_vertices::Integer, p::Integer)
  # the idea is to get all (p+1)-tuples from the integers 1, ..., n_vertices
  # and then cycle through each tuple so that every element of the tuple is
  # centre of the star

  n_pstars = (p+1) * binomial(n_vertices, p+1)
  pstars = Array{Integer,2}(n_pstars, p+1)
  current_index = 1
  tuples = EnumerateTuples(n_vertices, p+1)
  for i=1:binomial(n_vertices, p+1)  # number of tuples
    tuple = tuples[i,:]
    pstars[current_index:current_index+p,:] = TupleCycles(tuple)
    current_index += p+1
  end

  return pstars

end

function PstarInGraph(p_star::Vector{Integer}, x::LightGraphs.Graph)
  for i=2:length(p_star)
    if ~has_edge(x,Edge(p_star[1], p_star[i]))
      return false
    end
  end
  return true
end

function AddPstarToGraph(p_star::Vector{Integer}, x::LightGraphs.Graph)
  for i=2:length(p_star)
    add_edge!(x,Edge(p_star[1], p_star[i]))
  end
  return x
end

function SwendsenWangPstars(g::LightGraphs.AbstractGraph, theta::Vector{Float64}, p::Integer, n_iter::Integer)

  # Swendsen-Wang applied to p-star statistics
  if length(theta) != 2
    throw(ErrorException("only for 2-dimensional statistics"))
  end

  if theta[2] < 0
    throw(ErrorException("currently only for theta2 >= 0"))
  end
  n_vertices = nv(g)
  p_stars = EnumeratePstars(n_vertices, p)
  n_pstars = size(p_stars,1)
  return_value = Vector{Graph}(n_iter + 1)
  return_value[1] = g

  for i=1:n_iter
    p_stars_in_g = Vector{Bool}(n_pstars)
    for j=1:n_pstars
      p_stars_in_g[j] = PstarInGraph(p_stars[j,:], g)
    end
    y = rand(n_pstars) .* exp.(theta[2]*p_stars_in_g)
    g = Graph(n_vertices) # clear graph
    for j=1:n_pstars
      if y[j] > 1
        AddPstarToGraph(p_stars[j,:], g)
      end
    end
    for e in edges(CompleteGraph(n_vertices))
      if ~has_edge(g,e) && rand() <= exp(theta[1])/(1+exp(theta[1]))
        add_edge!(g,e)
      end
    end
    return_value[i+1] = g
  end
  return return_value

end

function SwendsenWangPstarsStep(g::LightGraphs.AbstractGraph, theta::Vector{Float64}, p::Integer; U1::Vector{Float64} = Vector{Float64}([]), U2::Vector{Float64} = Vector{Float64}([]), p_stars::Array{Integer} = Array{Integer}([]), all_edges::Vector{Edge{Int64}} = Vector{Edge}([]))

  # Swendsen-Wang applied to p-star statistics
  if length(theta) != 2
    throw(ErrorException("only for 2-dimensional statistics"))
  end

  if theta[2] < 0
    throw(ErrorException("currently only for theta2 >= 0"))
  end

  n_vertices = nv(g)

  if length(p_stars) == 0
    p_stars = EnumeratePstars(n_vertices, p)
  end
  n_pstars = size(p_stars,1)

  if length(all_edges) == 0
    all_edges = [e for e in edges(CompleteGraph(n_vertices))]
  end

  if length(U1) == 0 || length(U2) == 0
    U1 = rand(n_pstars)
    U2 = rand(length(all_edges))
  end

  p_stars_in_g = Vector{Bool}(n_pstars)
  for i=1:n_pstars
    p_stars_in_g[i] = PstarInGraph(p_stars[i,:], g)
  end
  y = U1 .* exp.(theta[2]*p_stars_in_g)
  x = Graph(n_vertices) # clear graph
  for i=1:n_pstars
    if y[i] > 1
      AddPstarToGraph(p_stars[i,:], x)
    end
  end
  for i=1:length(all_edges)
    if ~has_edge(x,all_edges[i]) && U2[i] <= exp(theta[1])/(1+exp(theta[1]))
      add_edge!(x,all_edges[i])
    end
  end
  return x
end

function ProppWilsonSwendsenWangPstars(n::Integer, theta::Vector{Float64}, p::Integer)

  n_steps = 2
  # distance = zeros(n_steps +1)
  coalesced = false
  current_iteration = 1
  p_stars = EnumeratePstars(n, p)
  n_pstars = size(p_stars, 1)
  all_edges = [e for e in edges(CompleteGraph(n))]
  n_edges = length(all_edges)
  while ~coalesced
    x_lower = Graph(n) # empty graph
    x_upper = CompleteGraph(n)
    x_test = LightGraphs.erdos_renyi(n, 0.5)
    if current_iteration == 1
      U1 = rand(n_pstars, n_steps)
      U2 = rand(n_edges, n_steps)
    else
      U1 = hcat(rand(n_pstars, n_steps - div(n_steps,2)), U1)
      U2 = hcat(rand(n_edges, n_steps - div(n_steps,2)), U2)
    end
    for i=1:n_steps
      x_lower = SwendsenWangPstarsStep(x_lower, theta, p, U1 = U1[:,i], U2 = U2[:,i], p_stars = p_stars, all_edges = all_edges)
      x_upper = SwendsenWangPstarsStep(x_upper, theta, p, U1 = U1[:,i], U2 = U2[:,i], p_stars = p_stars, all_edges = all_edges)
      x_test = SwendsenWangPstarsStep(x_test, theta, p, U1 = U1[:,i], U2 = U2[:,i], p_stars = p_stars, all_edges = all_edges)
      if ~lt(x_lower,x_upper)
        throw(ErrorException("Partial order lost."))
      end

    end
    coalesced = (x_lower == x_upper)
    if coalesced
      if (x_lower != x_test)
        throw(ErrorException("Test chain not equal to coalesced chains."))
      end
      println("Propp-Wilson-Swendsen-Wang coalescence after ", current_iteration, " iterations.")
      # print(".")
      return x_upper, current_iteration
    end
    n_steps = 2 * n_steps
    current_iteration += 1
  end
end
