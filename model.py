start_time = 0
t = list(range(1, 10001))
#start_time = time.time()

errors = []
manual_errors = []
values = []
true_values = []
station_values = []
time_interval = []
est_points = []
actual_points = []

p = 6  # Population size / Number of chromosomes
m = 3 # Number of variables / genes
n = 5 # Number of digits / Alleles
threshold = 3 # Crossover threshold
no_of_child = 3 # Number of childrens 
num_of_mutation = 3 # Number of point of mutation

#Dyamic Range
obj_fun = "Ackley"
x_min_range = -10
x_max_range = 10
y_min_range = -10
y_max_range = 10
z_min_range = -10
z_max_range = 10
optm = "max" 

# Compute the Euclidean distance between two points.
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Generate possible gene values and binary representation of genes.
def gene_generator(min_range, max_range, m, n):
    num_of_genes = 2**n - 1
    interval = (max_range - min_range) / num_of_genes
    gene_value = []
    for i in range(num_of_genes+1):
        gene_value.append(min_range+i*interval)
    gene = [[*bin(i)[2:].zfill(n)] for i in range(0, num_of_genes+1)]
    return gene_value, gene

# Griewank function used as an objective function.
def griewank(x, start_time):
    x = x - (time.time()-start_time)/2
    sum_term = np.sum(x**2 / 4000)
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return float(1 + sum_term - prod_term)

# Rosenbrock function used as an objective function.
def rosenbrock(x, a=1, b=100):
    sum_term = np.sum((a - x[:-1])**2)
    prod_term = np.sum(b * (x[1:] - x[:-1]**2)**2)
    return (sum_term + prod_term)

# Sphere function used as an objective function.
def sphere(x):
    return np.sum(x**2)

# Ackley function used as an objective function. Adjust x based on elapsed time since start_time.

# Function to introduce AWGN noise
def add_awgn_noise(x, snr_db):
    snr_linear = 10**(snr_db / 10.0)
    power_signal = np.mean(x**2)
    power_noise = power_signal / snr_linear
    noise = np.random.normal(0, np.sqrt(power_noise), x.shape)
    return x + noise

# Rayleigh fading
def apply_rayleigh_fading(x):
    return x * np.random.rayleigh(scale=1.0, size=x.shape)

# Rician fading
def apply_rician_fading(x, k_factor=3):
    s = np.sqrt(k_factor / (k_factor + 1))
    sigma = np.sqrt(1 / (2 * (k_factor + 1)))
    fading = np.random.normal(loc=s, scale=sigma, size=x.shape)
    return x * fading

# Nakagami-m fading
def apply_nakagami_fading(x, m=1):
    omega = 1  # Omega is a scaling parameter
    fading = np.random.gamma(m, omega/m, x.shape)
    return x * fading

# Co-channel interference
def add_co_channel_interference(x, interference_level=0.1):
    interference = np.random.normal(0, interference_level, x.shape)
    return x + interference

# Modified Ackley function with multimodality and impairments
def ackley(x, start_time, ti, snr_db=20, fading_type='rician', interference_level=0.1, nakagami_m=1):
    # Apply time decay
    r = 10
    import math
    x[0] = x[0] - r*math.cos(0.2*(ti-start_time))
    x[1] = x[1] - r*math.sin(0.2*(ti-start_time))
    x[2] = x[2] - (0.2*(ti-start_time))
    # Apply impairments
    x = add_awgn_noise(x, snr_db)
    
    if fading_type == 'rayleigh':
        x = apply_rayleigh_fading(x)
    elif fading_type == 'rician':
        x = apply_rician_fading(x)
    elif fading_type == 'nakagami':
        x = apply_nakagami_fading(x, m=nakagami_m)
    
    # Add co-channel interference
    x = add_co_channel_interference(x, interference_level)
    
    # Check if the input is within bounds
    if np.all(x >= -20) and np.all(x <= 20):
        # Ackley function parameters
        a = 20
        b = 0.2
        c = 2 * np.pi
        n = len(x)
        
        # Compute sum terms for the Ackley function
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        
        # Compute terms of the Ackley function
        term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)
        
        # Adding multimodal aspect by creating an additional sine wave modulation
        modulation = np.sin(5 * np.pi * x).sum()
        
        # Final result
        return 20 - (term1 + term2 + a + np.exp(1)) + modulation
    else:
        return 0

# Rastrigin function used as an objective function.
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Zakharov function used as an objective function.
def zakharov(x):
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
    return sum1 + sum2**2 + sum2**4

# Function to generate values for the population based on the given objective function.
def population_value_generator(obj_fun, population, x_gene_value, y_gene_value, z_gene_value, start_time, ti):
    population_values = []
    obj_func_values = []
    for chromosome in population: # Evaluate each chromosome in the population
        chromosome_values = []
        dim = 0
        for gene in chromosome:
            bin_str = ''.join(map(str, gene))
            dec_num = int(bin_str, 2)
            if dim == 0:
                chromosome_values.append(x_gene_value[dec_num])
            if dim == 1:
                chromosome_values.append(y_gene_value[dec_num])
            if dim == 2:
                chromosome_values.append(z_gene_value[dec_num])
            dim += 1
        if obj_fun == "Griewank":
            obj_func_value = griewank(np.array(chromosome_values), start_time)
        elif obj_fun == "Rosenbrock":
            obj_func_value = rosenbrock(np.array(chromosome_values))
        elif obj_fun == "Ackley":
            obj_func_value = ackley(np.array(chromosome_values),start_time, ti)
        elif obj_fun == "Sphere":
            obj_func_value = sphere(np.array(chromosome_values))
        elif obj_fun == "Rastrigin":
            obj_func_value = rastrigin(np.array(chromosome_values))
        elif obj_fun == "Zakharov":
            obj_func_value = zakharov(np.array(chromosome_values))
        else:
            print("Give Correct Objective Function") 
        obj_func_values.append(obj_func_value)
        population_values.append(chromosome_values)
    return population_values, obj_func_values

# Select a subset of the population based on objective function values and optimization goal.
def thresholding(optm, obj_func_values, threshold):
    if optm == "min":
        selected_population_ind = np.argsort(obj_func_values)[:threshold]
    elif optm == "max":
        selected_population_ind = np.argsort(obj_func_values)[::-1][:threshold] #[:threshold]
    else:
        print("Define optimisation as maximise or minimise")
    return selected_population_ind

# Estimate fitness values for the selected population.
def fitness_estimation(selected_population, obj_func_values):
    fitness_values = []
    for ind in selected_population:
        #fitness = 1/(1 + obj_func_values[ind])
        fitness = (obj_func_values[ind])**4
        fitness_values.append(fitness)
    return(fitness_values)

 # Perform roulette wheel selection based on fitness values.
def Roulette_wheel(selected_population_ind, fitness_values, threshold, no_of_child):    
    sorted_indices = np.argsort(fitness_values)[::-1]  # Descending order
    ranks = np.zeros_like(sorted_indices)
    for rank, index in enumerate(sorted_indices):
        ranks[index] = rank + 1
    
    rank_sum = sum(ranks)
    fitness_probability = [(rank / rank_sum) for rank in ranks]  # Add a small value to avoid zero probability
    
    parents = set()
    while len(parents) < no_of_child:
        parent = np.random.choice(selected_population_ind, p=fitness_probability, size=2, replace=False).tolist()
        parents.add(tuple(parent))
    
    return tuple(parents)

# Perform crossover operation to generate children from parent chromosomes.
def crossover(parents, population, m):
    children = []
    for parent in parents:
        cross_pt = np.random.randint(1, m)
        child = np.concatenate((population[parent[0]][:cross_pt], population[parent[1]][cross_pt:]))
        children.append(child)
    return children

# Perform mutation on the children chromosomes.
def mutation(children,m,n):
    mutated_children = []
    for child in children:
        m_mutate = np.random.randint(0, m)
        n_mutate = np.random.randint(0, n)
        if child[m_mutate][n_mutate] == 0:
            child[m_mutate][n_mutate] = 1
        else:
            child[m_mutate][n_mutate] = 0
        mutated_children.append(child)
    return mutated_children

#Genetic Algorithm Definition

def genetic_algorithm(file_name, seed, p, m, n, threshold, no_of_child, x_min_range, x_max_range, y_min_range, y_max_range, z_min_range, z_max_range, obj_fun, num_of_mutation, optm, iter, start_time, previous_population, ti):

    #np.random.seed(seed)
    f = []
    g = []
    f_lists = [[] for _ in range(p)]

    x_gene_value, x_gene = gene_generator(x_min_range, x_max_range, m, n)
    y_gene_value, y_gene = gene_generator(y_min_range, y_max_range, m, n)
    z_gene_value, z_gene = gene_generator(z_min_range, z_max_range, m, n)
    
    population  = previous_population

    for i in range(iter):

        population_values, obj_func_values = population_value_generator(obj_fun, population, x_gene_value, y_gene_value, z_gene_value, start_time, ti)
            
        #Monitoring Variables
        for j, value in enumerate(obj_func_values):
            f_lists[j].append(value)
        if optm == "min":
            f.append(min(obj_func_values))
            g.append(population_values[obj_func_values.index(max(obj_func_values))])
        elif optm == "max":
            f.append(max(obj_func_values))
            g.append(population_values[obj_func_values.index(max(obj_func_values))])
        else:
            print("Define optimisation as maximise or minimise")

        #Thresholding
        selected_population_ind = thresholding(optm, obj_func_values, threshold)
        selected_population = []
        for i in selected_population_ind:
            selected_population.append(population[i])

        #Fitness Estimation
        fitness_values = fitness_estimation(selected_population_ind, obj_func_values)

        #Roulette_wheel
        parents = Roulette_wheel(selected_population_ind, fitness_values, threshold, no_of_child)

        # Crossover
        children = crossover(parents, population, m)
           
        if num_of_mutation == 0:
            new_children = children
        else:
            for i in range(num_of_mutation):
                new_children = mutation(children, m, n) #Issue with mutation. Mutate at many places
            
        # New Population
        population = selected_population + new_children
        
    value = f[len(f)-1]
    location = g[len(g)-1]
    index = f.index(value) + 1
    score = 100/(1+(index*value))
    result = {"value": value, "iteration": index, "location": location}
    fs = {"f": f, "f_lists": f_lists}
    
    return result, fs, population
