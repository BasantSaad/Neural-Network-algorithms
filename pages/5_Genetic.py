import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time

# Set the page config
st.set_page_config(
    page_title="Genetic Algorithm Lab Visualization",
    page_icon="🧬",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("🧬 Genetic Algorithm Lab Visualization")
st.markdown("""
This application demonstrates genetic algorithms for the lab problems described in the task.
""")

# Sidebar for algorithm parameters
st.sidebar.header("Algorithm Parameters")

problem_type = st.sidebar.selectbox(
    "Problem Type",
    ["Group 1: Digit Chromosomes", "Group 2: Binary MAX-ONE"]
)
#                                                        *population size* N = 10
population_size = st.sidebar.slider(
    "Population Size",
    min_value=4,
    max_value=100,
    value=10,
    step=2,
    help="Number of individuals in the population"
)
#                                                        *generations* G = 50
generations = st.sidebar.slider(
    "Number of Generations",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
    help="Number of generations to evolve"
)
#                                                        *mutation rate* Pm = 0.1
mutation_rate = st.sidebar.slider(
    "Mutation Rate",
    min_value=0.01,
    max_value=0.5,
    value=0.1,
    step=0.01,
    help="Probability of mutation"
)
#                                                                *crossover rate* Pc = 0.8
crossover_rate = st.sidebar.slider(
    "Crossover Rate",
    min_value=0.1,
    max_value=1.0,
    value=0.8,
    step=0.05,
    help="Probability of crossover"
)
#                                                                 *elitism* E = True
elitism = st.sidebar.checkbox(
    "Use Elitism",
    value=True,
    help="Keep the best individuals in each generation"
)  

# Show problem description
if problem_type == "Group 1: Digit Chromosomes":
    st.markdown("""
    ## Problem Description: Group 1

    Chromosomes are represented as strings of 8 digits (0-9).

    **Fitness function:** `f(x) = (a + b) - (c + d) + (e + f) - (g + h)`

    The goal is to maximize this fitness function.

    **Example individuals from the task:**
    - x1 = 65413532 → f(x1) = (6+5)-(4+1)+(3+5)-(3+2) = 9
    - x2 = 87126601 → f(x2) = (8+7)-(1+2)+(6+6)-(0+1) = 23
    - x3 = 23921285 → f(x3) = (2+3)-(9+2)+(1+2)-(8+5) = -16
    - x4 = 41852094 → f(x4) = (4+1)-(8+5)+(2+0)-(9+4) = -19

    The optimal solution is 99009900 with fitness 36.
    """)
else:  # Group 2: Binary MAX-ONE
    st.markdown("""
    ## Problem Description: Group 2

    An individual is encoded as a string of binary digits.
    
    **Fitness function (MAX-ONE):** The number of ones in the genetic code.

    The goal is to maximize this fitness function.

    **Example individuals from the task:**
    - s1 = 1111010101 → f(s1) = 7
    - s2 = 0111000101 → f(s2) = 5
    - s3 = 1110110101 → f(s3) = 7
    - s4 = 0100010011 → f(s4) = 4
    - s5 = 1110111101 → f(s5) = 8
    - s6 = 0100110000 → f(s6) = 3
    """)

# Genetic Algorithm Implementation
class GeneticAlgorithm:
    def __init__(self, problem_type, **kwargs):
        self.problem_type = problem_type
        self.population_size = kwargs.get('population_size', 10)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        self.crossover_rate = kwargs.get('crossover_rate', 0.8)
        self.elitism = kwargs.get('elitism', True)
        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        self.best_fitness = float('-inf')  # Both problems are maximization
        self.generations_log = []
        
        # Problem-specific initialization
        if problem_type == "Group 1: Digit Chromosomes":
            self.chromosome_length = 8
            self.initialize_digit_population()
        
        elif problem_type == "Group 2: Binary MAX-ONE":
            self.chromosome_length = 10
            self.initialize_binary_population()
    
    def initialize_digit_population(self):
        self.population = []
        for i in range(self.population_size):
            # For the first 4 individuals in the initial population, use the examples from the task if possible
            if i == 0 and self.population_size >= 4:
                chromosome = [6, 5, 4, 1, 3, 5, 3, 2]  # x1
            elif i == 1 and self.population_size >= 4:
                chromosome = [8, 7, 1, 2, 6, 6, 0, 1]  # x2
            elif i == 2 and self.population_size >= 4:
                chromosome = [2, 3, 9, 2, 1, 2, 8, 5]  # x3
            elif i == 3 and self.population_size >= 4:
                chromosome = [4, 1, 8, 5, 2, 0, 9, 4]  # x4
            else:
                # Random chromosome with digits 0-9
                chromosome = [random.randint(0, 9) for _ in range(self.chromosome_length)]
            
            self.population.append(chromosome)
    
    def initialize_binary_population(self):
        self.population = []
        for i in range(self.population_size):
            # For the first 6 individuals in the initial population, use the examples from the task if possible
            if i == 0 and self.population_size >= 6:
                chromosome = [1, 1, 1, 1, 0, 1, 0, 1, 0, 1]  # s1
            elif i == 1 and self.population_size >= 6:
                chromosome = [0, 1, 1, 1, 0, 0, 0, 1, 0, 1]  # s2
            elif i == 2 and self.population_size >= 6:
                chromosome = [1, 1, 1, 0, 1, 1, 0, 1, 0, 1]  # s3
            elif i == 3 and self.population_size >= 6:
                chromosome = [0, 1, 0, 0, 0, 1, 0, 0, 1, 1]  # s4
            elif i == 4 and self.population_size >= 6:
                chromosome = [1, 1, 1, 0, 1, 1, 1, 1, 0, 1]  # s5
            elif i == 5 and self.population_size >= 6:
                chromosome = [0, 1, 0, 0, 1, 1, 0, 0, 0, 0]  # s6
            else:
                # Random binary chromosome
                chromosome = [random.randint(0, 1) for _ in range(self.chromosome_length)]
            
            self.population.append(chromosome)
    
    def fitness_function(self, individual):      
        if self.problem_type == "Group 1: Digit Chromosomes":
            # f(x) = (a + b) - (c + d) + (e + f) - (g + h)
            a, b, c, d, e, f, g, h = individual
            return (a + b) - (c + d) + (e + f) - (g + h)
        
        elif self.problem_type == "Group 2: Binary MAX-ONE":
            # Count the number of ones
            return sum(individual)
    
    def selection(self, population, fitnesses):
        # Convert to numpy arrays for easier manipulation
        fitnesses = np.array(fitnesses)
        
        # Adjust negative fitnesses for selection probability calculation
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            # Shift all fitnesses to be non-negative
            adjusted_fitnesses = fitnesses - min_fitness + 1
        else:
            adjusted_fitnesses = fitnesses
        
        # Calculate selection probabilities (higher fitness -> higher probability)
        total_fitness = np.sum(adjusted_fitnesses)
        if total_fitness == 0:  # All fitnesses are zero
            selection_probs = np.ones(len(adjusted_fitnesses)) / len(adjusted_fitnesses)
        else:
            selection_probs = adjusted_fitnesses / total_fitness
        
        # Roulette wheel selection
        selected_indices = np.random.choice(
            len(population), 
            size=len(population), 
            p=selection_probs,
            replace=True
        )
        
        return [population[i].copy() for i in selected_indices]
    
    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy(), "No crossover (copied parents)"
        
        # Apply different crossover types based on the problem
        crossover_type = random.choice(["one-point", "two-point", "uniform"])
        
        if crossover_type == "one-point":
            # One-point crossover
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            crossover_description = f"One-point crossover at position {crossover_point}"
            
        elif crossover_type == "two-point":
            # Two-point crossover
            points = sorted(random.sample(range(1, len(parent1)), 2))
            point1, point2 = points[0], points[1]
            child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
            crossover_description = f"Two-point crossover at positions {point1} and {point2}"
            
        else:  # uniform
            # crossover (randomly select genes from either parent)
            child1, child2 = [], []
            for i in range(len(parent1)):
                if random.random() < 0.5:
                    child1.append(parent1[i])
                    child2.append(parent2[i])
                else:
                    child1.append(parent2[i])
                    child2.append(parent1[i])
            crossover_description = "Uniform crossover"
        
        return child1, child2, crossover_description
    
    def mutate(self, individual):
        mutated = individual.copy()
        mutation_occurred = False
        mutation_positions = []
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutation_occurred = True
                mutation_positions.append(i)
                
                if self.problem_type == "Group 1: Digit Chromosomes":
                    # Replace with a random digit 0-9
                    mutated[i] = random.randint(0, 9)
                else:  # Binary
                    # Flip the bit
                    mutated[i] = 1 - mutated[i]
        
        return mutated, mutation_occurred, mutation_positions
    
    def evolve(self):
        # Calculate fitness for each individual
        fitnesses = [self.fitness_function(ind) for ind in self.population]
        
        # Track statistics
        best_idx = np.argmax(fitnesses)
        current_best = fitnesses[best_idx]
        current_avg = np.mean(fitnesses)
        
        self.avg_fitness_history.append(current_avg)
        
        if current_best > self.best_fitness:
            self.best_fitness = current_best
            self.best_individual = self.population[best_idx].copy()
        
        self.best_fitness_history.append(self.best_fitness)
        
        # Elitism: preserve the best individual
        if self.elitism:
            elite = self.population[best_idx].copy()
        
        # Selection
        selected = self.selection(self.population, fitnesses)
        
        # Create new population through crossover and mutation
        new_population = []
        crossover_log = []
        mutation_log = []
        
        while len(new_population) < self.population_size:
            # Select parents
            parent1_idx = random.randint(0, len(selected) - 1)
            parent2_idx = random.randint(0, len(selected) - 1)
            
            parent1 = selected[parent1_idx]
            parent2 = selected[parent2_idx]
            
            # Crossover
            if len(new_population) + 2 <= self.population_size:
                child1, child2, crossover_description = self.crossover(parent1, parent2)
                
                # Mutation
                child1, mutation1_occurred, mutation1_positions = self.mutate(child1)
                child2, mutation2_occurred, mutation2_positions = self.mutate(child2)
                
                # Track crossover and mutation for logging
                crossover_log.append({
                    'parents': [parent1, parent2],
                    'children': [child1, child2],
                    'description': crossover_description
                })
                
                if mutation1_occurred:
                    mutation_log.append({
                        'original': parent1,
                        'mutated': child1,
                        'positions': mutation1_positions
                    })
                
                if mutation2_occurred:
                    mutation_log.append({
                        'original': parent2,
                        'mutated': child2,
                        'positions': mutation2_positions
                    })
                
                new_population.append(child1)
                new_population.append(child2)
            else:
                # If we need just one more individual
                child1, child2, crossover_description = self.crossover(parent1, parent2)
                child1, mutation_occurred, mutation_positions = self.mutate(child1)
                
                crossover_log.append({
                    'parents': [parent1, parent2],
                    'children': [child1],
                    'description': crossover_description
                })
                
                if mutation_occurred:
                    mutation_log.append({
                        'original': parent1,
                        'mutated': child1,
                        'positions': mutation_positions
                    })
                
                new_population.append(child1)
        
        # Replace population with new population
        self.population = new_population
        
        # Add elite back if using elitism
        if self.elitism:
            # Replace a random individual with the elite
            random_idx = random.randint(0, self.population_size - 1)
            self.population[random_idx] = elite
        
        # Log generation details
        self.generations_log.append({
            'fitnesses': fitnesses,
            'avg_fitness': current_avg,
            'best_fitness': self.best_fitness,
            'best_individual': self.best_individual,
            'crossovers': crossover_log,
            'mutations': mutation_log
        })
        
        return self.best_fitness

# Main Streamlit App Logic
if 'ga_instance' not in st.session_state:
    st.session_state.ga_instance = None
    st.session_state.current_generation = 0
    st.session_state.running = False
    st.session_state.generation_results = []

# Create two columns for controls and population view
control_col, view_col = st.columns([1, 1])

with control_col:
    run_button = st.button("Run Genetic Algorithm", key="run_ga")
    reset_button = st.button("Reset", key="reset_ga")

    if run_button:
        st.session_state.running = True
        
        if problem_type == "Group 1: Digit Chromosomes":
            ga = GeneticAlgorithm(
                problem_type=problem_type,
                population_size=population_size,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                elitism=elitism
            )
        else:  # Group 2: Binary MAX-ONE
            ga = GeneticAlgorithm(
                problem_type=problem_type,
                population_size=population_size,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                elitism=elitism
            )
        
        st.session_state.ga_instance = ga
        st.session_state.current_generation = 0
        st.session_state.generation_results = []

    elif reset_button:
        st.session_state.ga_instance = None
        st.session_state.current_generation = 0
        st.session_state.running = False
        st.session_state.generation_results = []

# Run the evolution if we have a GA instance
if st.session_state.running and st.session_state.ga_instance and st.session_state.current_generation < generations:
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Create placeholders for dynamic content
    stats_container = st.container()
    
    # Evolution loop
    ga = st.session_state.ga_instance
    
    for generation in range(st.session_state.current_generation, generations):
        # Evolve the population
        best_fitness = ga.evolve()
        
        # Update progress bar
        progress_bar.progress((generation + 1) / generations)
        
        # Update generation counter
        st.session_state.current_generation = generation + 1
        
        # Store the results
        gen_results = {
            'generation': generation,
            'best_fitness': ga.best_fitness,
            'best_individual': ga.best_individual,
            'population': ga.population.copy(),
            'avg_fitness': ga.avg_fitness_history[-1],
            'log': ga.generations_log[-1]
        }
        st.session_state.generation_results.append(gen_results)
        
        # Sleep a bit to make the visualization more visible
        time.sleep(0.05)
    
    st.session_state.running = False
    st.success("Evolution complete!")

# Display results
if st.session_state.ga_instance:
    ga = st.session_state.ga_instance
    
    # Create tabs for different views
    evolution_tab, population_tab, log_tab = st.tabs(["Evolution Progress", "Population", "Evolution Log"])
    
    with evolution_tab:
        # Fitness chart
        if len(ga.best_fitness_history) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            generations_range = list(range(len(ga.best_fitness_history)))
            
            ax.plot(generations_range, ga.best_fitness_history, 'r-', label='Best Fitness')
            ax.plot(generations_range, ga.avg_fitness_history, 'b-', label='Average Fitness')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.set_title('Fitness Evolution')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            
            # Display best solution
            st.subheader("Best Solution")
            
            best_individual = ga.best_individual
            best_fitness = ga.best_fitness
            
            if problem_type == "Group 1: Digit Chromosomes":
                st.markdown(f"**Chromosome:** {''.join(map(str, best_individual))}")
                st.markdown(f"**Fitness:** {best_fitness}")
                
                a, b, c, d, e, f, g, h = best_individual
                st.markdown(f"**Calculation:** ({a}+{b})-({c}+{d})+({e}+{f})-({g}+{h}) = {best_fitness}")
                
                # Check if optimal solution was found
                if best_individual == [9, 9, 0, 0, 9, 9, 0, 0]:
                    st.success("Optimal solution found! (99009900)")
                    
            else:  # Group 2: Binary MAX-ONE
                st.markdown(f"**Chromosome:** {''.join(map(str, best_individual))}")
                st.markdown(f"**Fitness (Ones Count):** {best_fitness}")
                
                # Check if optimal solution was found
                if best_fitness == 10:
                    st.success("Optimal solution found! (All ones)")
    
    with population_tab:
        # Show the current population
        if len(st.session_state.generation_results) > 0:
            generation_slider = st.slider(
                "Generation",
                min_value=0,
                max_value=len(st.session_state.generation_results) - 1,
                value=len(st.session_state.generation_results) - 1
            )
            
            selected_gen = st.session_state.generation_results[generation_slider]
            
            st.subheader(f"Population at Generation {selected_gen['generation']}")
            
            # Calculate fitness for each individual in the population
            population_data = []
            for i, individual in enumerate(selected_gen['population']):
                fitness = ga.fitness_function(individual)
                population_data.append({
                    'Individual': f"Ind-{i+1}",
                    'Chromosome': ''.join(map(str, individual)),
                    'Fitness': fitness
                })
            
            # Create DataFrame and sort by fitness (descending)
            df = pd.DataFrame(population_data)
            df = df.sort_values('Fitness', ascending=False).reset_index(drop=True)
            
            # Highlight the best individual
            def highlight_best(row):
                if row['Fitness'] == selected_gen['best_fitness']:
                    return ['background-color: #e6f7e6'] * len(row)
                return [''] * len(row)
            
            st.dataframe(df.style.apply(highlight_best, axis=1), use_container_width=True)
    
    with log_tab:
        # Show the evolution log
        if len(st.session_state.generation_results) > 0:
            log_generation = st.slider(
                "Generation",
                min_value=0,
                max_value=len(st.session_state.generation_results) - 1,
                value=len(st.session_state.generation_results) - 1,
                key="log_generation_slider"
            )
            
            log_entry = st.session_state.generation_results[log_generation]
            
            st.subheader(f"Generation {log_entry['generation']} Log")
            
            # Display crossover operations
            if log_entry['log']['crossovers']:
                st.markdown("#### Crossover Operations")
                for i, crossover in enumerate(log_entry['log']['crossovers']):
                    with st.expander(f"Crossover {i+1}: {crossover['description']}"):
                        parent1, parent2 = crossover['parents']
                        children = crossover['children']
                        
                        st.markdown(f"**Parent 1:** {''.join(map(str, parent1))}")
                        st.markdown(f"**Parent 2:** {''.join(map(str, parent2))}")
                        
                        for j, child in enumerate(children):
                            st.markdown(f"**Child {j+1}:** {''.join(map(str, child))}")
            
            # Display mutation operations
            if log_entry['log']['mutations']:
                st.markdown("#### Mutation Operations")
                for i, mutation in enumerate(log_entry['log']['mutations']):
                    with st.expander(f"Mutation {i+1}: Positions {mutation['positions']}"):
                        original = mutation['original']
                        mutated = mutation['mutated']
                        
                        st.markdown(f"**Original:** {''.join(map(str, original))}")
                        st.markdown(f"**Mutated:** {''.join(map(str, mutated))}")
            
            # Display statistics
            st.markdown("#### Statistics")
            st.markdown(f"**Average Fitness:** {log_entry['avg_fitness']:.2f}")
            st.markdown(f"**Best Fitness:** {log_entry['best_fitness']}")
            st.markdown(f"**Best Individual:** {''.join(map(str, log_entry['best_individual']))}")

# If no GA instance, show information about genetic algorithms
else:
    st.header("About Genetic Algorithms")
    st.markdown("""
    Genetic algorithms are optimization techniques inspired by natural selection and genetics. They work by:
    
    1. **Initializing** a population of potential solutions
    2. **Evaluating** each solution using a fitness function
    3. **Selecting** the best solutions for reproduction
    4. **Crossover** - combining solutions to create new ones
    5. **Mutation** - introducing random changes
    6. **Repeating** until a satisfactory solution is found
    
    #### Main features of Genetic Algorithms:
    
    1. **Fitness Function:** Represents the main requirements of the desired solution.
    2. **Representation (Encoding):** How solutions are represented as chromosomes.
    3. **Selection Operator:** Defines how individuals are selected for reproduction.
    4. **Crossover Operator:** Defines how parent chromosomes are combined.
    5. **Mutation Operator:** Creates random changes in the genetic code.
    
    Click "Run Genetic Algorithm" to start the simulation and see the algorithm in action.
    """) 
