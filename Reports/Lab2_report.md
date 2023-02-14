## __Lab 2 - Set covering using a Genetic Algorithm__

### __My code__

#### Gene.py

```Python
def create_dict_genes(problem):
  id_to_genes = dict()
  id = 0

  for l in problem:
    id_to_genes[id] = Gene(id, l)
    id += 1
  
  return id_to_genes

# ----------------------------------------- #

class Gene:
  def __init__(self, id, values):
    self.id = id
    self.values = set(values)

  def __len__(self):
    return len(self.values)

  def __str__(self):
    return f'{self.values}'
  
  def display(self):
    print(f"{sorted(self.values)}")
```

#### Genome.py

```Python

from copy import deepcopy
import random
from Gene import Gene


class Genome:
  """
  Genome attributes:
  - genome: list of genes
  """

  def __init__(self, genes):
    self.genome = genes
  

  def __len__(self):
    len_ = 0
    for gene in self.genome:
      len_ += len(gene)  
    return len_


  def covered_values(self):
    covered = set()
    for gene in self.genome:
      covered |= gene.values
    
    return covered


  def cross_over(self, partner_genome, id_to_genes: dict):
    """
    returns the genome obtained from the crossover of the current genome
    with the genome of the partner individual
    """
    # randomly choose genes (choices between len(genome)//2 and len(genome))
    g_self = random.choices(self.genome,
                            k=random.randint(len(self.genome)//2, len(self.genome))) 
    g_parent = random.choices(self.genome,
                              k=random.randint(len(partner_genome)//2, len(partner_genome)))
    survivals = set([gene.id for gene in g_self + g_parent])

    return Genome([id_to_genes[g_id] for g_id in survivals])
  
  def smart_reproduction(self, partner_genome, id_to_genes: dict, N):
    # filter the duplicates ids -> exstract the candidate genes
    candidates = [id_to_genes[id_] for id_ in set([g.id for g in self.genome + partner_genome.genome])]

    stop = max(N, len(candidates))
    goal = set(range(N))
    covered = set()
    new_genome = list()

    for i in range(stop):
      best = max(candidates, key=lambda gene: (len(goal) - len(covered | gene.values), -len(gene))) # max based on how much the gene would contribute to the solution
      new_genome.append(best)
      candidates.remove(best)
      if not candidates:
        break
    return Genome(new_genome)


  def mutate(self, id_to_genes: dict):
    genome = deepcopy(self.genome)
    point = random.randint(1, len(self.genome)-1)  # point of mutation
    candidates = set(id_to_genes.keys()) - set([gene.id for gene in self.genome]) # set of genes not present in self.genome
    
    genome[point] = id_to_genes[random.choice([c for c in candidates])] # Update the genome
    return Genome(genome)

  
  def display(self):
    for gene in self.genome:
      gene.display()
```

#### Individual.py

```Python
from collections import namedtuple
from Genome import Genome

W_FACTOR = 2
INTW_FACTOR = 10

class Individual:
  """
  Individual:
  - genome = list of genes
  - covered: set of covered values between 0 and N-1
  - cost: weighted number of digits left to find the goal state + the weighted len of the list
  """
  def ideal_cost(N):
    """
    The cost is computed as follows:  
        cost = len(genome) * W_FACTOR + len(goal - self.covered) * INTW_FACTOR  
    The ideal cost will have:  
    - len(genome) = N (ideal minimum)  
    - len(goal - self.covered) = 0
    """
    return N * W_FACTOR


  def __init__(self, genome: Genome, N):
    self.genome = genome    # list of genes
    self.covered = genome.covered_values() # set
    goal = set(range(N))
    self.cost = len(genome) * W_FACTOR + len(goal - self.covered) * INTW_FACTOR


  def __len__(self):
    return len(self.genome)    


  def is_healty(self, N):
    """
      The individual is healty if it's genome contains
      all the numbers between 0 and N-1.
    """
    return len(self.covered) == N

  
  def fight(self, opponent):
    if self.cost < opponent.cost:
      return self
    else:
      return opponent
  

  #def reproduce(self, partner, id_to_genes: dict):
  #  return Individual(self.genome.cross_over(partner.genome, id_to_genes))


  def reproduce(self, partner, id_to_genes, N):
      return Individual(self.genome.smart_reproduction(partner.genome, id_to_genes, N), N)
  

  def mutate(self, id_to_genes, N):
    return Individual(self.genome.mutate(id_to_genes), N)


  def display(self):
    return self.genome.display()
```

#### lab2.ipynb

```Python
# ---
# Setup
# ---
import random
from matplotlib import pyplot as plt
from tqdm import tqdm   # pip install "git+https://github.com/tqdm/tqdm.git@devel#egg=tqdm"

from Gene import create_dict_genes
from Individual import Individual
from Genome import Genome

def problem(N, seed=None):
  random.seed(seed)
  return [
    list(set(random.randint(0, N-1) for n in range(random.randint(N//5, N//2))))
    for n in range(random.randint(N, N*5))
  ]

# ---
# Global functionalities and parameters
# ---
POPULATION_SIZE = 5
OFFSPRING_SIZE = 3
NUM_GENERATIONS = 1000

def tournament(population):
  x, y = tuple(random.choices(population, k=2))
  return x.fight(y)

def initial_population(id_to_genes, N):
  population = list()
  genes = list(id_to_genes.values())
  tot_genes = len(id_to_genes)

  for i in range(POPULATION_SIZE):
    genes = random.choices(genes, k=random.randint(1, N))
    population.append(Individual(Genome(genes), N))

  return population

def plot_gen_best(fitness_log):
  gen_best = [max(f[1] for f in fitness_log if f[0] == x) for x in range(NUM_GENERATIONS)]

  plt.figure(figsize=(15, 6))
  plt.ylabel("cost")
  plt.xlabel("generations")
  #plt.scatter([x for x, _ in fitness_log], [y for _, y in fitness_log], marker=".", label='fitness_log')
  plt.plot([x for x, _ in enumerate(gen_best)], [y for _, y in enumerate(gen_best)], label='gen_best')
  plt.legend()

def print_statistics(winner, N):
  print(f"Genetic Algorithm (N={N}):")
  print(f"\tsuccess = {len(winner.covered)*100/N}%")
  print(f"\tgenome length = {len(winner)}")
  print(f"\tlength - idael_length = {len(winner) - N}")
  print(f"Genome:")
  winner.display()

# ---
# Evolution
# ---

def genetic_algorithm(population, id_to_genes, N):
  fitness_log = [(0, i.cost) for i in population]

  for g in tqdm(range(NUM_GENERATIONS)):
    offspring = list()
    for i in range(OFFSPRING_SIZE):
      p1 = tournament(population)
      p2 = tournament(population)
      o = p1.reproduce(p2, id_to_genes, N)

      if random.random() < 0.3:
        o = o.mutate(id_to_genes, N)
        
      fitness_log.append((g+1, o.cost))
      offspring.append(o)
    population += offspring
    population = sorted(population, key=lambda i: i.cost)[:POPULATION_SIZE]

  winner = max(population, key=lambda i: i.cost)
  return winner, fitness_log

# ---
# Main
# ---

N = 5

clean_problem = set([tuple(sorted(l)) for l in problem(N, 42)])
id_to_genes = create_dict_genes(clean_problem) # Dictionaries to map genes to ids and vice versa
population = initial_population(id_to_genes, N)
winner, fitness_log = genetic_algorithm(population, id_to_genes, N)

print_statistics(winner, N)
plot_gen_best(fitness_log)

```

### __My README__


> # __Lab2 - Set covering using a Genetic Algorithm__
> 
> > Author: Luigi Federico  
> >  
> > Computational Intelligence 2022/23  
> > Prof: G. Squillero
>
>---
>
>## __Solution proposed__
>
>I implemented a genetic algorithm that operates as follows:  
>> For each generation:  
>> 1. Randomply select two parents, both through a tournament  
>> 2. Let those parents reproduce, generating the offspring  
>> 3. With a certain probability, the offspring could mutate  
>> 4. After having generated the entire offspring, they are added to the starting population and sorted baserd on the individual cost. Cost-based evolutionary selection is applied to have the population of the next generation.  
>> 
>> The best individual of the last generation will be the winner of the selection, i.e. the proposed solution
>
>
>### __Reproduction__
>
>The reproduction is "smart", i.e. it's not a random crossover but there is a selection of the genes that will compose the genome of the offspring individual.  
>
>The reproduction between two individuals works as follows:
>1. All the genes of both the genomes are grouped toghether and the duplicate are removed.
>2. The genes are iterativelly selected by looking for the best gene among the candidates. The selection is based on how much the gene would contribute to the solution and on the length of the gene.
>```Python
>best = max(candidates, key=lambda gene: (len(goal) - len(covered | gene.values), -len(gene)))
>```
>3. When there are no candidates left or it has selected a maximum of N genes, the genome of the new individual is ready.
>
>### __Mutation__
>
>The mutation concerns a randomly chosen gene of the genome that is replaced with another gene (different from the other already present inside the genome)
>
>
>---
>---
>
>## __Results__
>
>Running the algorithm with different values of N we obtain the following results.
>
>> The winner statistics are the following:
>> - Success = 100 * number_of_covered_values / N
>> - Cost = weighted number of digits left to find the goal state + the weighted len of the list
>> - Ideal_Cost = W_FACTOR * N 
>> - Genome = set of covered values
>
>---
>
>### __N=5__
>
>Winner individual: 
>
><details>
><summary>Genome (click me):</summary>
>[0]
>[2]
>[4]
>[1, 3]
></details>
>
>- Success = 100.0%
>- Genome Length = 5
>- Length - Ideal_Length = 0
>- Cost of best individual per generation:
>![Best individuals per generation](/images/plots_lab2/best_gen_5.png)
>
>---
>
>### __N=10__
>
>Winner individual: 
>
><details>
><summary>Genome (click me):</summary>
>[6]
>[0, 3]
>[6, 9]
>[2, 7, 8]
>[0, 4]
>[1, 3, 5]
></details>
>
>- Success = 100.0%
>- Genome Length = 13
>- Length - Ideal_Length = 3
>- Cost of best individual per generation:
>![Best individuals per generation](/images/plots_lab2/best_gen_10.png)
>
>---
>
>### __N=20__
>
>Winner individual: 
>
><details>
><summary>Genome (click me):</summary>
>[5, 8, 16]
>[0, 1, 2, 7]
>[4, 7, 8]
>[3, 6, 7, 13, 15]
>[3, 6, 7, 10, 14, 17]
>[2, 3, 9, 11, 12, 17, 18, 19]
></details>
>
>- Success = 100.0%
>- Genome Length = 29
>- Length - Ideal_Length = 9
>
>- Cost of best individual per generation:
>![Best individuals per generation](/images/plots_lab2/best_gen_20.png)
>
>---
>
>### __N=100__
>
>Winner individual: 
>
>- Success = 100.0%
>- Genome Length = 261
>- Length - Ideal_Length = 161
>
>- Cost of best individual per generation:
>![Best individuals per generation](/images/plots_lab2/best_gen_100.png)
>
>---
>
>### __N=500__
>
>- Success = 100.0%
>- Genome Length = 11757
>- Length - Ideal_Length = 11257
>- Cost of best individual per generation:
>![Best individuals per generation](/images/plots_lab2/best_gen_500.png)
>
>---
>
>### __N=1000__
>
>- Success = 100.0%
>- Genome Length = 24505
>- Length - Ideal_Length = 23505
>- Cost of best individual per generation:
>![Best individuals per generation](/images/plots_lab2/best_gen_1000.png)
>
>---
>


---  

### __Review 1__
- Review to Francesco Scalera   

>
>The algorithm appear well thought and with good strategies, so: __well done!__ 👏😃  
>The following "issues" are just minor suggestions or small possible improvements.
>
>## __The algorithm seems to be slow__
>
>Your algorithm could be very slow if `N` is really big because of the fact that you want only feasible solutions inside your population. This constrain of "only feasible solutions" represents the bottleneck. The thing is that you could loop for a really high amount of time, also if you have an individual that needs to tweak correctly just one list!  
>
>You could see this behavior inside the algorithm when, after creating the offspring, you check if this is feasible. If it's not,  you discard it. The thing is that you loop until you have `OFFSPRING_SIZE` new individuals and if your mutations/crossover is unlucky you could discard a lot of individuals before finding a good population. This could be very time consuming.  
>
>My suggestion is to add a mechanism that pick a new list in a smart way, maybe if your loop didn't generate any good individual after _n_ attempts.  
>
>> By the way, I'm not very sure about how well it could improve or if it is really a performance issue... It should be tested to understand how much is probable to find a feasible solution instead of an unfeasible one, just to understand if this approach is performing well or it could be way more faster.  
>
>## __Better use of sets__
>
>Inside the function `check_feasible(•)` you use two nested loops to extract the single numbers covered by the individual.  
>
>```Python
>def check_feasibile(individual, N):
>    '''From np array of Lists and size of problem, returns if it provides a possible solution '''
>    goal = set(list(range(N)))
>    coverage = set()
>    for list_ in individual:
>        for num in list_:
>            coverage.add(num)
>        if coverage == goal:
>            return True
>    return False
>```  
>
>You could use the optimization that comes with sets to achieve the same result, but in a more fast and elegant way. You could use the `|=` operator to compute the intersection between two sets just like this:
>
>```Python
>for list_ in individual:
>    coverage |= set(list_)
>    if coverage == goal:
>        return True
>return False
>```
>
>## __The readability could be improved__
>
>### __Different names for the same object__
>There is some confusion between the names of some data structures that affect the readability.   
>In your algorithm you use a list called `mutation_probability_list`. This suggests that it will contain a list of mutation probabilities but when you pass to the function `calculate_mutation_probabilityDet2(•)` you alias this list as `best_candidate_list`. This name seems more appropriate since the structure actually contains only the individual with the same best fitness.  
>
>My suggestion is to maintain parallel the names if the function is exactly the same, just to avoid redundancies.
>
>Another thing about this list: maybe it could be replaced with a counter and a variable. This should improve readability since your use of this list is to count how many times your best fitness repeats. So, why don't just use a counter to count? 😉  
>
>### __Not ideal use of the Jupyter Notebook format__  
>
>The jupyter notebook format is a really powerful tool to divide code in blocks, inserting some text to better structure the code. Your [lab2.ipynb](https://github.com/francescoscalera99/CI_2022_292432/blob/main/lab2/lab2.ipynb) file looks like a .py file more than a .ipynb file!  
>
>My suggestion is to isolate all the single pieces that is logically self-complete.
>- You could separate in different blocks all the functions, giving a title to it if needed.  
>- You could isolate the part with all the parameters, so if you want to change/add just one parameter you don't have to run again all the code since you can run just that part alone!  
>
>Another cool thing that you could do is to save the output of your code inside the .ipynb file, so who wants to read your code could see the output without running it.
>  

#### __Code reviewed__
This is from his jupyter notebook file

```Python
from base64 import decode
import random 
import numpy as np
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import time
# import sys

# Function for the problem 
def problem(N, seed=42):
    """Generates the problem, also makes all blocks generated unique"""
    random.seed(seed)
    blocks_not_unique = [
        list(set(random.randint(0, N - 1) for n in range(random.randint(N // 5, N // 2))))
        for n in range(random.randint(N, N * 5))
    ]
    blocks_unique = np.unique(np.array(blocks_not_unique, dtype=object))
    return blocks_unique.tolist()

def check_feasibile(individual, N):
    '''From np array of Lists and size of problem, returns if it provides a possible solution '''
    goal = set(list(range(N)))
    coverage = set()
    for list_ in individual:
        for num in list_:
            coverage.add(num)
        if coverage == goal:
            return True
    return False

def createFitness(individual):
    fitness = 0
    for list_ in individual:
        fitness += len(list_)
    return fitness

def select_parent(population, tournament_size = 2):
    subset = random.choices(population, k = tournament_size)
    return min(subset, key=lambda i: i [0])

def cross_over(g1,g2, len_):
    cut = random.randint(0,len_-1)
    return g1[:cut] + g2[cut:]
# cross_over con più tagli

def mutation(g, len_):
    point = random.randint(0,len_-1)
    return g[:point] + [not g[point]] + g[point+1:]

def calculate_mutation_probability(best_candidate, N):
    distance = abs(N - best_candidate[0])
    return 1-(distance/N)

best_candidate_option = ""

def calculate_mutation_probabilityDet2(best_candidate, N, best_candidate_list):
    global best_candidate_option

    probability_selected = 0.5
    probability_reason = ""

    # check if best changed (based on fitness func)
    if not best_candidate[0] == best_candidate_list[-1][0]:
        best_candidate_list.clear()
        best_candidate_list.append(best_candidate)
    else:
        best_candidate_list.append(best_candidate)

    # if list is bigger than 10 select opositive of current best
    if len(best_candidate_list) > 10:

        if len(best_candidate_list) < 21:
            if best_candidate[2] == "mutation":
                probability_reason= "cross"
                probability_selected = 0.1
            else:
                probability_reason= "mutation"
                probability_selected = 0.9
        else:
            probability_reason = best_candidate_option

        if len(best_candidate_list) % 20 == 0:
            if best_candidate_option == "mutation":
                probability_reason= "cross"
                probability_selected = 0.1
            else:
                probability_reason= "mutation"
                probability_selected = 0.9
    else:
        probability_reason = "distance-based"
        probability_selected = calculate_mutation_probability(best_candidate, N)

    best_candidate_option = probability_reason
    return probability_selected

PARAMETERS = {
    "N":[20, 100, 500, 1000, 5000],
    "POPULATION_SIZE":[50, 200, 300, 500, 600, 1000, 2000, 3000, 5000],
    "OFFSPRING_SIZE":[int(50*2/3), int(200*2/3), int(300*2/3), int(500*2/3), int(600*2/3), int(1000*2/3), int(2000*2/3), int(3000*2/3), 5000*(2/3)]
    # number of iterations? as 1000 is too small for some N values
}

configurations = {"configurations": []}
my_configs = ParameterGrid(PARAMETERS)
for config in my_configs:
    configurations["configurations"].append(config)

#Inital list of lists
random.seed(42)

with open("results.csv", "a") as csvf:
    header="N,POPULATION_SIZE,OFFSPRING_SIZE,fitness\n"
    csvf.write(header)

    for idx in tqdm(range(len(configurations["configurations"]))):

        config = configurations["configurations"][idx]

        start = time.time()

        initial_formulation = problem(config['N'])
        initial_formulation_np = np.array(initial_formulation, dtype=object)

        mutation_probability_list = list()
        mutation_probability_list.append((None, None, ""))
        population = list()

        # we use a while since if the checks will give always false, i can also have a population that too little in size
        while len(population) != (config['POPULATION_SIZE']):
            # list of random indexes
            # this avoid duplicate samples of the same index when initializing the first individuals
            random_choices = random.choices([True, False], k=len(initial_formulation))
            # np array of lists based on random indexes
            individual_lists = initial_formulation_np[random_choices]
            if check_feasibile(individual_lists,config['N']) == True:
                population.append((createFitness(individual_lists), random_choices, ""))

        for _ in range(1000):
            # print(f"interation {_}; w:{population[0][0]}; best calculated:{population[0][2]}")
            sum_of_cross = 0
            sum_of_mut = 0
            offspring_pool = list()
            offspring_pool_mask = list()
            i = 0
            mutation_probability = calculate_mutation_probabilityDet2(population[0], config['N'], mutation_probability_list)
            while len(offspring_pool) != config['OFFSPRING_SIZE']:
                reason = ""
                if random.random() < mutation_probability:
                    p = select_parent(population)
                    sum_of_mut += 1
                    offspring_mask = mutation(p[1], len(initial_formulation))
                    offspring_mask = mutation(offspring_mask, len(initial_formulation))
                    reason = "mutation"
                else:
                    p1 = select_parent(population)
                    p2 = select_parent(population)
                    sum_of_cross += 1
                    offspring_mask = cross_over(p1[1],p2[1], len(initial_formulation))
                    reason = "cross"
                
                offspring_lists = initial_formulation_np[offspring_mask]
                if check_feasibile(offspring_lists, config['N']) == True and offspring_mask not in offspring_pool_mask:
                    offspring_pool.append((createFitness(offspring_lists), offspring_mask, reason))
                    offspring_pool_mask.append(offspring_mask)

            population = population + offspring_pool
            unique_population = list()
            unique_population_mask = list()
            for ind in population:
                if ind[1] not in unique_population_mask:
                    unique_population.append(ind)
                    unique_population_mask.append(ind[1])
            unique_population=list(unique_population)
            unique_population.sort(key=lambda x: x[0])
            # take the fittest individual
            population = unique_population[:config['POPULATION_SIZE']]

        end = time.time()
        csvf.write(f"{config['N']},{config['POPULATION_SIZE']},{config['OFFSPRING_SIZE']},{population[0][0]},{end-start}\n")
```

---  

### __Review 2__
- Review to Leonor Gomes

>
> # __Major issues__  
>
>## __Crossover__
>
>After splitting the parents, you generate the children by gluing the splitted genomes __without checking if the offspring contains duplicate lists__.  
>This affects badly your fitness function because the length of the genome will increase: 
>```Python  
>fitness = N * size_genome /unique_values
>```
>This could be a problem if the fitness becomes worse just because of the noisy duplicates, also if the solution was a really good one if it had not the clones.  
>
>### __Exepli Gratia__:  
>> N = 5
>> Individual with noise:  
>> --> [0] [1, 2, 3] [3, 4] __[1, 2, 3]__ 
>> --> fitness = 5 * 8 / 5 = 8  
>> Individual without noise: 
>> --> [0] [1, 2, 3] [3, 4]
>> --> fitness =  5 * 5 / 5 = 5
>> Worst individual but with the same fitness of the noisy good individual:
>> --> [0, 1] [1, 2, 3] [2, 3, 4]
>> --> fitness = 5 * 8 / 5 = 8  
>>
>> The quality of these two individuals is identical for your algorithm!  
>
>### Possible solutions:  
>You could choose different strategies. Here are some:
>1. If the list is already selected, delete the list from the individual.
>> This could be seen as a mutation.  
>> - Pro: this kind of mutation could generate offspring that could lead you out of a local optimum. This could be useful after a lot of generations, when the individual are quite all the same, favoring exploitation.  
>> - Con: If that gene was a really good one, the risk is to affect badly the individual fitness.  
>> 
>
>2. If the list is already selected, just don't insert it again!
>> The most basic approach!
>> - Pro: easy to implement and correct the noise problem described above.
>> - Con: If you use this approach with the mutation (read the next point), you just waste the mutation!
>
>
>## __Mutation__
>
>When you select a random list of the genome to perform the mutation, you pick the new list from the entire pool of lists with the possibility of pick a noisy duplicate that affect badly the fitness of the individual. __It's just the same problem as before!__  
>
>## __Genetic Algorithm__
>
>If you obtain 10 times the same best fitness, you mutate the entire population.  
>Issue: you don't keep track of the absolute best solution that your algorithm generated so far. What if it generates the global optimum solution and the fitness will be the same for the next generations? You will discard it, losing the best solution! And you have no guarantee that you will find it again!  
>
>### Possible solution:  
>You could keep a variable that contains the best individual so far and update it every generation. In this way, not only you will always have a trace of the best individual generated by your algorithm, but if you change the whole population, you don't forget that individual! This is good if the mutations are worsening the entire individual population.
>
>
>
># __Minor issues__
>
>## __Generation of the next population__
>It might be more optimal to merge the current population with the offspring and to perform a single sort to let survive the best solutions among the merged population.
>What if the entire offspring population just generated have a very bad fitness if compared with the parent population? With your way of generating the next population, you are discarding part of the population with better fitness than the offspring!  
>> Let's notice that those "weak" individual will die  in the next generation anyway.  
>
>__Is it worth sacrificing potentially better solutions?__
>If your intention was to also retain individuals with potentially pejorative fitness than the previous generation average, then it can be an interesting strategy hoping that a crossover in the next generation will lead to a better individual.  
>
>We should test and understand which of the two approaches is better: always select the absolute best or favor individuals potentially worse than others?
>
>
>## __Poorly readable README__
>If you want to paste the genome you could use the following trick to make it collapsable inside a markdown file.  
>
>Just wrap your genome like this:  
>```
><details>
>  <summary> Click me to display the genome </summary>
>
>  genome lists
>
></details>
>```
>
>Feel free to look at [my lab2 README](https://github.com/LuigiFederico/Computational-Intelligence/blob/main/lab2/README.md) to see how it works! :)
>
>This should make your README more readable and more interactive!  
>
>


#### __Code reviewed__
This is from her jupyter notebook file

```Python
import logging
import random
from copy import copy
from collections import namedtuple
from operator import attrgetter


def problem(N, seed=None):
    """Creates an instance of the problem"""

    random.seed(seed)
    return [
        list(set(random.randint(0, N - 1) for n in range(random.randint(N // 5, N // 2))))
        for n in range(random.randint(N, N * 5))
    ]

Individual = namedtuple('Individual', ['genome', 'fitness'])
TOURNAMENT_SIZE = 5
POPULATION_SIZE = 30


#part of code from https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists

def unique(sol): # number of unique values in a possible solution/individual
   unique = set([item for sublist in sol for item in sublist])
   return len(unique)

#unique values
#size of solution
#we want to minimize fitness
def fitness(genome, N):
  fit = 0
  size = len([item for sublist in genome for item in sublist]) 

  unique_values = unique(genome)
  fit = N*(size/unique_values) #relation between current size of the list and the amount of unique values
  values_left = N - unique_values

  if (values_left > 0):
    fit = fit + N*values_left #adds values left

  return fit 

#mutates a random gene(list) and substitutes it with a possible list from the problem 
def mutation(individual, P, N):
  index = random.randrange(len(individual.genome))
  old_gene = individual.genome.pop(index)

  P_index = random.randrange(len(P))
  new_gene = P[P_index]

  new_genome = individual.genome + [new_gene]
  new_fitness = fitness(new_genome, N)

  new_individual = Individual(new_genome, new_fitness)

  return new_individual

# takes a random interval and generates two children, mixed from two parents
def crossover(first_individual, second_individual, N):
  min_size = min(len(first_individual.genome), len(second_individual.genome))

  interval = random.randrange(min_size) #interval can't be bigger than one of the individuals

  first_child_genome = first_individual.genome[:interval] + second_individual.genome[interval:]
  second_child_genome = second_individual.genome[:interval] + first_individual.genome[interval:]

  first_child = Individual(first_child_genome, fitness(first_child_genome, N))
  second_child = Individual(second_child_genome, fitness(second_child_genome, N))


  return first_child, second_child

# generates a random individual from the problem lists
def generate_individual(P):

  individual_size = random.randrange(1, len(P))

  individual = random.sample(P, individual_size) #gets a random sized sample of lists from the problem 

  return individual
     
# generates random initial population 
def generate_population(P, N):
  population = []

  for i in range(POPULATION_SIZE):
    new_individual = generate_individual(P)

    if new_individual not in population: #checks if the individual is already in the population
      population.append(Individual(new_individual, fitness(new_individual, N)))
  
  return population


# tournament selection
def select_parent_tournament(population): 
  tournament_selection = []

  while len(tournament_selection) != TOURNAMENT_SIZE: #randomly select a small subset of the population to compete against each other
    id = random.randrange(len(population))

    tournament_selection.append(population[id])

  tournament_selection.sort(key=attrgetter('fitness')) 

  return tournament_selection[0] #returns the fittest from the subset of the tournament


def should_mutate(): #given a small possibility, should we mutate or not?
  if random.random() < 0.3:
    return True
  return False


#generates new offspring 

def create_offspring(population, P, N):
  #generate POPULATION_SIZE offspring
  
  offspring = []
  
  for i in range(int(POPULATION_SIZE/2)): #for each iteration -> 2 new children
    tournament_parent = select_parent_tournament(population) #selects first parent from tournament

    random_id = random.randrange(POPULATION_SIZE) 
    random_parent = population[random_id] #selects second parent randomly

    first_child, second_child = crossover(tournament_parent, random_parent, N) #gets two new children from crossover

    if should_mutate():  #given a small possibility -> mutate new child
      first_child = mutation(first_child, P, N)
    if should_mutate(): #given a small possibility -> mutate new child
      second_child = mutation(second_child, P, N)

    offspring.append(first_child) #add new child to offspring
    offspring.append(second_child) #add new child to offspring

  return offspring


#sorts the current population and the new offspring - gets the best half from the current population and the best half from the offspring
def get_new_population(population, offspring):
  new_population = []

  population.sort(key=attrgetter('fitness')) 
  offspring.sort(key=attrgetter('fitness'))

  best_fitness = min(population[0].fitness, offspring[0].fitness)

  new_population = population[:int(POPULATION_SIZE/2)] + offspring[:int(POPULATION_SIZE/2)]
  return new_population, best_fitness


def print_fitness(population): #debug function to print the fitness of a population
  for i in range(len(population)):
    print(population[i].fitness)


def escape_local_optimum(population, P, N): #mutate the entire population in the hopes of escaping local optimum
  new_population = []

  for i in range(len(population)):
    new_individual = mutation(population[i], P, N)
    new_population.append(new_individual)

  return new_population


#steady state

def genetic_algorithm(P, N, generations = 100):
 
  #first generate a population
  population = generate_population(P, N)

  population.sort(key=attrgetter('fitness')) 

  current_best_fitness = population[0].fitness #var to check if the solution is improving
  counter = 0

  #termination criteria - number of desired generations
  for i in range(generations):

    offspring = create_offspring(population, P, N) #creates offspring
    population, new_best_fitness = get_new_population(population, offspring) #gets the new population with half from the fittest parents and the other half with the fittest children
    
    if (new_best_fitness == current_best_fitness):
      counter += 1
    else:
      counter = 0
      current_best_fitnes = new_best_fitness

    if (counter == 10):
      population = escape_local_optimum(population, P, N) #if the solution hasn't improved for 10 generations, we try to escape local optimum

  population.sort(key=attrgetter('fitness')) #sorts the population by fitness

  return population[0] #returns the individual with best fitness


def get_results(N, number_generations = 100):
  p = problem(N, 42)
  solution = genetic_algorithm(p, N, number_generations)
  print(f'LEN: {sum([len(l) for l in solution.genome])}')
  return solution

```

---
