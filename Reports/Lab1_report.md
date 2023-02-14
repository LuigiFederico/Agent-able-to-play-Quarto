# __Laboratories__

## __Lab 1 - Set Covering__  

### __My code__  

#### Setup  

```Python
import random

def problem(N, seed=None):
    random.seed(seed)
    return [
        list(set(random.randint(0, N-1) for n in range(random.randint(N//5, N//2))))
        for n in range(random.randint(N, N*5))
    ]
```

#### Solution Proposed  
```Python
W_FACTOR = 2
INTW_FACTOR = 10

def cost(l, goal, covered):
  """ Compute the weighted number of digits left to find the goal state + the weighted len of the list"""
  extended_covered = set(l) | covered   # Covered digits including the list l
  return len(l) * W_FACTOR + len(goal - extended_covered) * INTW_FACTOR


def find_solution(probl, N):
  """ Optimized greedy """

  def print_solution(solution, n_visited):
    w = 0
    for sol_l in solution:
      w += len(sol_l)

    print("We have found a solution!!")
    print(f"Weight = {w}")
    print(f"Number of visited nodes = {n_visited}")
    print(f"\nSolution: {solution}")

  n_visited = 0
  goal = set(range(N))
  covered = set()
  solution = list()
  frontier = sorted(probl, key=lambda l: cost(l, goal, covered))

  # Max number of iterations = N (worst case: I add ONE new digit to the covered set every time)
  # Possible becouse we clear the redundant lists
  for i in range(N):
    # Goal reached?
    if goal == covered:
      print_solution(solution, n_visited)
      return 

    # Look for another list
    if frontier:
      l = frontier.pop(0)
      n_visited += 1
      
      # If l is not a subset of covered, e.i. there is a new digit
      if not set(l) < covered: 
        solution.append(l)
        covered |= set(l)
      
      # Clear the redundant lists (lists with only digits already covered)
      for x in frontier:
        if set(x) < covered:
          frontier.remove(x)
      
      frontier.sort(key=lambda l: cost(l, goal, covered))
    
    # No list available
    else:
      print("No solution has beed found")
      return 

  if goal == covered:
    print_solution(solution, n_visited)
  else:
    print("No solution has beed found")

```



### __My REDME__  
  
  
>
>  # __Lab1: Set Covering__
>
>  > Author: Luigi Federico  
>  >  
>  > Computational Intelligence 2022/23  
>  > Prof: G. Squillero
>
>  ---
>
>  ## __Task__
>
>  Given a number $N$ and some lists of integers $P = (L_0, L_1, L_2, ..., L_n)$, 
>  determine is possible $S = (L_{s_0}, L_{s_1}, L_{s_2}, ..., L_{s_n})$
>   
> such that each number between $0$ and $N-1$ appears in at least one list
>
>  $$\forall n \in [0, N-1] \ \exists i : n \in L_{s_i}$$
>
>  and that the total numbers of elements in all $L_{s_i}$ is minimum. 
>
>  ---
>
>  ## __Solution proposed__
>
>  To find a solution I did as follows:
>
>  1. I sorted the lists $L_0, L_1, L_2, ..., L_n$ using the cost function (explained below). This sorted lists are the frontier.
>  2. For a maximum of N iterations, the algorithm pops out the list with the lowest cost and adds it to the solution.
>  3.  It clears from the frontier every list that is a strict subset of the covered set, becouse it can't contribute to build a solution.
>  4. The frontier is now resorted using the cost function.
>  5. If there will be at least a not redundant list inside the frontier, it will be considered to build the solution in the next iteration. If the frontier is empty after the next loop and we didn't find the solution I declare it.
>
>  > __Why at most N iterations?__  
>  > The worst case in this approach occurs when we take at each step just one new digit into the covered set. If at some point we want to append a new list to the solution, we surely want to add the one that gives at least one new digit to the covered set. If we do more then N operations, it's garanteed that at least one of the lists is redundant, i.e. it contains only digits that was already added to the solution.
>
>  ### __Cost function__
>
>  ``` Python
>  def cost(l, goal, covered):
>    extended_covered = set(l) | covered  
>    return len(l) * W_FACTOR + len(goal - extended_covered) * INTW_FACTOR
>  ```
>
>  The idea behind this function is to consider:
>  - the length of the current list *l*  
>  - the number of digits left to achive the solution if the list *l* was included in the *covered* set.  
>
> I wanted to give more importance to the amount of contribution that the given list *l* brings to the solution (in terms of number of new digits offered to the *covered* set) rather then the weight of *l*, i.e. the length of the list.  
> To do this, I scaled the weight and the number of new digits obtained adding *l* to the solution with __W_FACTOR__=2 (weight factor) and __INTW_FACTOR__=10 (internal weight factor), respectively.  
>
>  > Note that if we use as scale factor W_FACTOR = INTW_FACTOR, the algorithm will favorite short lists over longer ones, even if the first ones contribute with fewer new digits. I found it not ideal.
>
>
>  ## __Results__
>
>  Running the algorithm with different values of N we obtain the following results.
>  > Of course they depend on the generated list set, but they should be similar to the peer rewiever results
>
>  ### __N = 5__
>  Solution found:
>  - Weight of the solution = 5  
>  - Execution time = 0.1s  
>  - Number of visited nodes = 3  
>
>  ### __N = 10__
>  Solution found:
>  - Weight of the solution = 10
>  - Execution time = 0.1s  
>  - Number of visited nodes = 3  
>
>  ### __N = 20__
>  Solution found:
>  - Weight of the solution = 28
>  - Execution time = 0.1s  
>  - Number of visited nodes = 4   
>
>  ### __N = 100__ 
>  Solution found:
>  - Weight of the solution = 181
>  - Execution time = 0.1s  
>  - Number of visited nodes = 6
>
>  ### __N = 500__ 
>  Solution found:
>  - Weight of the solution = 1419
>  - Execution time = 1.6s  
>  - Number of visited nodes = 11
>
>  ### __N = 1000__ 
>  Solution found:
>  - Weight of the solution = 3110
>  - Execution time = 6.9s  
>  - Number of visited nodes = 13
> 

---

### __Review 1__  
- Review to Andrea D'Attila (s303339)

> ## Major issue: __What if there is no valid solution?__  
>
>You didn't consider this case in your algorithm.  
>Imagine that the function generates lists in which the value `k` is never present, >given k as integer s.t. 0<= k <N. The result is that your cost function `h` will always return ```float('inf')``` once it has covered all elements except k, since ```new_elements = 0``` anyhow.  
>Your algorithm will provide a new list that will be appended to the solution until the generated problem list will be empty. __Here comes the real problem__: the ```min(*iterable*, *[, _key, default_])``` function raises a 'ValueError' exception if the provided iterable is empty and there is no default value specified. [Give a look at the function doc for more info.](https://docs.python.org/3/library/functions.html#min)  
>
>Since you don't handle this problem, __your code could end badly!__
>
>> ### Possible solution:
>> You could simply check if the cost function returns ```float('inf')```: if so, break the loop and declare that there is not a valid solution.
>
>---  
>
>## Minor issue: __Pay attention to redundancies!__  
>
>The lists `flat_sol` and `sol` exploit the same functionality, i.e. they are redundant duplicates. Morover, you do two different operation that should be done on the same and unique "solution list" but once using `flat_sol` and another using `sol`:  you compute the cost on the `flat_sol` list but you return the `sol` list. 
>
>In this case it's not a problem since it's easy to maintain allined the two lists but per sè it's a best practice to avoid duplicates when it's possibile since it could give you some problem and it could waste your time!  
>
>


#### __Code reviewed__  
This is from his jupyter notebook file

```Python
import logging
import copy

def h(sol, current_x):
    common_elements = len(set(sol) & set(current_x))
    new_elements = len(current_x) - common_elements
    if (new_elements == 0):
        return float('inf')
    return common_elements/new_elements
    
def greedy(N):
    goal = set(range(N))

    lists = sorted(problem(N, seed=42), key=lambda l: -len(l))

    starting_x = lists.pop(0)
    
    sol = list()
    sol.append(starting_x)
    
    flat_sol = list(starting_x)
    nodes = 1

    covered = set(starting_x)
    while goal != covered:
        most_promising_x = min(lists, key = lambda x: h(flat_sol, x))
        lists.remove(most_promising_x)
        
        flat_sol.extend(most_promising_x)
        sol.append(most_promising_x)
        nodes = nodes + 1

        covered |= set(most_promising_x)

    w = len(flat_sol)

    logging.info(
        f"Greedy solution for N={N}: w={w} (bloat={(w-N)/N*100:.0f}%) - visited {nodes} nodes"
    )
    logging.debug(f"{sol}")
    return sol

```

---

### __Review 2__  

- Review to Francesco Carlucci

> # Major issues  
>
>## Generators are slower then sets  
>
>In order to check if you reached the goal you use a generator like the following one to compare the elements already covered with another set: 
>
>```Python
>set([item for sublist in selected for item in sublist])
>```
>You do this:
>- inside the `goal_test(slz)` function;
>- inside the `priority_function(newState)` function;
>- inside the inner loop of the `tree_search2(...)` function.
>
>This approach is not optimal since there is a better object that could perform those operations way faster: __sets__! 
>
>You could have used a set variable `covered` to hold the set of digits already covered by your solution. To update this set with the new digits added with an expanction you could do:
>
>```Python
>covered |= newlist
>```
>
>To verify if your selection of lists is a solution, you could use this simple and optimised piece of code:
>
>```Python
>if coveder == set(range(N))
>    return True
>else:
>    return False
>```
>
>To verify if the the new list has new digits to offer to the solution, i.e. if the new list is not a strict subset of the partial solution you could use:
>
>```Python
>if not set(newlist) < covered:
>    ....
>```
>
>This way should be faster, cleaner and more elegant! 😄 
>
>## Useless check
>
>Inside the function ```tree_search2```, when you find a solution you check if the the current solution `selected` costs less then `solution`:
>
>```Python
>if slzCost( selected ) < slzCost(solution):
>    solution = sekected
>break
>```
>
>This if is useless because `solution=lists` always. That means that if you find a solution that is a strick subset of the list of lists generated by the problem (from now on "problem list") it will necessary have a lower cost computed by `slzCost()`. This check could be useful if you try to find other solutions after you have discovered this one, but you don't (because of the `break`).  
>
>> ### __Possible solution__  
>> If you mean to mantain the `break`, just don't check the costs with that function.
>> ```Python
>> if goal_test(selected):
>>     logging.info(f"Founded!")
>>    return selected
>> ```
>
># Minor issue
>
>It's better to don't import stuff that you will not use: the imports executes code, thus it will slow your script and will in vain take up space. It's just a best practice since if you import a whole library, you could import bad staff or, even worst, there could be an open script that you will run (maybe without knowing what he is doing).  
>
> Just clean up the code skeleton before the final commit and your code will be safer and more readable! 😄 
>

#### __Code Reviewed__

This is from his the jupyter notebook file

```Python

from queue import PriorityQueue

def tree_search2(lists, goal_test, priority_function):
    frontier = PriorityQueue()

    state=(set(),(), lists) #initial state
    
    n=0
    while state is not None:
        
        selected,solution,available=state
        
        if goal_test(selected):
            logging.info(f"Found a solution in {n:,} steps: {solution}")
            break
        n+=1
        
        for i,newlist in enumerate(available):
            if not set(newlist) < selected:
                
                newState=(selected | set(newlist),solution+(newlist,),available[i+1 :])
                
                frontier.put((priority_function(selected,solution+(newlist,)),newState))
        
        if frontier:
            state = frontier.get()[1]
        else:
            state = None
        
    return solution

    def goal_test_gen(N):
    def goal_test(selected):
        return selected==set(range(N))
        
    return goal_test

# ----

def priority_function(selected,solution):
    newlist=solution[-1]
    return len(set(newlist)&selected),-len(set(newlist)|selected)

def priority_dijkstra(_,solution):
    cnt = Counter()
    cnt.update(sum((e for e in solution), start=()))
    return sum(cnt[c] - 1 for c in cnt if cnt[c] > 1), -sum(cnt[c] == 1 for c in cnt)

for N in [5, 10, 20]:
    lists = sorted(problem(N, seed=42), key=lambda l: len(l))
    filteredLists=sorted(list(list(_) for _ in set(tuple(l) for l in lists)), key=lambda l:len(l))

    tuples=tuple(tuple(sublist) for sublist in filteredLists)
    
    solution=tree_search2(tuples, goal_test_gen(N), lambda a,b: priority_function(a,b))
    print(f"Solution for N={N}: w={sum(len(_) for _ in solution)} (bloat={(sum(len(_) for _ in solution)-N)/N*100:.0f}%)")
    
    solution2=tree_search2(tuples, goal_test_gen(N), lambda a,b: priority_dijkstra(a,b))
    print(f"Dijkstra Solution for N={N}: w={sum(len(_) for _ in solution2)} (bloat={(sum(len(_) for _ in solution2)-N)/N*100:.0f}%)")

# ---------
# GREEDY
# ---------

def greedy(N, all_lists):
    """Vanilla greedy algorithm"""

    goal = set(range(N))
    covered = set()
    solution = list()
    all_lists = sorted(all_lists, key=lambda l: len(l))
    while goal != covered:
        x = all_lists.pop(0)
        if not set(x) < covered:
            solution.append(x)
            covered |= set(x)
    logging.debug(f"{solution}")
    return solution
  
for N in [5, 10, 20,100,500,1000]:
    solution = greedy(N, problem(N, seed=42))
    logging.info(
        f" Greedy solution for N={N:,}: "
        + f"w={sum(len(_) for _ in solution):,} "
        + f"(bloat={(sum(len(_) for _ in solution)-N)/N*100:.0f}%)" 
    )



```