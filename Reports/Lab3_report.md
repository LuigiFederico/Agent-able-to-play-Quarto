# __Lab 3 - Policy Search__

### __My code__

#### nim.py

```Python

# Nim class created by professor Giovanni Squillero:
#    Copyright **`(c)`** 2022 Giovanni Squillero `<squillero@polito.it>`  
#    [`https://github.com/squillero/computational-intelligence`](https://github.com/squillero/computational-intelligence)  
#    Free for personal or classroom use; see [`LICENSE.md`](https://github.com/squillero/computational-intelligence/blob/master/LICENSE.md) for details.  

from collections import namedtuple

Nimply = namedtuple("Nimply", "row, num_objects")

class Nim:
    def __init__(self, num_rows: int, k: int = None) -> None:
        self._rows = [i * 2 + 1 for i in range(num_rows)]
        self._k = k

    def __bool__(self):
        return sum(self._rows) > 0

    def __str__(self):
        return "<" + " ".join(str(_) for _ in self._rows) + ">"

    @property
    def rows(self) -> tuple:
        return tuple(self._rows)

    def nimming(self, ply: Nimply) -> None:
        row, num_objects = ply
        assert self._rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        self._rows[row] -= num_objects

    def is_game_over(self):
        # This method is made by me
        return sum(self._rows) == 0

```

#### play_nim.py

```Python

import random
import logging
from copy import deepcopy
from nim import Nimply, Nim


def nim_sum(elem: list):
  x = 0
  for e in elem:
    x = e ^ x
  return x

##################
##  STRATEGIES  ##
##################

# Level 0: Easy
def dumb_action(nim: Nim):
  """
  Always takes one obj from the first row available
  """
  row = [r for r, n in enumerate(nim._rows) if n > 0][0]
  return Nimply(row, 1)


# Level 1: Medium
def dumb_random_action(nim: Nim):
  """
  There is 0.5 of probability to make a dumb action or a random action
  """
  if random.random() < 0.5:
    return dumb_action(nim)
  else: return random_action(nim)


# Level 2: Medium-Advanced
def random_action(nim: Nim):
  """
    The agent performs a random action
  """
  row = random.choice([r for r, n in enumerate(nim._rows) if n > 0])
  if nim._k:
    num_obj = random.randint(1, min(nim._k, nim._rows[row]))
  else:
    num_obj = random.randint(1, nim._rows[row])
  
  return Nimply(row, num_obj)


# Level 3: Medium-Advanced
def layered_action(nim: Nim):
  """
  Always takes the whole row's objs choosing randomly the row
  """
  row, num_obj = random.choice([(r, n) for r, n in enumerate(nim._rows) if n > 0])
  if nim._k and num_obj > nim._k:
    return Nimply(row, nim._k)
  return Nimply(row, num_obj)


# Level 4: DEMIGOD 
def demigod_action(nim: Nim, prob_god=0.5):
  """
    There is a probability prob to play an expert move and a chance to play randomly
  """
  if random.random() < prob_god:
    return expert_action(nim)
  else: return random_action(nim)

  
# Level 5: GOD 
def expert_action(nim: Nim):
  """
    The agent uses fixed rules based on nim-sum (expert-system)
    Returns the index of the pile and the number of pieces removed as a Nimply namedtuple
  """
  board = nim._rows
  k = nim._k

  # Winning move if there is only one row left
  tmp = [(i, r) for i, r in enumerate(board) if r > 0]
  if len(tmp) == 1:
    row, num_obj = tmp[0]
    if not k or num_obj <= k:
      return Nimply(row, num_obj) # Take the entire row


  # Compute the nim-sum of all the heap sizes
  x = nim_sum(board)

  if x > 0:
    # Current player on a insucure position -> is winning
    # --> Has to generate a secure position (bad for the other player)
    # --> Find a heap where the nim-sum of X and the heap-size is less than the heap-size.
    # --> Then play on that heap, reducing the heap to the nim-sum of its original size with X
    
    good_rows = [] # A list is needed because of k
    for row, row_size in enumerate(board):
      if row_size == 0:
        continue
      ns = row_size ^ x # nim sum
      if ns < row_size:
        good_rows.append((row, row_size)) # This row will have nim sum = 0
        
    for row, row_size in good_rows:
      board_tmp = deepcopy(board)
      for i in range(row_size):
       board_tmp[row] -= 1 
       if nim_sum(board_tmp) == 0:  # winning move
        num_obj = abs(board[row] - board_tmp[row])
        if not k or num_obj <= k:
          return Nimply(row, num_obj)
  
  # x == 0 or k force a bad move to the player
  # Current player on a secure position or on a bad position bc of k -> is losing
  # --> Can only generate an insicure position (good for the other player)
  # --> Perform a random action bc it doesn't matter
  return random_action(nim)
  

opponents = {
  1: dumb_action,
  2: dumb_random_action,
  3: random_action,
  4: layered_action,
  5: demigod_action,
  6: expert_action
}



####################
##  PLAY MATCHES  ##
####################

def evaluate(nim: Nim, n_matches=20, *, my_action, opponent_action=random_action, debug=False):
  """
    This function let you evaluate how many matches your strategy wins against an opponent.  
    You are player 0.  
    Input:
      - nim: Nim
      - n_matches=20
      - my_action
      - opponent_action=random_action
      - debug=False # let's you display all the match moves for each match
    Output:
      - Percentage of won matches (number of wins / number of matches)
  """
  
  if debug:
    logging.getLogger().setLevel(logging.DEBUG)

  player_action = {
    0: my_action, # our champion
    1: opponent_action # our opponent
    }

  won = 0

  for m in range(n_matches):
    # Setup match
    nim_tmp = deepcopy(nim)
    if m/n_matches > 0.5:
      player = 1  # You start
    else:
      player = 0  # Opponent starts

    logging.debug(f'Board -> {nim_tmp}\tk = {nim_tmp._k}')
    logging.debug(f'Player {1-player} starts\n')
    
    # Play the match
    while not sum(nim_tmp._rows) == 0:
      player = 1 - player
      ply = player_action[player](nim_tmp)
      #logging.debug(f'Action P{player} = {ply}')
      nim_tmp.nimming(ply)
      logging.debug(f'player {player} -> {nim_tmp}\tnim_sum = {nim_sum(nim_tmp._rows)}')

    logging.debug(f'\n### Player {player} won ###\n')
    if player == 0:
      won += 1
      
  return won/n_matches


```


#### lab3_task1.ipynb

```Python
# ---
# Task 3.1 - An agent using fixed rules based on nim-sum
# Based on the explanation available here: https://en.wikipedia.org/wiki/Nim
#
# It wants to finish every move with a nim-sum of 0, called 'secure position' (then it will win if it does not make mistakes).
# ---

import logging
from copy import deepcopy
from nim import Nimply, Nim
from play_nim import nim_sum, random_action, evaluate

logging.basicConfig(format="%(message)s", level=logging.INFO)

# ---
# Implementation
# ---

def expert_action(nim: Nim):
  """
    The agent uses fixed rules based on nim-sum (expert-system)

    Returns the index of the pile and the number of pieces removed as a Nimply namedtuple
  """
  board = nim._rows
  k = nim._k

  # Winning move if there is only one row left
  tmp = [(i, r) for i, r in enumerate(board) if r > 0]
  if len(tmp) == 1:
    row, num_obj = tmp[0]
    if not k or num_obj <= k:
      return Nimply(row, num_obj) # Take the entire row


  # Compute the nim-sum of all the heap sizes
  x = nim_sum(board)

  if x > 0:
    # Current player on a insucure position -> is winning
    # --> Has to generate a secure position (bad for the other player)
    # --> Find a heap where the nim-sum of X and the heap-size is less than the heap-size.
    # --> Then play on that heap, reducing the heap to the nim-sum of its original size with X
    
    good_rows = [] # A list is needed because of k
    for row, row_size in enumerate(board):
      if row_size == 0:
        continue
      ns = row_size ^ x # nim sum
      if ns < row_size:
        good_rows.append((row, row_size)) # This row will have nim sum = 0
        
    for row, row_size in good_rows:
      board_tmp = deepcopy(board)
      for i in range(row_size):
       board_tmp[row] -= 1 
       if nim_sum(board_tmp) == 0:  # winning move
        num_obj = abs(board[row] - board_tmp[row])
        if not k or num_obj <= k:
          return Nimply(row, num_obj)
  
  # x == 0 or k force a bad move to the player
  # Current player on a secure position or on a bad position bc of k -> is losing
  # --> Can only generate an insicure position (good for the other player)
  # --> Perform a random action bc it doesn't matter
  return random_action(nim)

# ---
# Play
# ---
nim = Nim(7)
evaluate(nim, 20, my_action=expert_action)

```

#### lab3_task2.ipynb

```Python

# ---
# Task 3.2 - An agent using evolved rules
# Here for semplicity, I will consider the parameter k always equal to None
# ---

import logging
import random
from tqdm import tqdm
from matplotlib import pyplot as plt

from nim import Nimply, Nim
from play_nim import *
# inside play_nim: 
#   Functions:  nim_sum, dumb_action, dumb_random_action,
#               random_action, layered_action, demigod_action,
#               expert_action, evaluate
#   Dictionary: opponents

logging.basicConfig(format="%(message)s", level=logging.INFO)


# ---
# Implementation
# ---

# Individual
class EvolvedPlayer():
  """
    This played uses GA to evolve some rules
    that lets him play the game (hopefully better every time).  

    The genome will be a list of rules that will be applyed 
    in order. The information lies inside the order of the rules, that 
    can change with genetic operations (XOVER and MUT).
  """

  def __init__(self, nim: Nim, genome=None):
    self._k = nim._k
    self.score = -1
    self.__collect_info(nim) # Cooked info
    self.rules = self.__rules()
    if genome:
      assert len(genome) == len(self.rules)
      self.genome = genome
    else: 
      self.genome = list(self.rules.keys())
      random.shuffle(self.genome) 


  def __collect_info(self, nim: Nim):
    """
      Collects some info:
      - number of not zero rows
      - sorted rows by number of objects
      - average number of objects per row
    """
    self.n_rows_left = len([r for r in nim._rows if r > 0])
    self.sorted_rows = sorted([(r, n) for r, n in enumerate(nim._rows) if n > 0], key=lambda r: -r[1])
    self.avg_obj_per_row = sum(nim._rows) / len(nim._rows)    


  def __rules(self):
    """
    Returns a set of fixed rules as a dicitonary.  
    The dictionary will be as follows:
      - key: id as incremental number
      - value: tuple with (condition, action), where 
        - condition = boolean condition that has to be true in order to perform the action
        - action = Nimply action 
    """

    assert self.n_rows_left
    assert self.sorted_rows
    assert self.avg_obj_per_row
    
    ### Conditions and Actions ###

    # 1 row left --> take the entire row
    def c1(self): 
      return self.n_rows_left == 1
    def a1(self):
      return Nimply(self.sorted_rows[0][0], self.sorted_rows[0][1])

    # 2 rows left --> leave the same number of objs
    def c2(self):
      return self.n_rows_left == 2 and self.sorted_rows[0][1] != self.sorted_rows[1][1]
    def a2(self):
      num_obj = self.sorted_rows[0][1] - self.sorted_rows[1][1]
      return Nimply(self.sorted_rows[0][0], num_obj)

    # 2 rows left and len longest row > avg --> leave one obj in the higher row
    def c3(self):
      return self.n_rows_left == 2 and self.sorted_rows[0][1] > self.avg_obj_per_row and self.sorted_rows[0][1]>1
    def a3(self):
      return Nimply(self.sorted_rows[0][0], self.sorted_rows[0][1] - 1)

    # 3 rows left --> take the entire max row
    def c4(self):
      return self.n_rows_left == 3
    def a4(self):
      return Nimply(self.sorted_rows[0][0], self.sorted_rows[0][1])

    # 3 rows left --> take leave the longest row with one obj
    def c5(self):
      return self.n_rows_left == 3 and self.sorted_rows[0][1] > 1
    def a5(self):
      return Nimply(self.sorted_rows[0][0], self.sorted_rows[0][1] - 1)
    
    # avg < max+1 --> take the longest row
    def c6(self):
      return self.avg_obj_per_row < self.sorted_rows[0][1] + 1
    def a6(self):
      return Nimply(self.sorted_rows[0][0], self.sorted_rows[0][1])

    # default -> take one obj from the longest row
    def c7(self):
      return True
    def a7(self):
      return Nimply(self.sorted_rows[0][0], 1)


    ### Rule dictionary ###
    dict_rules = {
      1: (c1, a1),
      2: (c2, a2),
      3: (c3, a3),
      4: (c4, a4),
      5: (c5, a5),
      6: (c6, a6),
      7: (c7, a7)
    }
  
    return dict_rules


  def ply(self, nim: Nim):
    """
      Check the rules in order. The first rule that matches will be applyed
    """
    # Update the informations
    self.__collect_info(nim)

    # Apply a rule
    for key in self.genome:
      cond, act = self.rules[key]
      if cond(self):
        logging.debug(f'--> RULE number {key} applyed')
        return act(self)


  def set_score(self, score):
    self.score = score


  def clear_score(self):
    self.score = -1


  def cross_over(self, partner, nim):
    """ 
      Cycle crossover: choose two loci l1 and l2 (not included) and copy the segment
      between them from p1 to p2, then copy the remaining unused values
    """
    locus1 = random.randint(0, len(self.genome)-1)
    while (locus2 := random.randint(0, len(self.genome)-1)) == locus1:
      pass
    if locus1 > locus2:
      tmp = locus1
      locus1 = locus2
      locus2 = tmp

    # Segment extraction
    segment_partner = partner.genome[locus1:locus2] # slice
    alleles_left = [a for a in self.genome if a not in segment_partner]
    #random.shuffle(alleles_left)

    # Create the offspring genome
    piece1 = alleles_left[:locus1]
    piece2 = alleles_left[locus1:]
    offspring_genome = piece1 + segment_partner + piece2
    
    return EvolvedPlayer(nim, offspring_genome)


  def mutation(self):
    """
      Swap mutation: alleles in two random loci are swapped
    """
    locus1 = random.randint(0, len(self.genome)-1)
    while (locus2 := random.randint(0, len(self.genome)-1)) == locus1:
      pass
    
    # Swap mutation
    tmp = self.genome[locus1]
    self.genome[locus1] = self.genome[locus2]
    self.genome[locus2] = tmp


# Evolution
def initial_population(population_size: int, nim: Nim):
  population = []
  for i in range(population_size):
    population.append(EvolvedPlayer(nim))
  
  return population


def tournament(population, tournament_size=2):
  return max(random.choices(population, k=tournament_size), key=lambda i: -i.score)


def island(nim, population, generations=1, *, opponent, selective_pressure, mut_prob, matches_1v1, evolve=True, display_matches=False, display_survivals=False):

  for gen in range(generations):
    # Play
    for player in population:
      won_p = evaluate(nim, 
                        n_matches=matches_1v1, 
                        my_action=player.ply,
                        opponent_action=opponent)
      player.set_score(won_p)
      if display_matches:
        logging.info(f'Player {player.genome} has a win rate={player.score}')

    # Evolution
    
    best_population = [i for i in population if i.score > selective_pressure]
    offspring_size = len(population) - len(best_population)
    
    if display_survivals:
      logging.info(f'- Survivals = {len(best_population)}')

    if evolve:
      for i in range(offspring_size):
        p1 = tournament(best_population)
        p2 = tournament(best_population)
        o = p1.cross_over(p2, nim)       # XOVER
        if random.random() < mut_prob:   # MUT
          p1.mutation()
        best_population.append(o)
      population = best_population    

    return population


def genetic_algorithm(nim: Nim, population, *, generations=100, matches_island=5, matches_1v1=20, display_survivals=False):
  """
    The algorithm is based on islands.  
    Each island have a different opponent to match with increasing difficulty.  
    It goes from the dumb strategy to the god strategy (expert agent).  
    There will be matches_island matches on each island and the genetic operations 
    will be applyed after the end of the competition. The best individuals (the ones 
    that won more matches) will pass to the next generation, while the others will perish.
    The offsprings will replace the missing individuals.
  """
  # I imported the opponents and the evaluation function from play_nim.py 
  log_best1 = []
  log_best2 = []
  log_best3 = []
  log_best4 = []
  log_best5 = []
  selective_pressure = 0.5
  mut_prob = 0.01

  for i in tqdm(range(generations)):

    ####  ISLAND 1: population vs dumb agent  ####
    if display_survivals:
      logging.info(f'ISLAND 1: population vs dumb agent')

    survivals1 = island(nim, population, matches_island,
                          opponent=opponents[1],
                          selective_pressure=0.7,
                          mut_prob=mut_prob,
                          matches_1v1=matches_1v1,
                          display_survivals=display_survivals)
    log_best1.append((i, max([p for p in survivals1 if p.score>0], key=lambda i: -i.score)))


    ####  ISLAND 2: population vs dumb random agent  ####
    if display_survivals:
      logging.info(f'ISLAND 2: population vs dumb random agent')

    survivals2 = island(nim, survivals1,  matches_island,
                          opponent=opponents[2],
                          selective_pressure=selective_pressure,
                          mut_prob=mut_prob,
                          matches_1v1=matches_1v1,
                          display_survivals=display_survivals)  
    log_best2.append((i, max([p for p in survivals2 if p.score>0], key=lambda i: -i.score)))

    ####  ISLAND 3: population vs dumb random agent  ####
    if display_survivals:
      logging.info(f'ISLAND 3: population vs random agent')

    survivals3 = island(nim, survivals2, matches_island,
                          opponent=opponents[3],
                          selective_pressure=selective_pressure,
                          mut_prob=mut_prob,
                          matches_1v1=matches_1v1,
                          display_survivals=display_survivals)
    log_best3.append((i, max([p for p in survivals3 if p.score>0], key=lambda i: -i.score)))  
    
    ####  ISLAND 4: population vs layered agent  ####
    if display_survivals:
      logging.info(f'ISLAND 4: population vs layered agent')

    survivals4 = island(nim, survivals3, matches_island,
                          opponent=opponents[4],
                          selective_pressure=selective_pressure,
                          mut_prob=mut_prob,
                          matches_1v1=matches_1v1,
                          display_survivals=display_survivals)  
    log_best4.append((i, max([p for p in survivals4 if p.score>0], key=lambda i: -i.score)))  

    ####  ISLAND 5: population vs demigod agent  ####
    if display_survivals:
      logging.info(f'ISLAND 5: population vs demigod agent')

    survivals5 = island(nim, survivals4, matches_island,
                          opponent=opponents[5],
                          selective_pressure=selective_pressure,
                          mut_prob=mut_prob,
                          matches_1v1=matches_1v1,
                          display_survivals=display_survivals) 
    log_best5.append((i, max([p for p in survivals5 if p.score>0], key=lambda i: -i.score)))

    population = survivals5
   
  ####  ISLAND 6: population vs god agent  ####
  logging.info(f'ISLAND 6: population vs god agent')
  survivals6 = island(nim, survivals5,
                        opponent=opponents[6],
                        selective_pressure=0,
                        mut_prob=0,
                        matches_1v1=matches_1v1,
                        display_matches=False,
                        display_survivals=True,
                        evolve=False) 
  defeated_god = [p for p in survivals6 if p.score > 0]
  if defeated_god:
    for champion in defeated_god:
      logging.info(f'CHAMPION {champion.genome} defeated GOD (score={champion.score})')
  else:
    logging.info(f'God won\n')


  log_best_generation = (log_best1, log_best2, log_best3, log_best4, log_best5)
  return population, log_best_generation


# ---
# Play
# ---

POPULATION_SIZE = 500

nim = Nim(7)
population = initial_population(POPULATION_SIZE, nim)

survivals, log_best = genetic_algorithm(nim, population,
                                          matches_island=50,
                                          matches_1v1=10,
                                          display_survivals=False)

parent_survivals = [p for p in survivals if p.score > 0]
logging.info('Survivals:')
for i in range(len(parent_survivals)):
  logging.info(f'- Player {parent_survivals[i].genome} with score {parent_survivals[i].score}')

log1, log2, log3, log4, log5 = log_best
bests = []
bests.append(max(log1, key=lambda x: x[1].score))
bests.append(max(log2, key=lambda x: x[1].score))
bests.append(max(log3, key=lambda x: x[1].score))
bests.append(max(log4, key=lambda x: x[1].score))
bests.append(max(log5, key=lambda x: x[1].score))

logging.info('Best players:')
for i in range(5):
  logging.info(f'Island {i} - player {bests[i][1].genome} with score {bests[i][1].score} at generation {bests[i][0]}')

```

#### lab3_task3.ipynb

```Python

# ---
# Task3.3: An agent using minmax
# ---

import logging
import random
from copy import deepcopy
from nim import Nimply, Nim

logging.basicConfig(format="%(message)s", level=logging.INFO)

# ---
# Implementation
# ---

def hash_id(state: list, player: int):
  """
    Computes the hash of the tuple tuple(state) + (player, ), where:
    - state is the list of rows, i.e. the board
    - player is either 0 or 1
  """
  assert player == 1 or player == 0
  return hash(tuple(sorted(state)) + (player, ))

# ---

class Node():
  """
    State of the grapth that contains:
    - id: hash of tuple(state)+(player,)
    - state: copy of the state (nim._rows)
    - player: either 0 or 1
    - utility: value initialized to 0, becomes either -inf or +inf
    - children: list of nodes
    - parents: list of nodes 
    - actions: list of possible actions as Nimply objects
  """

  def __init__(self, state: list, player: int):
    assert player == 1 or player == 0
    
    self.id = hash_id(state, player)
    self.state = deepcopy(state)
    self.player = player # Me (0) -> max ; Opponent (1) -> min
    
    self.utility = 0  # -inf if I lose, +inf if I win
    self.children = []
    self.parents = []
    self.possible_acitions() # creates self.actions


  def __eq__(self, other):
    return isinstance(other, Node) and self.state == other.state and self.player == other.player


  def link_parent(self, parent):
    """
      Links the actual node with the parent node
    """
    assert isinstance(parent, Node)
    assert self.player != parent.player

    if parent not in self.parents:
      self.parents.append(parent)


  def link_child(self, child):
    """
      Links the child node to the actual node
    """
    assert isinstance(child, Node)
    assert self.player != child.player

    if child not in self.children:
      self.children.append(child)


  def is_leaf(self):
    return sum(self.state) == 0

  
  def leaf_utility(self):
    """
      Returns the utility of a leaf:
      - player 0 on leaf --> I lost, then utility = -inf
      - player 1 on leaf --> I won, then utility = +inf 
    """
    if self.is_leaf():
      if self.player == 0: 
        return float('-inf')     # I lost (the opponent took the last piece) 
      else: return float('+inf') # I won


  def possible_acitions(self, k=None):
    """
      Computes all the possible action reachable from the actual node
      and saves them inside self.actions 
    """
    self.actions = []
    
    if self.is_leaf():
      return

    not_zero_rows = [(r, n) for r, n in enumerate(self.state) if n > 0]
    for row, num_obj in not_zero_rows:  
      while num_obj > 0:
        if k and num_obj > k:
          num_obj = k
          continue
        self.actions.append(Nimply(row, num_obj))
        num_obj -= 1

# ---

class GameTree():
  """
    Game Tree comosed of nodes that could have multiple parents and multiple children.  
    
    The roots is one:
    - Starting state + starting player = 0 
    The leafs are two:
    - State of all zeros + finish player = 0  (I lose)
    - State of all zeros + finish player = 1  (I win)
    
    The class contains the following attributs:
    - k: nim._k
    - start_player: either 0 or 1
    - dict_id_node: dictionary that maps the node id to the actual node
    - dict_id_utility_action: dictionary that maps the node id to a tuple (utility, action), where:
      - utility: utility of the node
      - action: better action to take (Nimply object)
    - root: root node (Node object)    
  """

  def __init__(self, nim: Nim, start_player=0):
    self.k = nim._k
    self.start_player = start_player
    self.dict_id_node = {}    
    self.dict_id_utility_action = {} 
    
    self.root = Node(nim._rows, start_player)
    self.dict_id_node[self.root.id] = self.root


  def min_max(self):
    """
      MinMax using a recursive function that expands a node by trying every possible action of that node.  

      The recursive function returns the utility of the children and the parent will select  
       the best utility according to who is playing at that layer:
      - if player 1 is playing, than minimize the reward (look for utility = -inf)
      - if player 0 is playing, than maximize the reward (look for utility = +inf)

      The alpha-beta pruning is implemented:  
       if the player finds a child with the desired utility, it stops looking
      becouse he will win choosing that action to go to that state.
    """

    def recursive_min_max(node: Node):  
      # Stop condition
      if node.id in self.dict_id_utility_action:
        logging.debug(f'State {node.state} ({node.player}) already computed: {self.dict_id_utility_action[node.id][0]}')
        return self.dict_id_utility_action[node.id][0] # just the utility value
      
      if node.is_leaf():
        node.utility = node.leaf_utility()
        logging.debug(f'Leaf player {node.player}')
        return node.utility


      # Recursive part
      for ply in node.actions:
        row, num_obj = ply
        
        # Check rules
        assert node.state[row] >= num_obj
        assert self.k is None or num_obj <= self.k

        # Create the child
        child_state = deepcopy(node.state)
        child_state[row] -= num_obj # nimming
        child_id = hash_id(child_state, 1 - node.player)
        if child_id in self.dict_id_node: # node already exists
          child = self.dict_id_node[child_id]
        else: # create the new node
          child = Node(child_state, 1 - node.player)
        
        # Link parent and child
        node.link_child(child)
        child.link_parent(node)

        # Recursion
        best_utility = recursive_min_max(child)
        
        # Update the values
        opp_wins = node.player == 1 and best_utility == float('-inf')  # opponent will win
        i_win = node.player == 0 and best_utility == float('+inf')  # I will win
        if i_win or opp_wins:
          node.utility = best_utility
          self.dict_id_utility_action[node.id] = (node.utility, ply)
          return node.utility
          
      # This player will surelly lose otherwise he would have returned before
      node.utility = best_utility
      ply = random.choice(node.actions) # it doesn't matter the ply, he will lose
      self.dict_id_utility_action[node.id] = (node.utility, ply)
    
      return node.utility
    
    
    utility = recursive_min_max(self.root)
    if self.start_player == 0 and utility == float('+inf'):
      logging.info('The starting player will WIN')
      logging.info(f'--> move {self.dict_id_utility_action[self.root.id][1]}')
      return self.dict_id_utility_action[self.root.id]
    else:
      logging.info('The starting player will LOSE')
      return self.dict_id_utility_action[self.root.id] 
    

  def best_action(self, node: Node):
    """
      Returns the best aciton at that state
    """
    assert self.root.id in self.dict_id_utility_action
    assert node.id in self.dict_id_utility_action

    return self.dict_id_utility_action[node.id]

# ---
# Play
# ---

nim = Nim(5)
game_tree0 = GameTree(nim, start_player=0) # I start
game_tree1 = GameTree(nim, start_player=1) # Opponent starts

game_tree0.min_max()
game_tree1.min_max()

```

#### lab3_task4.ipynb

```Python

# ---
# Task3.4: An agent using reinforcement learning
# ---

import logging
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from nim import Nimply, Nim
from play_nim import opponents, evaluate

logging.basicConfig(format="%(message)s", level=logging.INFO)

# ---
# Implementation 
# ---

def hash_id(state: list, player: int):
  """
    Computes the hash of the tuple tuple(state) + (player, ), where:
    - state is the list of rows, i.e. the board
    - player is either 0 or 1
  """
  assert player == 1 or player == 0
  return hash(tuple(state) + (player, ))


# ---
# Node Class from Task 3
# ---

class Node():
  """
    State of the grapth that contains:
    - id: hash of tuple(state)+(player,)
    - state: copy of the state (nim._rows)
    - player: either 0 or 1
    - reward: value initialized to 0, becomes either 2 (win) or -2 (lose)
    - children: list of nodes
    - parents: list of nodes 
    - actions: list of possible actions as Nimply objects
  """

  def __init__(self, state: list, player: int):
    assert player == 1 or player == 0
    
    self.id = hash_id(state, player)
    self.state = deepcopy(state)
    self.player = player # Me (0) -> max ; Opponent (1) -> min
    
    self.reward = self.give_reward() 
    self.children = []
    self.parents = []
    self.possible_acitions() # creates self.actions


  def __eq__(self, other):
    return isinstance(other, Node) and self.state == other.state and self.player == other.player


  def link_parent(self, parent):
    """
      Links the actual node with the parent node
    """
    assert isinstance(parent, Node)
    assert self.player != parent.player

    if parent not in self.parents:
      self.parents.append(parent)


  def link_child(self, child):
    """
      Links the child node to the actual node
    """
    assert isinstance(child, Node)
    assert self.player != child.player

    if child not in self.children:
      self.children.append(child)


  def is_game_over(self):
    return sum(self.state) == 0

  
  def give_reward(self):
    """
    Returns the reward of the node
    - not end -> reward = -1  
    - win -> reward = 2
    - lose -> reward = -2  
    """
    if not self.is_game_over():
      #return -1
      return random.uniform(-1, 1)
    if self.player == 0: # I lose
      return -2
    return 2 # I win


  def possible_acitions(self, k=None):
    """
      Computes all the possible action reachable from the actual node
      and saves them inside self.actions 
    """
    self.actions = []
    
    if self.is_game_over():
      return

    not_zero_rows = [(r, n) for r, n in enumerate(self.state) if n > 0]
    for row, num_obj in not_zero_rows:  
      while num_obj > 0:
        if k and num_obj > k:
          num_obj = k
          continue
        self.actions.append(Nimply(row, num_obj))
        num_obj -= 1

# ---
# Game Tree (builded recursively, such us in task 3)
# ---

class GameTree():
  """
    Game Tree comosed of nodes that could have multiple parents and multiple children.
    Differently from task 3, this class expands the tree considering both player 1 and player 0 starting.
    
    The roots are two:
    - Starting state + starting player = 0 
    - Starting state + starting player = 1 
    The leafs are two:
    - State of all zeros + finish player = 0  (Agent loses)
    - State of all zeros + finish player = 1  (Agent wins)
    
    The class contains the following attributs:
    - k: nim._k
    - dict_id_node: dictionary that maps the node id to the actual node
    - dict_id_reward: dictionary that maps the node id to the state reward
    - root0: root node (Node object) when player 0 starts
    - root1: root node (Node object) when player 1 starts  

  """

  def __init__(self, nim: Nim):
    self.k = nim._k
    self.dict_id_node = {}   
    self.dict_id_reward = {} 

    # Build tree
    self.root0 = Node(nim._rows, player=0)
    self.root1 = Node(nim._rows, player=1)
    self.dict_id_node[self.root0.id] = self.root0
    self.dict_id_node[self.root1.id] = self.root1

    logging.info(f'Building the tree...')
    self.build_tree(self.root0)
    self.build_tree(self.root1)
    logging.info('Done')

  
  def build_tree(self, root):
    """
      Builds the tree using a recursive function that expands a node by trying every possible action of that node.

      The nodes are likend to each other, starting by the given root node.
    """
    
    def recursive(node: Node):
      # Stop condition
      if node.id in self.dict_id_reward:
        return
      
      if node.is_game_over():
        node.reward = node.give_reward()
        self.dict_id_reward[node.id] = node.reward
        return


      # Recursive part
      for ply in node.actions:
        row, num_obj = ply
        
        # Check rules
        assert node.state[row] >= num_obj
        assert self.k is None or num_obj <= self.k

        # Create the child
        child_state = deepcopy(node.state)
        child_state[row] -= num_obj # nimming
        child_id = hash_id(child_state, 1 - node.player)
        if child_id in self.dict_id_node: # node already exists
          child = self.dict_id_node[child_id]
        else: # create the new node
          child = Node(child_state, 1 - node.player)
          self.dict_id_node[child_id] = child
        
        # Link parent and child
        node.link_child(child)
        child.link_parent(node)

        # Recursion
        recursive(child)
          
      # Reward of the node (-1)
      node.reward = node.give_reward()
      self.dict_id_reward[node.id] = node.reward
    
      return 

    recursive(root)
    root.reward = root.give_reward()

# ---
# Agent
# ---

class Agent():
  """
    The agent that will use Reinforcement Learning to learn to play nim.  
    This class in based on the maze example given by the professor.

    Attributes:
    alpha: learning rate
    random_factor: probability of making a random action
    state_history: history of the match played before learning
    G: dictionary that maps the id node to the expected reward (initialized randomly)
  """
  
  def __init__(self, game_tree: GameTree, alpha=0.5, random_factor=0.2):
    self.alpha = alpha
    self.random_factor = random_factor
    
    self.state_history = [] # node -> inside has state and reward
    self.G = {} # (k, v) = id_node, expected reward
    for id, node in game_tree.dict_id_node.items():
        self.G[id] = random.uniform(1.0, 0.1)


  def choose_action(self, node: Node):
    """
      Returns a Nimply by choosing the move that gives the maximum reward.
      With self.random_factor probability returns a random move. 
    """
    maxG = -10e15
    next_move = None
    
    # Random action
    if random.random() < self.random_factor:
      next_move = random.choice(node.actions)
    # Action with highest G (reward)
    else: 
      for a in node.actions:  # a is a Nimply obj
        new_state = deepcopy(node.state)
        new_state[a.row] -= a.num_objects
        new_state_id = hash_id(new_state, player=1) # opponent's state
        if self.G[new_state_id] >= maxG:
          next_move = a
          maxG = self.G[new_state_id]

    return next_move      
        

  def update_history(self, node: Node):
    self.state_history.append(node)


  def learn(self):
    """
      Update the internal G function by looking at the past
       using the formula G[s] = G[s] + a * (v - G[s]), where:
       - G[s] is the expected reward for the state s
       - a is alpha, the learning rate
       - v is the actual value associated to that state

      After the learning part, it reset the history and decreases the random factor by 10e-5
    """
    target = 0

    for node in reversed(self.state_history):
      self.G[node.id] = self.G[node.id] + self.alpha * (target - self.G[node.id])
      target += node.reward
      #print(f'player {node.player}: {node.state} - {self.G[node.id]}')

    self.state_history = []     # Restart
    self.random_factor -= 10e-5 # Decrease random factor each episode of play

# ---
# Evaluatation function
# ---

def play(nim: Nim, game_tree: GameTree, agent: Agent, n_matches=40, *, opponent_action: callable, alternate_turns=True):
  """
    This function simulated n_matches games between the agent and the given opponent.

    If alternate_turns == True, than the games will have as strating player the player 0 
     50 % of the time and player 1 50% of the time.
    
    If alternate_turns == False, the player 0 will always start the match.
  """

  # Agent is player 0
  won = 0
  for m in range(n_matches):
    # Setup the match
    nim_tmp = deepcopy(nim)

    if alternate_turns:
      if m/n_matches > 0.5:
        player = 1 # The agent starts
      else:
        player = 0 # Opponent starts  
    else:
      player = 0 # The agent always starts
      
    # Play the match
    while not nim_tmp.is_game_over():
      player = 1 - player
      if player == 1:
        ply = opponent_action(nim_tmp)
      else: # player 0
        state_id = hash_id(nim_tmp._rows, player)
        node = game_tree.dict_id_node[state_id]
        ply = agent.choose_action(node)
      nim_tmp.nimming(ply)

    if player == 0:
      won += 1
  
  return won/n_matches

# ---
# Reinforcement Learning algorithm
# ---

def RL_nim(nim: Nim, game_tree: GameTree, agent: Agent, opponent: callable, episodes = 5000, alternate_turns=True):
  """
    A match is played by the agent aganst the given opponent episodes times.

    At each episode, the agent learns looking at the rewards that it receved.

    Every 50 epochs, the agent plays versus the opponent using the function play(•) above, 
    in order to look how many matches it wins.

    The win rates are sotred inside a log list that is returned
  
  """
  log_winrate = [] # episode, value 

  for e in range(episodes):
    # Play a game
    episode_nim = deepcopy(nim)

    if alternate_turns:
      if e % 2 == 0:
        state = game_tree.root0 # the agent starts
      else:
        state = game_tree.root1 # the opponent starts
    else:
      state = game_tree.root0 # The agent always starts

    agent.update_history(state)
    while not episode_nim.is_game_over():
      # My turn
      if state.player == 0:
        my_action = agent.choose_action(state) # Choose an action
        episode_nim.nimming(my_action)         # Update the state
        new_state_id = hash_id(episode_nim._rows, player = 1)
        state = game_tree.dict_id_node[new_state_id]
        
      # Opponent turn
      else:
        opp_action = opponent(episode_nim)
        episode_nim.nimming(opp_action)
        new_state_id = hash_id(episode_nim._rows, player = 0)
        state = game_tree.dict_id_node[new_state_id]
      
      agent.update_history(state)
    
    if state.player == 0:
      agent.update_history(state)  
    agent.learn()

    # Log
    if e % 50 == 0:
      winrate = play(nim, game_tree, agent, opponent_action=opponent, alternate_turns=alternate_turns)
      logging.debug(f'{e}: winrate = {winrate}')
      log_winrate.append((e, winrate))
    
  return log_winrate

# ---
# Plot funcion
# ---

def plot_agent_winrates(log_winrate):
  x = [e for e, w in log_winrate]
  y = [w for e, w in log_winrate]
  plt.xlabel('Episodes')
  plt.ylabel('Win rate')
  plt.plot(x, y)
  plt.show()

# ---
# Play
# ---

nim = Nim(5)
game_tree= GameTree(nim)
agent = Agent(game_tree)

# ---
# Agent vs Dumb player
# ---
opponent = opponents[1]
log_lv1 = RL_nim(nim, game_tree, agent, opponent=opponent, episodes=10000, alternate_turns=False)

logging.info(f'Agent winrate from {log_lv1[0]} to {log_lv1[len(log_lv1)-1]} ')
vals = [val for _, val in log_lv1] 
logging.info(f'Avg score = {sum(vals) / len(vals)}')
plot_agent_winrates(log_lv1)

# Agent winrate from (0, 0.825) to (9950, 1.0)
# Avg score = 0.993125

# ---
# Agent vs Dumb random player
# ---
opponent = opponents[2]
log_lv2 = RL_nim(nim, game_tree, agent, opponent=opponent, episodes=10000, alternate_turns=False)

logging.info(f'Agent winrate from {log_lv2[0]} to {log_lv2[len(log_lv2)-1]} ')
vals = [val for _, val in log_lv2] 
logging.info(f'Avg score = {sum(vals) / len(vals)}')
plot_agent_winrates(log_lv2)

# Agent winrate from (0, 0.9) to (9950, 1.0)
# Avg score = 0.9764999999999988

# --- 
# Agent vs Random player
# ---

opponent = opponents[3]
log_lv3 = RL_nim(nim, game_tree, agent, opponent=opponent, episodes=10000, alternate_turns=False)

logging.info(f'Agent winrate from {log_lv3[0]} to {log_lv3[len(log_lv3)-1]} ')
vals = [val for _, val in log_lv3] 
logging.info(f'Avg score = {sum(vals) / len(vals)}')
plot_agent_winrates(log_lv3)

# Agent winrate from (0, 0.95) to (9950, 0.975) 
# Avg score = 0.9664999999999988


# ---
# Agent vs Layered player
# ---
opponent = opponents[4]
log_lv4 = RL_nim(nim, game_tree, agent, opponent=opponent, episodes=10000, alternate_turns=False)

logging.info(f'Agent winrate from {log_lv4[0]} to {log_lv4[len(log_lv4)-1]} ')
vals = [val for _, val in log_lv4] 
logging.info(f'Avg score = {sum(vals) / len(vals)}')
plot_agent_winrates(log_lv4)

# Agent winrate from (0, 1.0) to (9950, 1.0) 
# Avg score = 0.9996250000000001

# ---
# Agent vs Demigod player (50% random - 50% num-sum)
# ---

opponent = opponents[5]
log_lv5 = RL_nim(nim, game_tree, agent, opponent=opponent, episodes=10000, alternate_turns=False)

logging.info(f'Agent winrate from {log_lv5[0]} to {log_lv5[len(log_lv5)-1]} ')
vals = [val for _, val in log_lv5] 
logging.info(f'Avg score = {sum(vals) / len(vals)}')
plot_agent_winrates(log_lv5)

# Agent winrate from (0, 0.775) to (9950, 0.825) 
# Avg score = 0.7732500000000002

# ---
# Agent vs God player (nim-sum)
# ---

opponent = opponents[6]
log_lv6 = RL_nim(nim, game_tree, agent, opponent=opponent, episodes=10000, alternate_turns=False)

logging.info(f'Agent winrate from {log_lv6[0]} to {log_lv6[len(log_lv6)-1]} ')
vals = [val for _, val in log_lv6] 
logging.info(f'Avg score = {sum(vals) / len(vals)}')
plot_agent_winrates(log_lv6)

# Agent winrate from (0, 0.0) to (9950, 0.0) 
# Avg score = 0.0

```



### __My README__

>
># Lab 3: Policy Search
>
>## Task
>
>Write agents able to play [*Nim*](https://en.wikipedia.org/wiki/Nim), with an arbitrary number of rows and an upper bound $k$ on the number of objects that can be removed in a turn (a.k.a., *subtraction game*).
>
>The player **taking the last object wins**.
>
>* Task3.1: An agent using fixed rules based on *nim-sum* (i.e., an *expert system*)
>* Task3.2: An agent using evolved rules
>* Task3.3: An agent using minmax
>* Task3.4: An agent using reinforcement learning
>
>## Instructions
>
>* Create the directory `lab3` inside the course repo 
>* Put a `README.md` and your solution (all the files, code and auxiliary data if needed)
>
>## Notes
>
>* Working in group is not only allowed, but recommended (see: [Ubuntu](https://en.wikipedia.org/wiki/Ubuntu_philosophy) and [Cooperative Learning](https://files.eric.ed.gov/fulltext/EJ1096789.pdf)). Collaborations must be explicitly declared in the `README.md`.
>* [Yanking](https://www.emacswiki.org/emacs/KillingAndYanking) from the internet is allowed, but sources must be explicitly declared in the `README.md`.
>
>**Deadline**
>
>* Tasks 3.1 and 3.2 -->  4/12  
>* Tasks 3.3 and 3.4 --> 11/12  
>
>---
>
>## __Note__  
>
>__The documentation is inside every function. The readme gives just the high level idea of the strategy, while the functions contains an in depth documentation of what they do!__
>
>---
>
>### Collaborations:  
>- Paolo Drago Leon
>
>--- 
>
>## Task 3.1 - An agent using fixed rules based on nim-sum  
>
>The agent uses an expert strategy based on nim-sum. If the agent is on a position with not zero nim-sum, then he will always win.  
>
>Based on the explanation available here: https://en.wikipedia.org/wiki/Nim   
>
>
>## Task 3.2 - An agent using evolved rules
>
>The agent uses evolved hard-coded rules. Those rules are labeled with a number from 1 to 7. The order of the rules represents the genome of the agent.  
>
>I used a hyerarcical island strategy to let evolve the initial population. Only the best individual surviving at each island where able to reproduce.  
>
>The opponents where the following, ordered from the lowest island to the highest:
>- Dumb opponent: always takes one obj from the first row available.  
>- Dumb random opponent: there is 0.5 of probability to make a dumb action or a random action.  
>- Random opponent: the opponent performs a random action.  
>- Layered opponent: it always takes the whole row's objs choosing randomly the row.  
>- Demigod opponent: there is a probability to play an expert move and a chance to play randomly
>- God opponent: it is the expert agent developed at task3.1, the one that uses nim-sum.
>
>
>## Task 3.3 - An agent using minmax   
>
>The agent uses the MinMax strategy applyed on the game tree, a tree with all the possible states of nim.
>
>I used the alpha-best pruning in order to reduce the time complexity of the function. The nodes of the tree stops expanding if there is at least a child already expanded that has the best result expected for the player at that layer.
>
>> If the layer is played by the opponent and a child has -inf utility, than the opponent will stop expanding the current state because it knows that it will win with that move.
>> The same if for the layer played by the agent: if there is at least a child with +inf utility, than stop expanding the subtree.
>
>
>## Task 3.4 - An agent using reinforcement learning  
>
>The solution is inspired by the maze example given by the professor.
>
>The states are encoded inside a game tree similar to the one of the previous task and the reward is set to:
>- 100 if the agent wins
>- -100 if the agent loses
>- -1 if the state is not a win or lose state
>
>
---  

## __Review 1__
- Review to Paolo Drago Leon

>
>Overall you did a splendid job! So __well done__! 👏 
>
># Task 1 - Expert agent  
>
>### If a row has more than k objects, you force a move without looking at nim-sum  
>
>At the beginning of the expert_strategy(•) you verify if there is a row with a number of objects grater than k. If so, than you check:
>- `if  board.rows[i] % (board.k + 1) != 0`, you choose `Nimply(i, board.rows[i] % (board.k + 1))` as ply.
>- if `board.rows[i] % (board.k + 1)  == 0`, you just pick k objects from the maximum row.
>
> The problem is that __you actually choose a move that is not optimal!__ Look at this example:  
>
>> k = 3 
>> After some moves we have: 
>> `[0, 3, 5, 0, 0]` --> nim-sum = 6
>> Row 2 has num_objects > k, so you apply the first part of the strategy:
>> ``` Python
>> if board.rows[i] % (board.k + 1) != 0:
>>     ply = Nimply(i, board.rows[i] % board.k + 1))
>> ```
>> That means:
>> ``` Python
>> if 5 % (3 + 1) != 0: # True bc 5 % 4 = 1
>>     ply = Nimply(2, 1)
>> ```
>> So your next state will be:  
>> `[0, 3, 4, 0, 0]` --> nim-sum = 7 !!!
>> 
>> Instead, the best move would have been this:
>> `Nimply(2, 2)` -> `[0, 3, 3, 0, 0]` --> nim-sum=0  
>
>So the thing is your expert agent will do a not optimal move becouse it doesn't check nim-sum first.
>
>The same happens if you find a row with num_objects > k but `board.rows[i] % (board.k + 1)` is always False, than you apply a 'default' rule: `Nimply(max_row, board.k)
>
>Here is an example:
>> k = 4
>> After some point we have: `[0, 3, 5, 0, 0 ]` --> nim-sum=6
>> `board.rows[i] % (board.k + 1)` = 5 % (4+1) = 0 --> False  
>> but `one_gt_k is True` (you found a row with num_objs > k)
>>  
>> Than you apply the 'defaul' rule, so your ply will be `Nimply(2, 4)` because k=4
>> The next state will be `[0, 3, 1, 0, 0]` --> nim-sum = 2 !!!
>> 
>> Using the nim-sum approach you would have played `Nimply(2, 2)`
>> Next state -> `[0, 3, 3, 0 , 0]` --> nim-sum = 0
>
>You could verify that the limit on k is respected within the nim-sum algorithm and if you don't find any better move, than apply this strategy.
>
>### Redundant actions that the strategy would have covered  
>
>In the second part of the algorithm, after this line 
>```Python
>board_nimsum = calc_nimsum(board.rows)
>```
>you handle the cases where the board has (or you can force the board to have) at most one object at each row.  
>
>This is not an issue per se, __it is just redundant__ becouse the nim-sum algorithm would have covered this case by making you choose the best move (the same one you choose here).  
>
>The code would be less loaded and the computational complexity would not change because you iterate over the entire array in the `all_ones()` function and with the `functools.reduce()` function.
>
>
>---
>
># Task 2 - Evolved agent
>
>The strategy is well thought out and from the results I have to say that it works better than mine. I haven't found any problems in the code, except for two little things
>
>### SemiExpert B agent  
>
>Here you don't take into account the ending strategies but the nim-sum algorithm covers that case so __this 'semi-expert' agent is actually an expert agent!__
>
>### Rule 5 inside class Rules  
>
>This rule does nothing becouse of that `return None` before the condition check. I think it was intentional becouse you were using the expert strategy (nim-sum) as rule inside the individual, but leaving that will steale clock cycles. Not a big issue, but cleaning it could save some time (maybe).
>
>---
>
># Task 3 - MinMax  
>
>Nothing to say about this task, it seems to be by the book!
>
>---
>
># Task 4 - Reinforcement Learning  
>
>Here again it is a good work! your plots have really nice shapes and the results ara really good. Good job again!
> 
>

NOTE: The cose is really long and verbose and since my review should be quite detailed and self-explanatory, I don't report Paolo's code hoping it won't be a problem.

--- 

## __Review 2__
- Review to Flavio Patti

>   
> # Task 1 - Expert player using nim-sum strategy  
>
>Here you have not implemented what was asked, but instead you implemented an hard coded rule-based agent.  
>
>The thing is that __this agent will always lose against an expert agent that uses the nim-sum strategy__ becouse your code just looks for a winning move only if you are in an final state that your rules can handle.    
>
>Furthermore, your agent is very computational expencive since it's a mix between a recursive function, that goes on until it finds a good move, and 10 entire iterations over the state board.  
>
>The nim-sum strategy only does 2 iterations over the state board and will always force the opponent in a state where he can't win; moreover it is easy to implement and to read.  
>
># Task 2 - Evolved agent  
>
>You kinda implemented the optimal strategy inside task3.2.ipynb but trying all the possible states without using the real strategy to look for the only row that can give you nim-sum = 0. Again, following the algorithm that you can find online, you can have a more performant agent.  
>
>Your evolved agent uses just 4 hard-coded rules that are very generic. Without other actions, your agent will always perform poorly. You could have used some of the hard-coded rules that you implemented in task3.1. Maybe you could have added a gamma variable in order to use 9 hard-coded rules. This could be beneficial for your agent.  
>
>Maybe it was intetional, but the ifs inside the function that chooses which ply the agent will perform are not mutually exclusive according to alpha and beta.  
>
>```Python  
>def evolvable(state: Nim, genome: tuple):
>    threshold_alpha = 0.5
>    threshold_beta = 0.5
>
>    #choose the strategy to use based on the parameters inside the genome
>    if threshold_alpha <= genome[0] and threshold_beta <= genome[1]:
>        ply = dumb_PCI_max_longest(state)
>    if threshold_alpha <= genome[0] and threshold_beta >= genome[1]:
>        ply = dump_PCI_min_longest(state)
>    if threshold_alpha >= genome[0] and threshold_beta <= genome[1]:
>        ply = dump_PCI_max_lowest(state)
>    if threshold_alpha >= genome[0] and threshold_beta >= genome[1]:
>        ply = dumb_PCI_min_lowest(state)
>
>    return ply
>```
>
>If you have alpha and/or beta equal to 0.5, than the first rules will always be replaced by the ones below. If it was not willful, be aware that this could change the expected behaviour of you agent.  
>
>
># Task 3 - MinMax agent  
>
>You adopt a 'deep pruning' combined with alpha-beta pruning in order to cut off the search and reduce the computational complexity.  
>
>An approach that you did not considered was to not consider superimposable states. Here is an example:  
>
>> `[1, 0, 3, 1,  2]`
>> The above state will have the same outcome of the following states:
>> - `[1, 0, 3, 2, 1]`
>> - `[3, 1, 2, 0 , 1]`
>> - `[2, 1, 3, 0, 1]`
>> - etc.. with every possible combination of those rows  
>> 
>> All the states can be represented by a single state that has a sorted number of objects: `[3, 2, 1, 1, 0]`.  
>
>So, the idea is to map the actual state with his 'ordered version' in and take the result inside a dictionary. Doing so, you can avoid using the deep pruning and you can expand the tree to actually verify if the agent wins or not. With this strategy I runend MinMax on Nim with 8 rows and it took only 10 min (without deep pruning).  
>
>
># Task 4 - LR agent  
>
>Nothing to say here, it looks a good implementation. The only think that i could suggest is to train the RL agent with more than one opponent in a gradual way. First against an easy-to-beat agent and than gradually towards the expert. It should make more robust the learning of the agent.  
> 
> 

NOTE: Same here, the cose is really long and verbose and since my review should be quite detailed and self-explanatory, I don't report Flavio's code hoping it won't be a problem.