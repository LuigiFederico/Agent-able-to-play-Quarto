from copy import deepcopy
import numpy as np
from objects import Player, Quarto

WIN = 5
LOSE = -5
DROW = 1

dec_to_bin = {
    0: [0, 0, 0, 0],
    1: [0, 0, 0, 1],
    2: [0, 0, 1, 0],
    3: [0, 0, 1, 1],
    4: [0, 1, 0, 0],
    5: [0, 1, 0, 1],
    6: [0, 1, 1, 0],
    7: [0, 1, 1, 1],
    8: [1, 0, 0, 0],
    9: [1, 0, 0, 1],
    10: [1, 0, 1, 0],
    11: [1, 0, 1, 1],
    12: [1, 1, 0, 0],
    13: [1, 1, 0, 1],
    14: [1, 1, 1, 0],
    15: [1, 1, 1, 1]
}

class the_choosen_one(Player):

    def __init__(self, quarto: Quarto) -> None:
        super.__init__(quarto)
        self.frontiers = self.get_frontiers()
        self.piece_to_give = -1    

    # ToDo
    def choose_piece(self) -> int:
        # Best piece to give computed inside self.place_piece() with MinMax
        if self.piece_to_give != -1:
            p = self.piece_to_give
            self.piece_to_give = -1
            return p
        
        # Setup
        board = self.get_game().get_board_status()
        safe, threat = self.distinguish_pieces(board)

        

        raise NotImplemented

    
    def place_piece(self) -> tuple[int, int]:
    
        # Setup
        board = self.get_game().get_board_status()
        piece_to_place = self.get_game().get_selected_piece()
        safe, threat = self.distinguish_pieces(board)
        
        # Threat mode -> MinMax
        if len(threat) > 0: 
            # If I have to place a threat piece, I win
            if piece_to_place in threat:   
                score_board = self.__board_intersection_scores(piece_to_place)      
                ply = np.unravel_index(np.argmax(score_board), score_board.shape)   # Max score on the winning position
                return ply

            # MinMax
            ply, piece, _ = self.min_max_place_piece(board, piece_to_place, player = 0)
            if piece != -1:
                self.piece_to_give = piece  # No need to run MinMax again to choose the piece

        # Safe mode -> Euristic strategy
        else:   
            ply = self.__safe_mode_place_piece(piece_to_place, delay_minmax = True)

        return ply

    # ToDo (Sistamre la descrizione)
    @staticmethod
    def get_frontiers():
        """
            Generates all the frontiers for each possible combination of 1, 2 and 3 elements.

            Returns a dictionary with a tuple as key, containing the numbers (1, 2 or 3 numbers)
            representative for the frontier. The frontier is a dictionary with keys between 0, 1, 2 and 3 
            according to how many attributes the representative tuple has in common with the number inside the value of the frontier.

            DA RIPARAFRASARE #########################################
        """
        
        def get_frontier(p: int):
            """ Frontiers with 1 piece """
            assert p >= 0 and p <= 15

            x = dec_to_bin[p]
            frontier = {
                0: [],
                1: [],
                2: [],
                3: []
            }

            for num, binary in dec_to_bin.items():
                if num == p:
                    continue
                cnt = 0
                for i, j in zip(x, binary):
                    if i == j:
                        cnt += 1
                frontier[cnt].append(num)
            
            return frontier

        def get_frontier_2d(a: int, b:int):
            """ Frontiers with 2 pieces """
            assert a >= 0 and a <= 15
            assert b >= 0 and b <= 15
            assert a != b

            x = dec_to_bin[a]
            y = dec_to_bin[b]
            frontier = {
                0: [],
                1: [],
                2: [],
                3: []
            }

            for num, binary in dec_to_bin.items():
                if num == a or num == b:
                    continue
                cnt = 0
                for x_i, y_i, j in zip(x, y, binary):
                    if x_i == j and y_i == j:
                        cnt += 1
                frontier[cnt].append(num)
            
            return frontier

        def get_frontier_3d(a: int, b:int, c: int):
            """ Frontiers with 3 pieces """
            assert a >= 0 and a <= 15
            assert b >= 0 and b <= 15
            assert c >= 0 and c <= 15
            assert a != b and a != c and b != c

            x = dec_to_bin[a]
            y = dec_to_bin[b]
            z = dec_to_bin[c]
            frontier = {
                0: [],
                1: [],
                2: [],
                3: []
            }

            for num, binary in dec_to_bin.items():
                if num == a or num == b or num == c:
                    continue
                cnt = 0
                for x_i, y_i, z_i, j in zip(x, y, z, binary):
                    if x_i == j and y_i == j and z_i == j:
                        cnt += 1
                frontier[cnt].append(num)
            
            return frontier

        frontier = {}

        # one piece -> 16
        for p in range(16):
            frontier[(p, )] = get_frontier(p)
        
        # two pieces -> 120 combinations
        for i in range(16):
            for j in range(i+1, 16):
                frontier[(i, j)] = get_frontier_2d(i, j)

        # three pieces -> 560 combinations
        for i in range(16):
            for j in range(i+1, 16):
                for k in range(j+1, 16):
                    frontier[(i, j, k)] = get_frontier_3d(i, j, k)
                    
        return frontier
  

    def distinguish_pieces(self, board):
        """ Returns a list of safe pieces and a list of threat pieces """

        threat = set()

        # Rows
        for row in range(board.shape[0]):
            k = tuple(sorted([p for p in board[row, :] if p != -1])) # frontier source (key)
            if len(k) == 3: # Threat line
                k_threat = set(self.frontiers[k][1]) | set(self.frontiers[k][2])  # pieces with attributes in common with all of 3
                threat |= k_threat

        # Cols
        for col in range(board.shape[1]):
            k = tuple(sorted([p for p in board[:, col] if p != -1]))
            if len(k) == 3: # Threat line
                k_threat = set(self.frontiers[k][1]) | set(self.frontiers[k][2])  # pieces with attributes in common with all of 3
                threat |= k_threat
        
        # Diag 1
        k = tuple(sorted([p for p in board.diagonal() if p != -1]))    
        if len(k) == 3: # Threat line
            k_threat = set(self.frontiers[k][1]) | set(self.frontiers[k][2])  # pieces with attributes in common with all of 3
            threat |= k_threat

        # Diag 2   
        k = tuple(sorted([p for p in np.fliplr(board).diagonal() if p != -1]))  
        if len(k) == 3: # Threat line
            k_threat = set(self.frontiers[k][1]) | set(self.frontiers[k][2])  # pieces with attributes in common with all of 3
            threat |= k_threat   
        

        # Clean up
        safe = set(range(16))
        placed = set([int(p) for p in np.nditer(board) if p != -1])
        threat -= placed
        safe -= placed
        safe -= threat
        
        return list(safe), list(threat)


    def __board_intersection_scores(self, piece: int):
        """
            Computes the mapping of the board with scores in it using the frontiers
        """

        def extract_source(board_slice) -> tuple:
            """ Extracts the a key for self.frontiers """
            k = tuple(sorted([p for p in board_slice if p != -1]))
            return k

        def compute_score(frontier) -> int:
            rank = -1                           # Frontier rank = number of common attributes with piece
            for i in range(4):
                if piece in frontier[i]:
                    rank = i
                    break

            rescaling = 10**(len(k) - 1)
            score = rank * rescaling
            return score
            

        # Setup
        board = self.get_game().get_board_status()
        board_scores = np.zeros(shape=(Quarto.BOARD_SIDE, Quarto.BOARD_SIDE), dtype=int)   # Initialized mapped board

        # Rows
        for row in range(board.shape[0]):
            k = extract_source(board[row, :])   # Extract the frontier's source
            if k == () or len(k) == 4:
                continue 
            score = compute_score(self.frontiers[k]) 
            board_scores[row, :] += score
        
        # Cols
        for col in range(board.shape[1]):
            k = extract_source(board[:, col])       
            if k == () or len(k) == 4:
                continue 
            score = compute_score(self.frontiers[k]) 
            board_scores[:, col] += score      

        # Diag 1
        k = extract_source(board.diagonal())    
        if k != () and len(k) != 4:
            score = compute_score(self.frontiers[k]) 
            tmp = np.zeros((4, 4), dtype=int)   # Update the diagonal
            np.fill_diagonal(tmp, score)
            board_scores += tmp

        # Diag 2   
        k = extract_source(np.fliplr(board).diagonal())       
        if k != () and len(k) != 4:
            score = compute_score(self.frontiers[k]) 
            tmp = np.zeros((4, 4), dtype=int)      
            np.fill_diagonal(np.fliplr(tmp), score)
            board_scores += tmp

        # Clean up board_scores
        mask = np.array(board == -1, dtype=int)     # 0 if there is a piece in the i,j board coordinate, 1 otherwise
        board_scores = board_scores * mask

        return board_scores

    # Tuning: delay_minmax
    def __safe_mode_place_piece(self, piece: int, *, delay_minmax = True) -> tuple[int, int]:
        
        # Setup
        board = self.get_game().get_board_status()    
        possible_actions = self.__list_ply_score(board, piece)

        # Sort the action 
        if delay_minmax:
            possible_actions = sorted(possible_actions, key=lambda a: a[1])    # ascending score (MinMax delayed amap)
        else:
            possible_actions = sorted(possible_actions, key=lambda a: -a[1])     # descending score (MinMax asap) 

        # Check if the action is safe (if there are enough pieces to choose)
        idx = 0
        ply, score = possible_actions[idx]
        while (self.__check_place_ply(ply, piece) == False):
            idx += 1
            ply, score = possible_actions[idx]

        return ply


    def __list_ply_score(self, board, piece):
        """
            List all the possible positions where to place the piece with score as tuples ((x,y), score)
        """
        possible_actions = []
        score_board = self.__board_intersection_scores(piece)

        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                if board[row, col] == -1:       # Possible action
                    possible_actions.append( ((row, col), score_board[row, col]) )
        
        return possible_actions

    # Tuning: safe_threashold
    def __check_place_ply(self, ply: tuple[int, int], piece: int, *, safe_threshold = 1):
        """ 
            Returns True if there are enough safe pieces to choose after placing 
            the given piece on board using the ply coordinates.
        """
        board = self.get_game().get_board_status() 

        assert board[ply[0], ply[1]] == -1  # The ply must be legal
        
        board[ply[0], ply[1]] = piece
        safe, _ = self.distinguish_pieces(board)

        # I want enough safe pieces to choose after having placed the given piece on the board
        if len(safe) >= safe_threshold:
            return True
        else:
            return False


    def min_max_place_piece(self, board_now, piece, player):
        """
            MinMax algorithm with alpha-beta pruning and greedy sorting for placing a piece.

            The greedy sorting tryes to speed up MinMax by sorting the possible actions by 
            the corresponding score computed with self.__list_ply_score(•).
            IDEA: Expand first the actions that could lead faster to a winning/drow condition,
            hoping to favor the alpha-beta pruning. 

            Args:
            - board -> current state of the board (numpy.ndarray)
            - piece -> piece to place (int)
            - player -> 0: my agent, 1: opponent

            Returns:
            - ply = coordinates as tuple es. (x, y)
            - piece = best piece to give associated with the best ply
            - score -> WIN = 5, LOSE = -5, DROW = 1
        """

        possible_actions = self.__list_ply_score(board_now, piece)          # Descending score:
        possible_actions = sorted(possible_actions, key=lambda a: a[1])     # Greedy sorting to speed up alpha-beta pruning

        for ply, _ in iter(possible_actions):
            if board_now[ply[0], ply[1]] != -1:     # Check that the ply is legal
                continue
            board = deepcopy(board_now)             # Ply     
            board[ply[0], ply[1]] = piece
            
            if self.check_win(board):               # Termination
                if player == 0:
                    return ply, -1, WIN
                else:
                    return ply, -1, LOSE
            
            piece, score = self.min_max_choose_piece(board, player)  # Recursion

            if score == WIN and player == 0:        # Alpha-Beta Pruning
                return ply, piece, WIN
            if score == LOSE and player == 1:
                return ply, piece, LOSE
            if score == DROW:                       # If drow, keep looking for a better ply
                ply_drow = ply
                piece_drow = piece

        # No winning condition -> drow 
        return ply_drow, piece_drow, DROW


    def min_max_choose_piece(self, board, player):
        """
            MinMax algorithm for choosing a piece with alpha-beta pruning and greedy sorting.

            The greedy sorting tryes to speed up MinMax by sorting the safe pieces (the only one expanded)
            by decreasing score. The score is computed as sum of the scores inside the board_score computed 
            by self.__board_intersection_scores(p) for each piece p in the safe pool.
            IDEA: look for pieces that offer more aggressive plays hoping to find faster a winning/drow condition
            in oreder to assist the alpha-beta pruning.

            Args:
            - board -> current state of the board, after having placed the previous given piece (np.ndarray)
            - player -> 0: my agent, 1: opponent

            Returns:
            - piece = choosen piece
            - score -> WIN = 5, LOSE = -5, DROW = 1
        """

        safe, threat = self.distinguish_pieces(board)   # Threat and safe pieces pools

        # Terminations
        if len(safe) + len(threat) == 0:
            return -1, DROW

        if len(safe) == 0:
            if player == 0:
                return threat[0], LOSE
            else:
                return threat[0], WIN

        # Greedy sorting to speed up MinMax pruning
        piece_score = []
        for p in iter(safe):
            board_score = self.__board_intersection_scores(p)
            piece_score.append((p, board_score.sum()))
        piece_score = sorted(piece_score, key=lambda x: -x[1])   # Descending order

        # Recursion
        for piece, _ in iter(piece_score):
            _, _, score = self.min_max_place_piece(board, piece, player = (player + 1) % 2 )  
            
            if player == 0 and score == WIN:        # Alpha-Beta Pruning
                return piece, WIN
            if player == 1 and score == LOSE:
                return piece, LOSE
            if score == DROW:                       # If drow, keep looking
                piece_drow = piece

        # No winning condition -> drow 
        return piece_drow, DROW


    def check_finished(self, board):
        """ Returns True if the board is a winning condition, False otherwise """

        def __check(k):
            for attribute in range(4):
                sum = 0
                for idx in range(4):                      
                    sum += dec_to_bin[ k[idx] ][attribute]
                if sum == 4 or sum == 0:    # Winning condition
                    return True

        # Rows
        for row in range(board.shape[0]):
            k = tuple(sorted([p for p in board[row, :] if p != -1]))
            if len(k) == 4 and __check(k):
                return True

        # Cols
        for col in range(board.shape[1]):
            k = tuple(sorted([p for p in board[:, col] if p != -1]))
            if len(k) == 4 and __check(k):
                return True

        # Diag 1
        k = tuple(sorted([p for p in board.diagonal() if p != -1]))
        if len(k) == 4 and __check(k):
            return True

        # Diag 2
        k = tuple(sorted([p for p in np.fliplr(board).diagonal() if p != -1]))
        if len(k) == 4 and __check(k):
            return True    
                        
        return False

        