import numpy as np
import random
from objects import Player, Quarto


class my_player(Player):
    def __init__(self, quarto: Quarto) -> None:
        super.__init__(quarto)
        self.safe_pieces = [True for _ in range(16)]     # True if safe, False if threat
        self.critical_lines = [False for _ in range(10)] # (from left to right) vertical x4, 
                                                         # (top to bottom) horizontal x4, 
                                                         # (left to right first) diagonal x2 
        self.placed_pieces = [False for _ in range(16)]  # True if placed, False if not placed
        self.best_position = -1

    def choose_piece(self) -> int:
        self.__update_cooked_info()

        # STATO = vettore con indx i pezzi e come valore la posizione su cui sono piazzati
        #
        # ESISTE UNA RIGA CRITICA?
        # SI 
        # -- VANTAGGIO ==> DAI UN PEZZO SAFE
        #                  - POSSIBILITà: RANDOM, RL, GA, MIN-MAX
        # -- SVANTAGGIO
        #    -- ASPETTA ==> DAI UN PEZZO SAFE (MAGARI L'AVVERSARIO TI DA' QUELLO VINCENTE)
        #                  - POSSIBILITà: RANDOM, RL, GA, MIN-MAX
        #    -- INIBISCI LA RIGA THREAT 
        # 
        # NO
        # -- SCEGLI IL PEZZO
        #        - POSSIBILITà: RANDOM, RL, GA, MIN-MAX

        raise NotImplemented


    #  def choose_piece(self) -> int:
    #     self.__update_cooked_info()
        
    #     piece_to_place = self.get_game().get_selected_piece()
    #     choosen_piece = -1

    #     # First ply
    #     if piece_to_place == -1:
    #         choosen_piece = random.randint(0, 15)    # Choose a random piece
    #         self.placed_pieces[choosen_piece] = True # Flag the piece as placed
    #         return choosen_piece

        
    #     # Critical lines - threat mode
    #     if self.__check_critical_lines():
    #         # Distinguish between safe and threat pieces
    #         dict_threat_line = self.__find_safe_pieces()

    #         # If I hold a winning piece
    #         # No matter what I choose, I'll win if I place it on the critical line
    #         if self.safe_pieces[piece_to_place] == False:
    #             # TROVA LA GIUSTA CRITICAL LINE E PIAZZALO LA'
    #             # usa dict threat line
    #             raise NotImplementedError

    #         # Choose the best piece
    #         else:
    #             num_safe_pieces = sum([1 for _ in self.safe_pieces if _])

    #             # Advantageed position - odd number of safe pieces
    #             if (num_safe_pieces % 2) != 0:
    #                 # --> Gioca un safe random / prova tutti i casi e scegli la migliore 
    #                 raise NotImplementedError

    #             # Disadvantaged position - even number of safe pieces
    #             else:
    #                 # --> # pezzi safe > 2? (4, 6, 8, etc?)
    #                 if num_safe_pieces > 2:
    #                     # -----> SI, posticipa e prova tutti i casi successivi
    #                     raise NotImplementedError
    #                 else:
    #                     raise NotImplementedError              
    #                     # -----> NO, inibisci la riga critica
        
    #     # No critical lines - evolved mode
    #     else: 
    #         # Evolved strategy / RL / random / dumb
    #         raise NotImplementedError 

    #     # Flag the piece as placed
    #     self.placed_pieces[choosen_piece] = True

    #     raise NotImplementedError



    def place_piece(self) -> tuple[int, int]:
        self.__update_cooked_info()

        # STATO = ??
        #
        # ESISTE RIGA CRITICA?
        # SI
        # -- HO UN PEZZO THREAT ?
        #    -- SI --> WIN (PIAZZALO SULLA RIGA CRITICA)
        #    -- NO 
        #       -- SONO IN VANTAGGIO?
        #           -- NO -> GIOCA DOVE NON CREA SVANTAGGIO
        #                   -- RANDOM, GA, LR, MIN-MAX
        #           -- SI -> SE ASPETTO -> GIOCA DOVE NON CREA SVANTAGGIO (COME IL NO)
        #                 -> NON ASPETTO -> TAPPA LA RIGA CRITICA  
        #
        # NO
        # -- PIAZZA IL PEZZO (RANDOM, GA, RL, MIN-MAX, HARD-CODED)



        raise NotImplemented

    # def place_piece(self) -> tuple[int, int]:
    #     # IN TEORIA LO PRENDI DA UNO STATO SETTATO DA CHOOSE_PIECE
    #     # CAPISCI COME VENGONO PIAZZATI I PEZZI!!!!!!!
    #     piece_to_place = self.get_game().get_selected_piece()
        
    #     # Flag the piece as places
    #     self.placed_pieces[piece_to_place] = True

    #     raise NotImplementedError


    def __update_cooked_info(self):
        self.__check_critical_lines()
        # Safe and threat pieces..


    def __check_critical_lines(self) -> bool:
        """
        Check if there are "critic lines", i.e. lines with 3 pieces.
        When it finds one, it turns on the corrisponding flag inside self.critical_lines.

        Return True if it finds at least one critical line.
        """

        def board_mask(board):
            """ Board mask with 1 on (x,y) if there is a piece, 0 otherwise """
            mask = board != -1
            return mask.astype(int)

        # Setup
        mask = board_mask(self.get_game().get_board_status())
        self.critical_lines = [False for _ in range(10)] # Reset flags
        found = False

        # Check vertical
        vsum = np.sum(mask, axis=0)
        for col_idx, n in enumerate(vsum):
            if n == 3:
                self.critical_lines[col_idx] = True

        # Check horizontal
        hsum = np.sum(mask, axis=1)
        for row_idx, n in enumerate(hsum):
            if n == 3:
                self.critical_lines[row_idx + 4] = True

        # Check diagonal
        dsum1 = np.trace(mask)              # Left to right
        dsum2 = np.trace(np.fliplr(mask))   # Right to left

        if dsum1 == 3:
            self.critical_lines[8] = True
        if dsum2 == 3:
            self.critical_lines[9] = True

        return np.any(self.critical_lines)


    # DA SISTEMARE
    # def __find_safe_pieces(self) -> dict:
    #     """
    #     For each critical line, identify the piece attributes (max 2) 
    #     which would make the opponent win and map the piaces 
    #     on the flag vector self.__safe_piaces as follows:

    #     - Threat piece -> piece with that attribute(s)
    #     - Safe piece   -> piece without that attribute(s)

    #     Returns a dictionay with:
    #     - key = piece number (from 0 to 15)
    #     - value = list of index for self.critical_lines. If you place the key piece on that line, you win
    #     """

    #     # Setup
    #     board = self.get_game().get_board_status()
    #     dim = self.get_game().BOARD_SIDE 
    #     dict_threat_piece_crit_line = {}  # key = piece number; value = list of critical line where to put to win
    #     for i in range(16):
    #         dict_threat_piece_crit_line[i] = []

    #     # Identify critical lines
    #     vertical_crit_lines = [idx for idx, flag in enumerate(self.critical_lines) if idx < 4 and flag ]   
    #     horizontal_crit_lines = [idx for idx, flag in enumerate(self.critical_lines) if (idx >= 4 and idx < 8) and flag ] 
    #     diag1_crit_line = self.critical_lines[9]  # diagonal from left to right
    #     diag2_crit_line = self.critical_lines[10] # diagonal from right to left

    #     # Reset - True if safe, False if threat
    #     self.safe_pieces = [True for _ in range(16)]

    #     # Vertical
    #     for col in vertical_crit_lines:
    #         crit_pieces = [piece for piece in board[:, col] if piece != -1]
    #         self.__find_critical_attributes(crit_pieces, col, dict_threat_piece_crit_line)
            
    #     # Horizontal
    #     for row in horizontal_crit_lines:
    #         crit_pieces = [piece for piece in board[row - 4, :] if piece != -1]
    #         self.__find_critical_attributes(crit_pieces, row, dict_threat_piece_crit_line)

    #     # Diagonal 
    #     crit_pieces = []
    #     if diag1_crit_line:
    #         for idx in range(dim):
    #             piece = board[idx, idx]
    #             if piece != -1:
    #                 crit_pieces.append(piece)
    #         self.__find_critical_attributes(crit_pieces, 9, dict_threat_piece_crit_line)

    #     crit_pieces = []
    #     if diag2_crit_line:
    #         for idx in range(dim):
    #             piece = board[idx, idx]
    #             if piece != -1:
    #                 crit_pieces.append(piece)
    #         self.__find_critical_attributes(crit_pieces, 10, dict_threat_piece_crit_line)       

    #     return dict_threat_piece_crit_line    

    
    # def __find_critical_attributes(self, crit_pieces: list, crit_line: int, dict_piece_line: dict) -> None:
    #     """
    #     Finds the critical attribute, i.e. the attribute that would make the opponent win
    #     and marks on self.safe_pieces if a piece is a threat (has that attribute) 
    #     or is safe (not threat) to give.

    #     This function uses bitwise operations (AND and NOR) with masks in order to determine
    #     if an attribute is common to all three pieces.

    #     NOTE: There could be at most 2 critical attribute on a critical line and at least 1.
    #     """
        
    #     assert len(crit_pieces) == 3

    #     def mark_threat_pieces(attribute_mask, one: bool) -> None:
    #         # For each piece..
    #         # if it has the attribute, flag it as threat
    #         for p in range(16):
    #             if one and (p & mask):          # True attribute
    #                 self.safe_pieces[p] = False
    #                 if crit_line not in dict_piece_line[p]:
    #                     dict_piece_line[p].append(crit_line)
    #             if not one and (~(p | ~mask)):  # False attribute
    #                 self.safe_pieces[p] = False
    #                 if crit_line not in dict_piece_line[p]:
    #                     dict_piece_line[p].append(crit_line)


    #     mask = 0b1000
    #     found = False

    #     # For each piece attribute..
    #     for _ in range(4):
    #         # Verify the True attribute - AND
    #         found = True
    #         for p in crit_pieces:
    #             if (p & mask) == 0:
    #                 found = False
    #         if found:
    #             mark_threat_pieces(mask, True)
            
    #         # Verifiy the False attribute - NOR
    #         found = True
    #         for p in crit_pieces:
    #             if (~(p | ~mask)) == 0:
    #                 found = False
    #         if found:
    #             mark_threat_pieces(mask, False)
            
    #         # Next pair of attributes
    #         mask >> 1 


