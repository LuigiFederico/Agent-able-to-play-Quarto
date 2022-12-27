import numpy as np
import random
from objects import Player, Quarto


class my_player(Player):
    def __init__(self, quarto: Quarto) -> None:
        super.__init__(quarto)
        self.safe_pieces = [True for _ in range(16)]     # True if safe, False if threat
        self.critical_lines = [False for _ in range(10)] # (from left to right) vertical x4, horizontal x4, diagonal x2 
        self.placed_pieces = [False for _ in range(16)]  # True if placed, False if not placed
        self.best_position = -1


    def choose_piece(self) -> int:
        piece_to_place = self.get_game().get_selected_piece()
        choosen_piece = -1

        # First ply
        if piece_to_place == -1:
            choosen_piece = random.randint(0, 15)    # Choose a random piece
            self.placed_pieces[choosen_piece] = True # Flag the piece as placed
            return choosen_piece

        
        # Critical lines - threat mode
        if self.__check_critical_lines():
            # Distinguish between safe and threat pieces
            dict_threat_line = self.__find_safe_pieces()

            # If I hold a winning piece
            # No matter what I choose, I win if I place it on the critical line
            if self.safe_pieces[piece_to_place] == False:
                # TROVA LA GIUSTA CRITICAL LINE E PIAZZALO LA'
                # usa dict threat line
                raise NotImplementedError

            # Choose the best piece
            else:
                num_safe_pieces = sum([1 for _ in self.safe_pieces if _])

                # Advantageed position - odd number of safe pieces
                if (num_safe_pieces % 2) != 0:
                    # --> Gioca un safe random / prova tutti i casi e scegli la migliore 
                    raise NotImplementedError

                # Disadvantaged position - even number of safe pieces
                else:
                     # --> # pezzi safe > 2? (4, 6, 8, etc?)
                    if num_safe_pieces > 2:
                        # -----> SI, posticipa e prova tutti i casi successivi
                        raise NotImplementedError
                    else:
                        raise NotImplementedError              
                    # -----> NO, inibisci la riga critica
            
                pass
        
        # No critical lines - evolved mode
        else: 
            # Evolved strategy / RL / random / dumb
            pass

        # Flag the piece as placed
        self.placed_pieces[choosen_piece] = True

        raise NotImplementedError


    def place_piece(self) -> tuple[int, int]:
        # IN TEORIA LO PRENDI DA UNO STATO SETTATO DA CHOOSE_PIECE
        # CAPISCI COME VENGONO PIAZZATI I PEZZI!!!!!!!
        piece_to_place = self.get_game().get_selected_piece()
        
        # Flag the piece as places
        self.placed_pieces[piece_to_place] = True

        raise NotImplementedError


    def __check_critical_lines(self) -> bool:
        """
        Check if there are "critic lines", i.e. lines with 3 pieces.
        When it finds one, it turns on the corrisponding flag inside self.critical_lines.

        Return True if it finds at least one critical line.
        """

        # Setup
        board = self.get_game().get_board_status()
        found = False
        self.critical_lines = [False for _ in range(10)] # Reset flags
        line_index = 0   # It follows self.critical_lines
        dim = self.get_game().BOARD_SIDE

        # Check vertical
        for col in range(dim):
            cnt = 0
            for row in range(dim):
                if board[row, col] == -1:
                    cnt += 1
            if cnt == 1:
                self.critical_lines[line_index] = True
                found = True
            line_index += 1

        # Check horizontal
        for row in range(dim):
            cnt = 0
            for col in range(dim):
                if board[col, row] == -1:
                    cnt += 1
            if cnt == 1:
                self.critical_lines[line_index] = True
                found = True
            line_index += 1

        # Left to right
        cnt = 0
        for i in range(dim):
            if board[i, i] == -1:
                cnt += 1
        if cnt == 1:
            self.critical_lines[line_index] = True
            found = True
        line_index += 1
            
        
        # Right to left
        cnt = 0
        for i in range(dim):
            if board[i, dim - 1 - i] == -1:
                cnt += 1
        if cnt == 1:
            self.critical_lines[line_index] = True
            found = True
        line_index += 1

        return found


    def __find_safe_pieces(self) -> dict:
        """
        For each critical line, identify the piece attributes (max 2) that 
        which would make the opponent win and maps the piaces 
        on the flag vector self.__safe_piaces as follows:

        - Threat piece -> piece with that attribute(s)
        - Safe piece   -> piece without that attribute(s)

        Returns a dictionay with:
        - key = piece number (from 0 to 15)
        - value = list of index for self.critical_lines. If you place the key piece on that line, you win
        """

        # Setup
        board = self.get_game().get_board_status()
        dim = self.get_game().BOARD_SIDE 
        dict_threat_piece_crit_line = {}  # key = piece number; value = list of critical line where to put to win
        for i in range(16):
            dict_threat_piece_crit_line[i] = []

        # Identify critical lines
        vertical_crit_lines = [idx for idx, flag in enumerate(self.critical_lines) if idx < 4 and flag ]   
        horizontal_crit_lines = [idx for idx, flag in enumerate(self.critical_lines) if (idx >= 4 and idx < 8) and flag ] 
        diag1_crit_line = self.critical_lines[9]  # diagonal from left to right
        diag2_crit_line = self.critical_lines[10] # diagonal from right to left

        # Reset - True if safe, False if threat
        self.safe_pieces = [True for _ in range(16)]

        # Vertical
        for col in vertical_crit_lines:
            crit_pieces = [piece for piece in board[:, col] if piece != -1]
            self.__find_critical_attributes(crit_pieces, col, dict_threat_piece_crit_line)
            
        # Horizontal
        for row in horizontal_crit_lines:
            crit_pieces = [piece for piece in board[row - 4, :] if piece != -1]
            self.__find_critical_attributes(crit_pieces, row, dict_threat_piece_crit_line)

        # Diagonal 
        crit_pieces = []
        if diag1_crit_line:
            for idx in range(dim):
                piece = board[idx, idx]
                if piece != -1:
                    crit_pieces.append(piece)
            self.__find_critical_attributes(crit_pieces, 9, dict_threat_piece_crit_line)

        crit_pieces = []
        if diag2_crit_line:
            for idx in range(dim):
                piece = board[idx, idx]
                if piece != -1:
                    crit_pieces.append(piece)
            self.__find_critical_attributes(crit_pieces, 10, dict_threat_piece_crit_line)       

        return dict_threat_piece_crit_line    

    
    def __find_critical_attributes(self, crit_pieces: list, crit_line: int, dict_piece_line: dict) -> None:
        """
        Finds the critical attribute, i.e. the attribute that would make the opponent win
        and marks on self.safe_pieces if a piece is a threat (has that attribute) 
        or is safe (not threat) to give.

        This function uses bitwise operations (AND and NOR) with masks in order to determine
        if an attribute is common to all three pieces.

        NOTE: There could be at most 2 critical attribute on a critical line and at least 1.
        """
        
        assert len(crit_pieces) == 3

        def mark_threat_pieces(attribute_mask, one: bool) -> None:
            # For each piece..
            # if it has the attribute, flag it as threat
            for p in range(16):
                if one and (p & mask):          # True attribute
                    self.safe_pieces[p] = False
                    if crit_line not in dict_piece_line[p]:
                        dict_piece_line[p].append(crit_line)
                if not one and (~(p | ~mask)):  # False attribute
                    self.safe_pieces[p] = False
                    if crit_line not in dict_piece_line[p]:
                        dict_piece_line[p].append(crit_line)


        mask = 0b1000
        found = False

        # For each piece attribute..
        for _ in range(4):
            # Verify the True attribute - AND
            found = True
            for p in crit_pieces:
                if (p & mask) == 0:
                    found = False
            if found:
                mark_threat_pieces(mask, True)
            
            # Verifiy the False attribute - NOR
            found = True
            for p in crit_pieces:
                if (~(p | ~mask)) == 0:
                    found = False
            if found:
                mark_threat_pieces(mask, False)
            
            # Next pair of attributes
            mask >> 1 





# Mi serve un modo per capire se
# - esiste una linea critica
#   - guarda la board
#   - esistono righe da tre valori?
#   - se sì, deve dirmi i pezzi threat
#   - ovvero quelli che se dò all'avversario perdo 
# - quali pezzi sono safe to play