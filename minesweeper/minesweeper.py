import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        # The known_mines function should return a set of all of the cells in self.cells
        #  that are known to be mines.
        # If the number of cells is equal to the number of mines, they must be all be mines
        if self.count == len(self.cells):
            return self.cells
        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        # The known_safes function should return a set of all the cells in self.cells
        # wrong! if self.count != len(self.cells):

        if self.count == 0:
            return self.cells
        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        # The mark_mine function should first check to see if cell is one of the cells included in the sentence.
        if cell in self.cells:
            # If cell is in the sentence, the function should update the sentence so that cell is no longer in
            #  the sentence, but still represents a logically correct sentence given that cell is known to be a mine.
            self.cells.remove(cell)
            self.count = self.count - 1
        # If cell is not in the sentence, then no action is necessary

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        # The mark_safe function should first check to see if cell is one of the cells included in the sentence.
        if cell in self.cells:
            # If cell is in the sentence, the function should update the sentence so that cell is no longer in
            # the sentence, but still represents a logically correct sentence given that cell is known to be safe.
            self.cells.remove(cell)
        # If cell is not in the sentence, then no action is necessary.


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # add_knowledge should accept a cell (represented as a tuple (i, j))
        #  and its corresponding count, and update
        #   self.mines, self.safes, self.moves_made, and self.knowledge with any new information that the AI can infer,
        #   given that cell is known to be a safe cell with count mines neighboring it.
        # The function should mark the cell as one of the moves made in the game.
        self.moves_made.add(cell)
        # The function should mark the cell as a safe cell, updating any sentences that contain the cell as well.
        self.mark_safe(cell)
        # The function should add a new sentence to the AI’s knowledge base, based on the value of cell and count,
        # to indicate that count of the cell’s neighbors are mines. Be sure to only include cells whose state
        # is still undetermined in the sentence.

        # Find all neighbors and count, append to knowledge base
        neighbors = set()  # Empty set
        # Loop through cells -1 to +1 beside and on top/below of cell.
        for delta_i in range(-1, 2):
            for delta_j in range(-1, 2):
                new_ref_i = cell[0] + delta_i
                new_ref_j = cell[1] + delta_j
                # check if valid pixel
                if ((new_ref_i >= 0) and (new_ref_i < self.height) and (new_ref_j >= 0) and
                        (new_ref_j < self.width) and not (new_ref_i == cell[0] and new_ref_j == cell[1])
                        and ((new_ref_i, new_ref_j) not in self.safes)):
                    neighbors.add((new_ref_i, new_ref_j))

        new_sentence = Sentence(sorted(neighbors), count)
        self.knowledge.append(new_sentence)

        # If, based on any of the sentences in self.knowledge, new sentences can be inferred
        # (using the subset method described in the Background), then those sentences should be
        # added to the knowledge base as well.
        # Note that any time that you make any change to your AI’s knowledge, it may be possible to draw new
        # inferences that were not possible before. Be sure that those new inferences are added to the
        # knowledge base if it is possible to do so.

        # Update knowledge base
        for sentence in self.knowledge:
            for cell in sentence.cells.copy():
                if cell in self.mines:
                    self.mark_mine(cell)
                elif cell in self.safes:
                    self.mark_safe(cell)

        # If the number of cells in sentence is equal to the count, they are mines
        for sentence in self.knowledge:
            if sentence.count == len(sentence.cells):
                for cell in sentence.cells.copy():
                    self.mark_mine(cell)

        # If sentence count is 0, all cells are safe
        for sentence in self.knowledge:
            if sentence.count == 0:
                for cell in sentence.cells.copy():
                    self.mark_safe(cell)

        # Merge Subsets (TODO: There is a better way to do this eventually, OK for now).
        for sentence in self.knowledge.copy():
            if new_sentence.cells and new_sentence.cells != sentence.cells:
                if new_sentence.cells.issubset(sentence.cells):
                    diff = sentence.cells.difference(new_sentence.cells)
                    new_count = sentence.count - new_sentence.count
                    new_sentence = Sentence(sorted(diff), new_count)
                    if new_sentence not in self.knowledge:
                        self.knowledge.append(new_sentence)
            if sentence.cells and sentence.cells != new_sentence.cells:
                if sentence.cells.issubset(new_sentence.cells):
                    diff = new_sentence.cells.difference(sentence.cells)
                    new_count = new_sentence.count - sentence.count
                    new_sentence = Sentence(sorted(diff), new_count)
                    if new_sentence not in self.knowledge:
                        self.knowledge.append(new_sentence)

        # Remove empty sets
        for sentence in self.knowledge.copy():
            if len(sentence.cells) == 0:
                self.knowledge.remove(sentence)

        # Remove duplicated
        for sentence1, sentence2 in itertools.combinations(self.knowledge.copy(), 2):
            if sorted(sentence1.cells) == sorted(sentence2.cells):
                self.knowledge.remove(sentence1)

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        # make_safe_move should return a move (i, j) that is known to be safe.
        # The move returned must be known to be safe, and not a move already made.
        # Use Python’s documentation on classes remaining_cells = self.safes - self.moves_made
        # If found 0, these cells are safe, need to add
        remaining_cells = self.safes.difference(self.moves_made)
        if len(remaining_cells) > 0:
            move = random.sample(remaining_cells, 1)[0]
            return move

        # If no safe move can be guaranteed, the function should return None.
        # The function should not modify self.moves_made, self.mines, self.safes, or self.knowledge.
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        # make_random_move should return a random move (i, j).
        # This function will be called if a safe move is not possible:
        # if the AI does not know where to move, it will choose to move randomly instead.
        # The move must not be a move that has already been made.
        # The move must not be a move that is known to be a mine.
        # If no such moves are possible, the function should return None.
        cells = set()
        # Loop through all cells
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.mines and (i, j) not in self.moves_made:
                    cells.add((i, j))
        if len(cells) == 0:
            return None
        move = random.sample(cells, 1)[0]
        return move
