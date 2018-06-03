import numpy as np
import warnings
from threading import Timer
from copy import deepcopy

warnings.simplefilter(action='ignore', category=FutureWarning)

class Reversi:

    def __init__(self, Matrix = None):

        rows, cols = 8, 8
        if Matrix == None: 
            self.Matrix = np.full((rows, cols), '+', dtype=str)
            self.Matrix[3, 4] = self.Matrix[4, 3] = 'X'
            self.Matrix[3, 3] = self.Matrix[4, 4] = 'O'
        else:
            self.Matrix = Matrix

        # Prints the initial instance of the game
        self.printGame()
        gameType = input("Enter 1 for User vs. Comp, 2 for User vs. User")

        self.timeout = input("How long does each player have to make a move? (in seconds) \n")
        #t = Timer(self.timeout, print, ['Sorry you took too long to make a move, press enter to continue'])
        #t.start()
        #prompt = "You have %d seconds to choose the correct answer...\n" % timeout
        #answer = input(prompt)
        #t.cancel()


        self.flagMove = 0
        ###which game type
        if int(gameType[-1]) == 1:
            self.vsCompGame()
        else:
            self.vsUserGame()



    ##function to check for validity of a move

    # player is 0 for black 1 for white, swapFlag is 0 to check if move is valid and 1 to swap tiles
    def validMove(self, row, col, player, swapFlag, game_field=False):

        if game_field == False:
            Matrix = self.Matrix
        else:
            Matrix = deepcopy(game_field)

        ##returnValue should only be 1 when there is a valid move being made
        returnValue = 0
        # Valid moves for:

        # UP
        # player is black (0) or white (1)
        #print('U',self.validMoveU(row, col, player, swapFlag, Matrix))
        if self.validMoveU(row, col, player, swapFlag, Matrix):
            returnValue = 1

        # DOWN
        #print('D',self.validMoveD(row, col, player, swapFlag, Matrix))
        if self.validMoveD(row, col, player, swapFlag, Matrix):
            returnValue = 1

        # LEFT
        #print('L',self.validMoveL(row, col, player, swapFlag, Matrix))
        if self.validMoveL(row, col, player, swapFlag, Matrix):
            returnValue = 1

        # RIGHT
        #print('R',self.validMoveR(row, col, player, swapFlag, Matrix))
        if self.validMoveR(row, col, player, swapFlag, Matrix):
            returnValue = 1

        # DIAGONAL RIGHT UP
        #print('DRU',self.validMoveDRU(row, col, player, swapFlag, Matrix))
        if self.validMoveDRU(row, col, player, swapFlag, Matrix):
            returnValue = 1

        # DIAGONAL RIGHT DOWN
        #print('DRD',self.validMoveDRD(row, col, player, swapFlag, Matrix))
        if self.validMoveDRD(row, col, player, swapFlag, Matrix):
            returnValue = 1

        # DIAGONAL LEFT UP
        #print('DLU',self.validMoveDLU(row, col, player, swapFlag, Matrix))
        if self.validMoveDLU(row, col, player, swapFlag, Matrix):
            returnValue = 1

        # DIAGONAL LEFT DOWN
        #print('DLD',self.validMoveDLD(row, col, player, swapFlag, Matrix))
        if self.validMoveDLD(row, col, player, swapFlag, Matrix):
            returnValue = 1

        if Matrix[row][col] == 'X' or Matrix[row][col] == 'O':
            returnValue = 0

        # Make current tile an X or O if the return value is 1
        if returnValue == 1 and player == 0 and swapFlag == 1:
            Matrix[row][col] = 'X'
        elif returnValue == 1 and player == 1 and swapFlag == 1:
            Matrix[row][col] = 'O'
        return returnValue, Matrix

    ######### Valid Moves ###################

    def validMoveU(self, row, col, player, swapFlag, Matrix=False):

        if Matrix == False:
            Matrix = self.Matrix

        # check to make sure inputs are in bounds
        if col - 1 < 0:
            return 0

            # checks which player is making the move
        if player == 0:
            # print("UP Move" + str(row) + str(col))
            # Checks to see if a tile has been flipped and there is a tile of the same
            # value after
            if Matrix[row][col - 1] == 'X' and self.flagMove == 1:
                self.flagMove = 0
                return 1

            # Checks to see if there is a tile of opposing value next to the cursor
            if Matrix[row][col - 1] == 'O':
                # mark that a move is possible
                self.flagMove = 1

                # Recursion in order to continue checking if there are opposing tiles
                pointer = self.validMoveU(row, col - 1, player, swapFlag, Matrix)
                if pointer == 1 and swapFlag == 1:

                    Matrix[row][col - 1] = 'X'

                return pointer

            self.flagMove = 0
            pointer = 0
            return 0
        elif player == 1:
            # Checks to see if a tile has been flipped and there is a tile of the same
            # value after
            if Matrix[row][col - 1] == 'O' and self.flagMove == 1:
                self.flagMove = 0
                return 1

            # Checks to see if there is a tile of opposing value next to the cursor
            if Matrix[row][col - 1] == 'X':
                # mark that a move is possible
                self.flagMove = 1

                # Recursion in order to continue checking if there are opposing tiles
                pointer = self.validMoveU(row, col - 1, player, swapFlag, Matrix)
                if pointer == 1 and swapFlag == 1:
                    Matrix[row][col - 1] = 'O'

                return pointer

            # flag
            self.flagMove = 0
            pointer = 0
            return 0
            # Return 0 if no moves are found
        return 0

    def validMoveD(self, row, col, player, swapFlag, Matrix=False):

        if Matrix == False:
            Matrix = self.Matrix

        # check to make sure inputs are in bounds
        if col + 1 > 7:
            return 0
        # checks which player is making the move
        if player == 0:
            # print("Down Move" + str(row) + str(col))
            # Checks to see if a tile has been flipped and there is a tile of the same
            # value after
            if Matrix[row][col + 1] == 'X' and self.flagMove == 1:
                self.flagMove = 0
                return 1

            # Checks to see if there is a tile of opposing value next to the cursor
            if Matrix[row][col + 1] == 'O':
                # mark that a move is possible
                self.flagMove = 1

                # Recursion in order to continue checking if there are opposing tiles
                pointer = self.validMoveD(row, col + 1, player, swapFlag, Matrix)
                if pointer == 1 and swapFlag == 1:
                    Matrix[row][col + 1] = 'X'

                return pointer

            self.flagMove = 0
            pointer = 0
            return 0
        elif player == 1:
            # Checks to see if a tile has been flipped and there is a tile of the same
            # value after
            if Matrix[row][col + 1] == 'O' and self.flagMove == 1:
                self.flagMove = 0
                return 1

            # Checks to see if there is a tile of opposing value next to the cursor
            if Matrix[row][col + 1] == 'X':
                # mark that a move is possible
                self.flagMove = 1

                # Recursion in order to continue checking if there are opposing tiles
                pointer = self.validMoveD(row, col + 1, player, swapFlag, Matrix)
                if pointer == 1 and swapFlag == 1:
                    Matrix[row][col + 1] = 'O'
                return pointer

            # flag
            self.flagMove = 0
            pointer = 0
            return 0
            # Return 0 if no moves are found
        return 0

    def validMoveR(self, row, col, player, swapFlag, Matrix=False):
        self.flagMove
        if Matrix == False:
            Matrix = self.Matrix
        # check to make sure inputs are in bounds
        if row + 1 > 7:
            return 0

            # checks which player is making the move
        if player == 0:
            # print("Right Move")

            # Checks to see if a tile has been flipped and there is a tile of the same
            # value after
            if Matrix[row + 1][col] == 'X' and self.flagMove == 1:
                self.flagMove = 0
                return 1

            # Checks to see if there is a tile of opposing value next to the cursor
            if Matrix[row + 1][col] == 'O':

                # mark that a move is possible
                self.flagMove = 1

                # Recursion in order to continue checking if there are opposing tiles
                pointer = self.validMoveR(row + 1, col, player, swapFlag, Matrix)
                if pointer == 1 and swapFlag == 1:

                    Matrix[row + 1][col] = 'X'
                return pointer

            self.flagMove = 0
            pointer = 0
            return 0
        elif player == 1:
            # Checks to see if a tile has been flipped and there is a tile of the same
            # value after
            if Matrix[row + 1][col] == 'O' and self.flagMove == 1:
                self.flagMove = 0
                return 1

            # Checks to see if there is a tile of opposing value next to the cursor
            if Matrix[row + 1][col] == 'X':
                # mark that a move is possible
                self.flagMove = 1

                # Recursion in order to continue checking if there are opposing tiles
                pointer = self.validMoveR(row + 1, col, player, swapFlag, Matrix)
                if pointer == 1 and swapFlag == 1:
                    Matrix[row + 1][col] = 'O'
                return pointer

            # flag
            self.flagMove = 0
            pointer = 0
            return 0
            # Return 0 if no moves are found
        return 0

    def validMoveL(self, row, col, player, swapFlag, Matrix=False):
        self.flagMove
        if Matrix == False:
            Matrix = self.Matrix
        # check to make sure inputs are in bounds
        if row - 1 < 0:
            return 0

        # checks which player is making the move
        if player == 0:
            # print("Left Move")
            # Checks to see if a tile has been flipped and there is a tile of the same
            # value after
            if Matrix[row - 1][col] == 'X' and self.flagMove == 1:
                self.flagMove = 0
                return 1

            # Checks to see if there is a tile of opposing value next to the cursor
            if Matrix[row - 1][col] == 'O':
                # mark that a move is possible
                self.flagMove = 1

                # Recursion in order to continue checking if there are opposing tiles
                pointer = self.validMoveL(row - 1, col, player, swapFlag, Matrix)
                if pointer == 1 and swapFlag == 1:
                    Matrix[row - 1][col] = 'X'
                return pointer

            self.flagMove = 0
            pointer = 0
            return 0
        elif player == 1:
            # Checks to see if a tile has been flipped and there is a tile of the same
            # value after
            if Matrix[row - 1][col] == 'O' and self.flagMove == 1:
                self.flagMove = 0
                return 1

            # Checks to see if there is a tile of opposing value next to the cursor
            if Matrix[row - 1][col] == 'X':
                # mark that a move is possible
                self.flagMove = 1

                # Recursion in order to continue checking if there are opposing tiles
                pointer = self.validMoveL(row - 1, col, player, swapFlag, Matrix)
                if pointer == 1 and swapFlag == 1:
                    Matrix[row - 1][col] = 'O'
                return pointer

            # flag
            self.flagMove = 0
            pointer = 0
            return 0
            # Return 0 if no moves are found
        return 0

    def validMoveDRU(self, row, col, player, swapFlag, Matrix=False):
        if Matrix == False:
            Matrix = self.Matrix
        self.flagMove

        # check to make sure inputs are in bounds
        if row + 1 > 7 or col - 1 < 0:
            return 0

            # checks which player is making the move
        if player == 0:

            # Checks to see if a tile has been flipped and there is a tile of the same
            # value after
            if Matrix[row + 1][col - 1] == 'X' and self.flagMove == 1:
                self.flagMove = 0
                return 1

            # Checks to see if there is a tile of opposing value next to the cursor
            if Matrix[row + 1][col - 1] == 'O':
                # mark that a move is possible
                self.flagMove = 1

                # Recursion in order to continue checking if there are opposing tiles
                pointer = self.validMoveDRU(row + 1, col - 1, player, swapFlag, Matrix)
                if pointer == 1 and swapFlag == 1:
                    Matrix[row + 1][col - 1] = 'X'
                return pointer

            self.flagMove = 0
            pointer = 0
            return 0
        elif player == 1:
            # Checks to see if a tile has been flipped and there is a tile of the same
            # value after
            if Matrix[row + 1][col - 1] == 'O' and self.flagMove == 1:
                self.flagMove = 0
                return 1

            # Checks to see if there is a tile of opposing value next to the cursor
            if Matrix[row + 1][col - 1] == 'X':
                # mark that a move is possible
                self.flagMove = 1

                # Recursion in order to continue checking if there are opposing tiles
                pointer = self.validMoveDRU(row + 1, col - 1, player, swapFlag, Matrix)
                if pointer == 1 and swapFlag == 1:
                    Matrix[row + 1][col - 1] = 'O'
                return pointer

            # flag
            self.flagMove = 0
            pointer = 0
            # Return 0 if no moves are found
        return 0

    def validMoveDRD(self, row, col, player, swapFlag, Matrix=False):
        if Matrix == False:
            Matrix = self.Matrix

        self.flagMove

        # check to make sure inputs are in bounds
        if col + 1 > 7 or row + 1 > 7:
            return 0

            # checks which player is making the move
        if player == 0:

            # Checks to see if a tile has been flipped and there is a tile of the same
            # value after
            if Matrix[row + 1][col + 1] == 'X' and self.flagMove == 1:
                self.flagMove = 0
                return 1

            # Checks to see if there is a tile of opposing value next to the cursor
            if Matrix[row + 1][col + 1] == 'O':
                # mark that a move is possible
                self.flagMove = 1

                # Recursion in order to continue checking if there are opposing tiles
                pointer = self.validMoveDRD(row + 1, col + 1, player, swapFlag, Matrix)
                if pointer == 1 and swapFlag == 1:
                    Matrix[row + 1][col + 1] = 'X'
                return pointer

            self.flagMove = 0
            pointer = 0
            return 0
        elif player == 1:
            # Checks to see if a tile has been flipped and there is a tile of the same
            # value after
            if Matrix[row + 1][col + 1] == 'O' and self.flagMove == 1:
                self.flagMove = 0
                return 1

            # Checks to see if there is a tile of opposing value next to the cursor
            if Matrix[row + 1][col + 1] == 'X':
                # mark that a move is possible
                self.flagMove = 1

                # Recursion in order to continue checking if there are opposing tiles
                pointer = self.validMoveDRD(row + 1, col + 1, player, swapFlag, Matrix)
                if pointer == 1 and swapFlag == 1:
                    Matrix[row + 1][col + 1] = 'O'
                return pointer

            # flag
            self.flagMove = 0
            pointer = 0
            return 0
            # Return 0 if no moves are found
        return 0

    def validMoveDLD(self, row, col, player, swapFlag, Matrix=False):
        self.flagMove
        if Matrix == False:
            Matrix = self.Matrix

        # check to make sure inputs are in bounds
        if col + 1 > 7 or row - 1 < 0:
            return 0

            # checks which player is making the move
        if player == 0:

            # Checks to see if a tile has been flipped and there is a tile of the same
            # value after
            if Matrix[row - 1][col + 1] == 'X' and self.flagMove == 1:
                self.flagMove = 0
                return 1

            # Checks to see if there is a tile of opposing value next to the cursor
            if Matrix[row - 1][col + 1] == 'O':
                # mark that a move is possible
                self.flagMove = 1

                # Recursion in order to continue checking if there are opposing tiles
                pointer = self.validMoveDLD(row - 1, col + 1, player, swapFlag, Matrix)
                if pointer == 1 and swapFlag == 1:
                    Matrix[row - 1][col + 1] = 'X'
                return pointer

            self.flagMove = 0
            pointer = 0
            return 0
        elif player == 1:
            # Checks to see if a tile has been flipped and there is a tile of the same
            # value after
            if Matrix[row - 1][col + 1] == 'O' and self.flagMove == 1:
                self.flagMove = 0
                return 1

            # Checks to see if there is a tile of opposing value next to the cursor
            if Matrix[row - 1][col + 1] == 'X':
                # mark that a move is possible
                self.flagMove = 1

                # Recursion in order to continue checking if there are opposing tiles
                pointer = self.validMoveDLD(row - 1, col + 1, player, swapFlag, Matrix)
                if pointer == 1 and swapFlag == 1:
                    Matrix[row - 1][col + 1] = 'O'
                return pointer

            # flag
            self.flagMove = 0
            pointer = 0
            return 0
            # Return 0 if no moves are found
        return 0

    def validMoveDLU(self, row, col, player, swapFlag, Matrix=False):
        if Matrix == False:
            Matrix = self.Matrix
        self.flagMove

        # check to make sure inputs are in bounds
        if col - 1 < 0 or row - 1 < 0:
            return 0

            # checks which player is making the move
        if player == 0:

            # Checks to see if a tile has been flipped and there is a tile of the same
            # value after
            if Matrix[row - 1][col - 1] == 'X' and self.flagMove == 1:
                self.flagMove = 0
                return 1

            # Checks to see if there is a tile of opposing value next to the cursor
            if Matrix[row - 1][col - 1] == 'O':
                # mark that a move is possible
                self.flagMove = 1

                # Recursion in order to continue checking if there are opposing tiles
                pointer = self.validMoveDLU(row - 1, col - 1, player, swapFlag, Matrix)
                if pointer == 1 and swapFlag == 1:
                    Matrix[row - 1][col - 1] = 'X'
                return pointer

            self.flagMove = 0
            pointer = 0
            return 0
        elif player == 1:
            # Checks to see if a tile has been flipped and there is a tile of the same
            # value after
            if Matrix[row - 1][col - 1] == 'O' and self.flagMove == 1:
                self.flagMove = 0
                return 1

            # Checks to see if there is a tile of opposing value next to the cursor
            if Matrix[row - 1][col - 1] == 'X':
                # mark that a move is possible
                self.flagMove = 1

                # Recursion in order to continue checking if there are opposing tiles
                pointer = self.validMoveDLU(row - 1, col - 1, player, swapFlag, Matrix)
                if pointer == 1 and swapFlag == 1:
                    Matrix[row - 1][col - 1] = 'O'
                return pointer

            # flag
            self.flagMove = 0
            pointer = 0
            return 0
            # Return 0 if no moves are found
        return 0

    ############## end Movement Validity Functions ##############

    ##############---- GameType Functions ---###################

    # end of game = 1,   Check: 1 Board is full or 2 There is only one color or 3 No more Valid moves
    def endOfGame(self, Matrix=False):
        if Matrix == False:
            Matrix = self.Matrix

        # variables to check number of crosses, X's, O's left on board
        numCrosses = self.gameScore()[2]
        numX = self.gameScore()[0]
        numO = self.gameScore()[1]

        xList = self.listofMoves(0)
        oList = self.listofMoves(1)



        # if no crosses left or there are no x's or o's then game is over
        if numCrosses == 0 or numX == 0 or numO == 0:
            return 1
        # if there are no more valid moves for each player then game is over

        elif len(xList) == 0 and len(oList) == 0:
            print("End of Game xList")
            print(xList)
            print(len(xList))
            return 1
        else:
            return 0

    # returns a list of all possible moves for a certain player, list is in [(x1,y1), (x2,y2), ...] format
    def listofMoves(self, player, Matrix=False):
        if Matrix == False:
            Matrix = self.Matrix

        moves = []
        for x in range(0, 8):
            for y in range(0, 8):
                if player == 0:
                    if self.validMove(x, y, player, 0, Matrix)[0] == 1:
                        moves.append((x + 1, y + 1))
                elif player == 1:
                    if self.validMove(x, y, player, 0. Matrix)[0] == 1:
                        moves.append((x + 1, y + 1))
        return moves

        # returns score of game, positive numbers mean Black (X) is winning, negative numbers mean White (O) is winning

    # user move
    # def userMove(self):
    def gameScore(self, Matrix = False):
        if Matrix == False:
            Matrix = self.Matrix
        numX = 0
        numO = 0
        numCrosses = 0

        for x in range(0, 8):
            for y in range(0, 8):
                if self.Matrix[x][y] == 'X':
                    numX += 1
                if self.Matrix[x][y] == 'O':
                    numO += 1
                if self.Matrix[x][y] == '+':
                    numCrosses += 1
        return numX, numO, numCrosses


    def timer(self):
        return 0

    # Game for User vs. Computer
    def vsCompGame(self):
        print("\n \nWho is the first player?")

        playerVar = input("1 for User, 2 for Computer")

        if int(playerVar[-1]) == 1:
            self.user = 0
            self.computer = 1
            self.usercolor = 'X'
            self.computercolor = 'O'
            startvalue = False
        elif int(playerVar[-1]) == 2:
            self.user = 1
            self.computer = 0
            self.usercolor = 'O'
            self.computercolor = 'X'
            startvalue = True

        while not self.endOfGame():

            validFlag = 0
            # ComputerPlayer
            if startvalue == True:
                a = self.alpha_beta_search()
                print('the computer decided to set his tile on {},{}'.format(a[0]+1, a[1]+1))
                print('asdlkjf')
                self.validMove(a[0], a[1], self.computer, 1)

                self.printGame()
                validFlag = 0

            startvalue = True

            # UserPlayer
            while validFlag == 0:
                # Player 2 Movement
                print("Possible moves in (row,col) format")
                print(self.listofMoves(self.user))
                player2move = input("Player {} in row,col of your current move".format(self.usercolor))
                x2 = int(player2move[-3]) - 1
                print(x2)
                y2 = int(player2move[-1]) - 1
                print(y2)
                if self.validMove(x2, y2, self.user, 1)[0] == 1:
                    print("Move is valid")
                    validFlag = 1
                else:
                    print("Not a Valid move, try a different one")

                    # reprint gameboard
            self.printGame()

        scoreX = self.gameScore()[0]
        scoreO = self.gameScore()[1]
        if scoreX == scoreO:
            print("Game is a tie")
        elif scoreX > scoreO:
            print("Player Black Wins with" + str(scoreX) + " Points")
        elif scoreX < scoreO:
            print("Player White Wins with" + str(abs(scoreO)) + " Points")

        return 0

    # Game for User vs. User

    # Game for User vs. User
    def vsUserGame(self):

        while not self.endOfGame():

            validFlag = 0

            while validFlag == 0:

                # Player1 Move, inputted and printed to std out, and inputted to gameboard
                print("Possible X moves in (row,col) format")
                print(self.listofMoves(0))

                moveTimer = Timer(int(self.timeout), print, ['You took too long, move will be made for you\n Press Enter to Continue'] )
                moveTimer.start()
                player1move = input("Player Black(X) input your current move in row,col\n")
                moveTimer.cancel()

                # Checks to see if player made a move or if move will be
                # made for them
                if player1move != "":
                    x1 = int(player1move[-3]) - 1
                    y1 = int(player1move[-1]) - 1
                    print("Player move is")
                    print(player1move)
                else:
                    tempMove = self.listofMoves(0).pop()
                    x1 = int(tempMove[-2]) -1
                    y1 = int(tempMove[-1]) - 1
                    print("Player move is")
                    print(tempMove)


                # player1's move should be validated first, and the turn should not pass until
                # player1 makes a valid move
                if self.validMove(x1, y1, 0, 1)[0] == 1:
                    print("move is Valid")
                    validFlag = 1
                else:
                    print("Not a Valid move, try a different one")

            # reprint gameboard, reset flag
            self.printGame()
            validFlag = 0

            while validFlag == 0:
                # Player 2 Movement
                print("Possible O moves in (row,col) format")
                print(self.listofMoves(1))
                player2move = input("Player White(O) input row,col of your current move")

                try:
                    x2 = int(player2move[-3]) - 1

                    y2 = int(player2move[-1]) - 1
                except:
                    print('The input you gave was not in the right format')
                    continue
                if self.validMove(x2, y2, 1, 1)[0] == 1:
                    print("Move is valid")
                    validFlag = 1
                else:
                    print("Not a Valid move, try a different one")

                    # reprint gameboard
            self.printGame()

        scoreX = self.gameScore()[0]
        scoreO = self.gameScore()[1]
        if scoreX == scoreO:
            print("Game is a tie")
        elif scoreX > scoreO:
            print("Player Black Wins with" + str(scoreX) + " Points")
        elif scoreX < scoreO:
            print("Player White Wins with" + str(abs(scoreO)) + " Points")

    # function to print out game matrix
    def printGame(self):
        print('  1 2 3 4 5 6 7 8')
        for x in range(0, 8):
            print(x + 1, end=' ')
            for y in range(0, 8):
                print(self.Matrix[x][y], end=' ')
            print()

    ############## end GameType Functions ###################

    def alpha_beta_search(self):

        '''
            Builds the MinMax tree

            param
            type:

            return:


            '''

        CalcMatrix = deepcopy(self.Matrix)
        v, moves_with_values = self.maxValue(CalcMatrix, -np.inf, +np.inf, 0)
        #print(v, moves_with_values)
        a = max(moves_with_values, key=moves_with_values.get)  # takes the last maximum

        return a

    def maxValue(self, calculationMatrix, alpha, beta, depth):

        if self.endOfGame(calculationMatrix):
            weight = self.utility(calculationMatrix, endPoint=True)
            return weight, weight
        if depth == 4:
            weight = int(self.utility(calculationMatrix, endPoint=True))
            #   print(weight)
            return weight, weight
        v = -np.inf
        print(calculationMatrix)
        possible_moves = self.listofMoves(self.computer, calculationMatrix)
        possible_moves[:] = [(z[0] - 1, z[1] - 1) for z in possible_moves]
        print(possible_moves)
        # print('possible_moves_max',possible_moves)  # Computer is 0 -> should be covert by a global value
        moves_with_values = {}
        for x in possible_moves:
            # print('Matrix',Matrix)
            # print(x)
            # print(self.validMove( x[0],x[1], self.computer, 1, Matrix)[1])

            v = max(v, self.minValue(self.validMove(x[0], x[1], self.computer, 1, calculationMatrix)[1], alpha, beta,
                                     (depth + 1)))  # changing the borad
            # print(v)
            moves_with_values[x] = v
            if v >= beta:
                return v, v
            alpha = max(alpha, v)
            # print('alpha', alpha)
        return v, moves_with_values

    def minValue(self,calculationMatrix, alpha, beta, depth):
        #  print(Matrix)
        if self.endOfGame(calculationMatrix):
            weight = self.utility(calculationMatrix, endPoint=True)
            return weight
        v = np.inf
        possible_moves = self.listofMoves(self.user, calculationMatrix)
        possible_moves[:] = [(y[0] - 1, y[1] - 1) for y in possible_moves]
        #  print('possible_moves_min',possible_moves)
        for x in possible_moves:
            #  print('MinMatrix',Matrix)
            # print(x)
            v = min(v, self.maxValue(self.validMove(x[0], x[1], self.user, 1, calculationMatrix)[1], alpha, beta, (depth + 1))[0])
            # print('v',v,'alpha',alpha)
            if v <= alpha:
                return v

            beta = np.min([beta, v])
        #    print('beta',beta)
        return v

        # Heuristic depending on the progress of the game

    def utility(self, calculationMatrix, endPoint=False):

        # if endPoint=True or terminal_test(Matrix):  # We need a check wheater it's an Endpoint or not
        weight = self.coin_parity(calculationMatrix)

        # elif amount of tiles <= 20: #global variable which counts the amount of tiles
        #   weight = stability(Matrix)+mobility(Matrix)   #
        # else:
        #    weight = corner(Matrix) + stability(Matrix)

        return weight

    # Heuristic functions
    def coin_parity(self, calculationMatrix):
        MaxPlayer = np.count_nonzero(calculationMatrix == self.computercolor)  # Amount of coins
        # print('Maxplayer',MaxPlayer)
        MinPlayer = np.count_nonzero(calculationMatrix == self.usercolor)
        # print('MinPlayer',MinPlayer)
        return 100 * (MaxPlayer - MinPlayer) / (MaxPlayer + MinPlayer)

