import chess
import random

import chess.polyglot


#Thanks for Source suggest idea and use for improve from https://github.com/PaulJeFi/ramses-chess/blob/main/src/evaluation.py

SKILL = 20
GRAIN = 10
GRAIN = int(max(1, -GRAIN, GRAIN))

# VALUE[chess.PieceType] = (EG_PSQT, MG_PSQT, MG, EG, phase)
VALUES = {
	chess.PAWN: ([
	0,   0,   0,   0,   0,   0,  0,   0,
	98, 134,  61,  95,  68, 126, 34, -11,
	-6,   7,  26,  31,  65,  56, 25, -20,
	-14,  13,   6,  21,  23,  12, 17, -23,
	-27,  -2,  -5,  12,  17,   6, 10, -25,
	-26,  -4,  -4, -10,   3,   3, 33, -12,
	-35,  -1, -20, -23, -15,  24, 38, -22,
	  0,   0,   0,   0,   0,   0,  0,   0],
	
	[
	0,   0,   0,   0,   0,   0,   0,   0,
	178, 173, 158, 134, 147, 132, 165, 187,
	 94, 100,  85,  67,  56,  53,  82,  84,
	 32,  24,  13,   5,  -2,   4,  17,  17,
	 13,   9,  -3,  -7,  -7,  -8,   3,  -1,
	  4,   7,  -6,   1,   0,  -5,  -1,  -8,
	 13,   8,   8,  10,  13,   0,   2,  -7,
	  0,   0,   0,   0,   0,   0,   0,   0], 92, 94, 0),

	chess.KNIGHT: ([
	-167, -89, -34, -49,  61, -97, -15, -107,
	 -73, -41,  72,  36,  23,  62,   7,  -17,
	 -47,  60,  37,  65,  84, 129,  73,   44,
	  -9,  17,  19,  53,  37,  69,  18,   22,
	 -13,   4,  16,  13,  28,  19,  21,   -8,
	 -23,  -9,  12,  10,  19,  17,  25,  -16,
	 -29, -53, -12,  -3,  -1,  18, -14,  -19,
	-105, -21, -58, -33, -17, -28, -19,  -23],
	
	[
	-58, -38, -13, -28, -31, -27, -63, -99,
	-25,  -8, -25,  -2,  -9, -25, -24, -52,
	-24, -20,  10,   9,  -1,  -9, -19, -41,
	-17,   3,  22,  22,  22,  11,   8, -18,
	-18,  -6,  16,  25,  16,  17,   4, -18,
	-23,  -3,  -1,  15,  10,  -3, -20, -22,
	-42, -20, -10,  -5,  -2, -20, -23, -44,
	-29, -51, -23, -15, -22, -18, -50, -64], 337, 281, 1),

	chess.BISHOP: ([
	-29,   4, -82, -37, -25, -42,   7,  -8,
	-26,  16, -18, -13,  30,  59,  18, -47,
	-16,  37,  43,  40,  35,  50,  37,  -2,
	 -4,   5,  19,  50,  37,  37,   7,  -2,
	 -6,  13,  13,  26,  34,  12,  10,   4,
	  0,  15,  15,  15,  14,  27,  18,  10,
	  4,  15,  16,   0,   7,  21,  33,   1,
	-33,  -3, -14, -21, -13, -12, -39, -21],
	
	[
	-14, -21, -11,  -8, -7,  -9, -17, -24,
	 -8,  -4,   7, -12, -3, -13,  -4, -14,
	  2,  -8,   0,  -1, -2,   6,   0,   4,
	 -3,   9,  12,   9, 14,  10,   3,   2,
	 -6,   3,  13,  19,  7,  10,  -3,  -9,
	-12,  -3,   8,  10, 13,   3,  -7, -15,
	-14, -18,  -7,  -1,  4,  -9, -15, -27,
	-23,  -9, -23,  -5, -9, -16,  -5, -17], 365, 297, 1),

	chess.ROOK: ([
	 32,  42,  32,  51, 63,  9,  31,  43,
	 27,  32,  58,  62, 80, 67,  26,  44,
	 -5,  19,  26,  36, 17, 45,  61,  16,
	-24, -11,   7,  26, 24, 35,  -8, -20,
	-36, -26, -12,  -1,  9, -7,   6, -23,
	-45, -25, -16, -17,  3,  0,  -5, -33,
	-44, -16, -20,  -9, -1, 11,  -6, -71,
	-19, -13,   1,  17, 16,  7, -37, -26],
	
	[
	13, 10, 18, 15, 12,  12,   8,   5,
	11, 13, 13, 11, -3,   3,   8,   3,
	 7,  7,  7,  5,  4,  -3,  -5,  -3,
	 4,  3, 13,  1,  2,   1,  -1,   2,
	 3,  5,  8,  4, -5,  -6,  -8, -11,
	-4,  0, -5, -1, -7, -12,  -8, -16,
	-6, -6,  0,  2, -9,  -9, -11,  -3,
	-9,  2,  3, -1, -5, -13,   4, -20], 477, 512, 2),

	chess.QUEEN: ([
	-28,   0,  29,  12,  59,  44,  43,  45,
	-24, -39,  -5,   1, -16,  57,  28,  54,
	-13, -17,   7,   8,  29,  56,  47,  57,
	-27, -27, -16, -16,  -1,  17,  -2,   1,
	 -9, -26,  -9, -10,  -2,  -4,   3,  -3,
	-14,   2, -11,  -2,  -5,   2,  14,   5,
	-35,  -8,  11,   2,   8,  15,  -3,   1,
	 -1, -18,  -9,  10, -15, -25, -31, -50],
	 
	 [
	 -9,  22,  22,  27,  27,  19,  10,  20,
	-17,  20,  32,  41,  58,  25,  30,   0,
	-20,   6,   9,  49,  47,  35,  19,   9,
	  3,  22,  24,  45,  57,  40,  57,  36,
	-18,  28,  19,  47,  31,  34,  39,  23,
	-16, -27,  15,   6,   9,  17,  10,   5,
	-22, -23, -30, -16, -16, -23, -36, -32,
	-33, -28, -22, -43,  -5, -32, -20, -41], 1025, 936, 4),

	chess.KING: ([
	-65,  23,  16, -15, -56, -34,   2,  13,
	 29,  -1, -20,  -7,  -8,  -4, -38, -29,
	 -9,  24,   2, -16, -20,   6,  22, -22,
	-17, -20, -12, -27, -30, -25, -14, -36,
	-49,  -1, -27, -39, -46, -44, -33, -51,
	-14, -14, -22, -46, -44, -30, -15, -27,
	  1,   7,  -8, -64, -43, -16,   9,   8,
	-15,  36,  12, -54,   8, -28,  24,  14],
	
	[
	-74, -35, -18, -18, -11,  15,   4, -17,
	-12,  17,  14,  17,  17,  38,  23,  11,
	 10,  17,  23,  15,  20,  45,  44,  13,
	 -8,  22,  24,  27,  26,  33,  26,   3,
	-18,  -4,  21,  24,  27,  23,   9, -11,
	-19,  -3,  11,  21,  23,  16,   7,  -9,
	-27, -11,   4,  13,  14,   4,  -5, -17,
	-53, -34, -21, -11, -28, -14, -24, -43], 0, 0, 0)}

WPAWN   = chess.Piece(chess.PAWN,   chess.WHITE)
WKNIGHT = chess.Piece(chess.KNIGHT, chess.WHITE)
WBISHOP = chess.Piece(chess.BISHOP, chess.WHITE)
WROOK   = chess.Piece(chess.ROOK,   chess.WHITE)
WQUEEN  = chess.Piece(chess.QUEEN,  chess.WHITE)
WKING   = chess.Piece(chess.KING,   chess.WHITE)
BPAWN   = chess.Piece(chess.PAWN,   chess.BLACK)
BKNIGHT = chess.Piece(chess.KNIGHT, chess.BLACK)
BBISHOP = chess.Piece(chess.BISHOP, chess.BLACK)
BROOK   = chess.Piece(chess.ROOK,   chess.BLACK)
BQUEEN  = chess.Piece(chess.QUEEN,  chess.BLACK)
BKING   = chess.Piece(chess.KING,   chess.BLACK)

ISOLATED_MASK = [
	chess.BB_FILE_A,
	chess.BB_FILE_A | chess.BB_FILE_C,
	chess.BB_FILE_B | chess.BB_FILE_D,
	chess.BB_FILE_C | chess.BB_FILE_E,
	chess.BB_FILE_D | chess.BB_FILE_F,
	chess.BB_FILE_E | chess.BB_FILE_G,
	chess.BB_FILE_F | chess.BB_FILE_H,
	chess.BB_FILE_G
]

MOP_UP_VALUES = {
	None: 0,
	chess.Piece(chess.PAWN, chess.WHITE): 100,
	chess.Piece(chess.BISHOP, chess.WHITE): 300,
	chess.Piece(chess.KNIGHT, chess.WHITE): 300,
	chess.Piece(chess.ROOK, chess.WHITE): 500,
	chess.Piece(chess.QUEEN, chess.WHITE): 900,
	chess.Piece(chess.KING, chess.WHITE): 0,
	chess.Piece(chess.PAWN, chess.BLACK): -100,
	chess.Piece(chess.KNIGHT, chess.BLACK): -300,
	chess.Piece(chess.BISHOP, chess.BLACK): -300,
	chess.Piece(chess.ROOK, chess.BLACK): -500,
	chess.Piece(chess.QUEEN, chess.BLACK): -900,
	chess.Piece(chess.KING, chess.BLACK): 0
}

def mop_up(board: chess.Board) -> int:
	material = [0, 0]  # material score for [WHITE, BLACK]
	for square in chess.SquareSet(board.occupied):
		piece = board.piece_at(square)
		value = MOP_UP_VALUES[piece]
		if value > 0:
			material[0] += value
		else:
			material[1] -= value

	if material[0] == material[1]:
		return 0

	winner = 1 if material[0] > material[1] else -1
	return winner * (chess.square_distance(board.king(chess.WHITE if winner == 1 else chess.BLACK), chess.E4)
					 + 4 * (14 - chess.square_manhattan_distance(board.king(chess.WHITE), board.king(chess.BLACK))))

def evaluate_pawn(board: chess.Board, square: chess.Square, pawns: chess.SquareSet, is_white: bool):
	file_ = chess.square_file(square)
	rank = chess.square_rank(square)
	#check if no pawn block front or adjacent 
	def is_passed_pawn():
		opponent_pawns = board.pieces(chess.PAWN, not is_white)

		#FILE ON LEFT or RIGHT OF PIECE PAWN
		adjacent_files_mask = chess.BB_FILES[file_]
		if file_ > 0:
			adjacent_files_mask |= chess.BB_FILES[file_ - 1]
		if file_ < 7:
			adjacent_files_mask |= chess.BB_FILES[file_ + 1]

		# All rank front of pawn
		blocking_pawns_mask = 0
		if is_white:
			for r in range(rank + 1, min(rank + 7, 8)):
				blocking_pawns_mask |= chess.BB_RANKS[r]
		else:
			for r in range(max(rank - 1, 0), max(rank - 7, -1), -1):
				blocking_pawns_mask |= chess.BB_RANKS[r]
		#bitboard (all - not column left and right - row same of behind pawn position)
		opponent_blockers = opponent_pawns & adjacent_files_mask & blocking_pawns_mask
		
		# If there are no opponent pawns in the path, it's a passed pawn
		return opponent_blockers == 0

	MG, EG = 0, 0
	file_BB = chess.BB_FILES[file_]
	opponent_pawns = board.pieces(chess.PAWN, chess.BLACK if is_white else chess.WHITE)

	# doubled pawns
	if chess.popcount(int(pawns & file_BB)) >= 1:
		MG -= 10
		EG -= 20

	# isolated pawn
	if int(pawns & ISOLATED_MASK[file_]) == 0:
		MG -= 10
		EG -= 20

	 # Backward pawns
	if not (pawns & chess.BB_FILES[file_]) & (chess.BB_RANKS[rank + 1] if is_white else chess.BB_RANKS[rank - 1]):
		adjacent_files = 0
		if file_ > 0:  # Left file exists
			adjacent_files |= chess.BB_FILES[file_ - 1]
		if file_ < 7:  # Right file exists
			adjacent_files |= chess.BB_FILES[file_ + 1]
		
		if opponent_pawns & adjacent_files & (chess.BB_RANKS[rank + 1] if is_white else chess.BB_RANKS[rank - 1]):
			MG -= 15
			EG -= 25
			# score -= 8

	# Pawn chains
	chain_exists = False
	
	# Check left diagonal if file exists
	if file_ > 0:
		if pawns & chess.BB_FILES[file_ - 1] & (chess.BB_RANKS[rank + 1] if is_white else chess.BB_RANKS[rank - 1]):
			chain_exists = True
			
	# Check right diagonal if file exists
	if file_ < 7:
		if pawns & chess.BB_FILES[file_ + 1] & (chess.BB_RANKS[rank + 1] if is_white else chess.BB_RANKS[rank - 1]):
			chain_exists = True

	if chain_exists:
		MG += 10
		EG += 15
	
	if is_passed_pawn():
	# Calculate the advancement bonus
		if is_white:
			advancement = rank - 1  #white increase rank
		else:
			advancement = 6 - rank  #black decrease brank because of flip

		# closer to promote is higher score
		bonus1 = advancement * 20  #score on middle game
		bonus2 = advancement * 40  #endgame score
		MG += bonus1
		EG += bonus2

		# Extra bonus for passed pawns close to promotion (6th and 7th ranks for white, 3rd and 2nd for black)
		if (is_white and rank >= 6) or ((not is_white) and rank <= 3):
			MG += 40
			EG += 70  

		# Bonus for supported passed pawns
	if is_white:
		# Check if supporting squares are valid (within board)
		support_squares = []
		if square >= 8:  # Can have support from below
			support_squares.append(square - 8)  # directly below
			if square % 8 > 0:  # Not on a-file
				support_squares.append(square - 9)  # diagonal left
			if square % 8 < 7:  # Not on h-file
				support_squares.append(square - 7)  # diagonal right
				
		for support_sq in support_squares:
			if board.piece_at(support_sq) == WPAWN:
				MG += 15
				EG += 25
				break
	else:
		# Check if supporting squares are valid (within board)
		support_squares = []
		if square <= 55:  # Can have support from above
			support_squares.append(square + 8)  # directly above
			if square % 8 > 0:  # Not on a-file
				support_squares.append(square + 7)  # diagonal left
			if square % 8 < 7:  # Not on h-file
				support_squares.append(square + 9)  # diagonal right
				
		for support_sq in support_squares:
			if board.piece_at(support_sq) == BPAWN:
				MG += 15
				EG += 25
				break

	return MG, EG

def evaluate_knight(board: chess.Board, square: chess.Square, is_white: bool):
	MG, EG = 0, 0
	score = 0
	
	# Decreasing value as pawns disappear (knights are better with more pawns on board)
	pawn_count = chess.popcount(board.pawns)
	MG -= 5 * (16-pawn_count)
	EG -= 8 * (16-pawn_count)

	# Mobility excluding squares controlled by enemy pawns
	enemy_pawn_attacks = 0
	enemy_pawns = board.pieces(chess.PAWN, not is_white)
	for p_square in chess.SquareSet(enemy_pawns):
		if is_white:
			if chess.square_file(p_square) > 0:
				enemy_pawn_attacks |= chess.BB_SQUARES[p_square - 9]
			if chess.square_file(p_square) < 7:
				enemy_pawn_attacks |= chess.BB_SQUARES[p_square - 7]
		else:
			if chess.square_file(p_square) > 0:
				enemy_pawn_attacks |= chess.BB_SQUARES[p_square + 7]
			if chess.square_file(p_square) < 7:
				enemy_pawn_attacks |= chess.BB_SQUARES[p_square + 9]
	
	safe_mobility = chess.popcount(board.attacks_mask(square) & ~enemy_pawn_attacks)
	MG += safe_mobility * 8
	
	trapped_squares = {
		chess.A1, chess.H1, chess.A2, chess.H2,  # White's trapped squares
		chess.A8, chess.H8, chess.A7, chess.H7   # Black's trapped squares
	}
	if square in trapped_squares:
		MG -= 100
		EG -= 100
	
	# Penalty for blocking c-pawn in closed positions
	if (is_white and square == chess.C3 and 
		board.piece_at(chess.C2) == WPAWN and 
		board.piece_at(chess.D4) == WPAWN and 
		not board.piece_at(chess.E4)):
		MG -= 30
	elif (not is_white and square == chess.C6 and 
		  board.piece_at(chess.C7) == BPAWN and 
		  board.piece_at(chess.D5) == BPAWN and 
		  not board.piece_at(chess.E5)):
		MG += 30
	
	# Outpost evaluation
	if is_white:
		pawn_attack_squares = 0
		support_squares = 0
		
		# Check forward-left attack (if not on a-file)
		if chess.square_file(square) > 0 and square + 7 < 64:
			pawn_attack_squares |= chess.BB_SQUARES[square + 7]
		# Check forward-right attack (if not on h-file)
		if chess.square_file(square) < 7 and square + 9 < 64:
			pawn_attack_squares |= chess.BB_SQUARES[square + 9]
			
		# Check backward-left support (if not on a-file)
		if chess.square_file(square) > 0 and square - 9 >= 0:
			support_squares |= chess.BB_SQUARES[square - 9]
		# Check backward-right support (if not on h-file)
		if chess.square_file(square) < 7 and square - 7 >= 0:
			support_squares |= chess.BB_SQUARES[square - 7]
	else:
		pawn_attack_squares = 0
		support_squares = 0
		
		# Check forward-left attack (if not on a-file)
		if chess.square_file(square) > 0 and square - 9 >= 0:
			pawn_attack_squares |= chess.BB_SQUARES[square - 9]
		# Check forward-right attack (if not on h-file)
		if chess.square_file(square) < 7 and square - 7 >= 0:
			pawn_attack_squares |= chess.BB_SQUARES[square - 7]
			
		# Check backward-left support (if not on a-file)
		if chess.square_file(square) > 0 and square + 7 < 64:
			support_squares |= chess.BB_SQUARES[square + 7]
		# Check backward-right support (if not on h-file)
		if chess.square_file(square) < 7 and square + 9 < 64:
			support_squares |= chess.BB_SQUARES[square + 9]

	# Check if square cannot be attacked by enemy pawns
	if not (pawn_attack_squares & int(enemy_pawns)):
		friendly_pawns = board.pieces(chess.PAWN, is_white)
		if support_squares & int(friendly_pawns):  # Protected by friendly pawn
			MG += 25  # Bonus for protected outpost
			EG += 15
		else:
			MG += 10  # Regular outpost
			EG += 20
	
	# Bonus for knight defended by pawn
	if support_squares & int(board.pieces(chess.PAWN, is_white)):
		MG += 10
		EG += 5
	
	# Penalty for undefended minor piece
	if not board.attackers_mask(is_white, square):
		MG -= 15
		EG -= 10
	
	# Central control bonus
	central_squares = {chess.D4, chess.E4, chess.D5, chess.E5}
	if square in central_squares:
		MG += 25
		EG += 15
	
	# Bonus for knights near enemy king in middlegame
	enemy_king_square = board.king(not is_white)
	distance_to_king = chess.square_distance(square, enemy_king_square)
	MG += (8 - distance_to_king) * 3

	return MG, EG


def evaluate_rook(board: chess.Board, square: chess.Square, is_white: bool) -> int:
	score = 0

	# setup bitboard mask of file and rank at position square
	rank = chess.square_rank(square)
	file = chess.square_file(square)
	file_mask = chess.BB_FILES[file]
	rank_mask = chess.BB_RANKS[rank]

	# Open file (no pawns on the file front of)
	if int(board.pawns & file_mask) == 0:
		score += 40  # Bonus for open file
	# Half-open file (no friendly pawns on the file front of)
	elif int(board.pieces(chess.PAWN, is_white) & file_mask) == 0:
		score += 20  # Bonus for half-open file

	# # Mobility - count legal moves the rook can make
	# rook_moves = list(board.legal_moves)
	# rook_mobility = len([move for move in rook_moves if move.from_square == square])
	# score += rook_mobility * 2  # Each legal move adds to mobility score

	# Rook on the 7th or 8th rank is powerful
	if (is_white and rank == 6) or (not is_white and rank == 1):
		score += 30  # Strong position for white on 7th rank (rank 6), for black on rank 2 (rank 1)
	if (is_white and rank == 7) or (not is_white and rank == 0):
		score += 40  # Rook on 8th rank (final rank) is very strong

	# Rook connection (two rooks on the same rank or file)
	rooks = board.pieces(chess.ROOK, is_white)
	countFiles = chess.popcount(rooks & file_mask)
	countRanks = chess.popcount(rooks & rank_mask)
	if countFiles==2 or countRanks==2: #check if 2 rocks on same rank or file
		score += 25

	#King has castling but dont use so penalty for Rook
	if is_white:
		if board.has_kingside_castling_rights(True) and square==chess.H1:
			score -= 40
		elif board.has_queenside_castling_rights(True) and square==chess.A1:
			score -= 20
	else:
		if board.has_kingside_castling_rights(False) and square==chess.H8:
			score -= 40
		elif board.has_queenside_castling_rights(False) and square==chess.A8:
			score -= 20

	#Bonus for Rook when same file with enemy Queen
	enemyQueen = board.pieces(chess.QUEEN, not is_white)
	countQueenFiles = chess.popcount(enemyQueen & file_mask)
	if countQueenFiles > 0:
		score += 15

	# Increasing power Rook as many pawns disappear
	pawn_factor = max(0, 16 - chess.popcount(board.pawns)) 
	score += pawn_factor * 15  

	return score

def evaluate_bishop(board: chess.Board, square: chess.Square, is_white: bool):
	MG, EG = 0, 0
	
	# Get bishop's square color (light or dark)
	is_dark_squared = (square + (square // 8)) % 2 == 1
	
	# Bad Bishop evaluation
	def evaluate_bad_bishop():
		friendly_pawns = board.pieces(chess.PAWN, is_white)
		MG, EG = 0, 0
		# Count pawns on same colored squares as bishop
		bad_pawns = 0
		for pawn_square in chess.SquareSet(friendly_pawns):
			is_pawn_dark = (pawn_square + (pawn_square // 8)) % 2 == 1
			if is_pawn_dark == is_dark_squared:
				bad_pawns += 1
				
				# Extra penalty if pawns are blocked
				if is_white:
					if board.piece_at(pawn_square + 8):  # Square above is occupied
						MG -= 5
						EG -= 8
				else:
					if board.piece_at(pawn_square - 8):  # Square below is occupied
						MG -= 5
						EG -= 8
		MG -= bad_pawns * 5  # Base penalty for each pawn on same color
		EG -= bad_pawns * 8  
			
		return MG, EG
	
	# Bishop Pair bonus
	def evaluate_bishop_pair():
		friendly_bishops = chess.popcount(board.pieces(chess.BISHOP, is_white))
		if friendly_bishops == 2:
			return 50, 100  # Standard bonus for bishop pair
		return 0, 0
	
	# Bishop versus Knight evaluation
	def evaluate_bishop_vs_knight():
		MG, EG = 0, 0
		pawn_count = chess.popcount(board.pawns)
		
		# Bishops generally better in open positions (fewer pawns)
		if pawn_count < 8:  # Very open position
			MG += 20
			EG += 40
		elif pawn_count < 12:  # Somewhat open position
			MG += 10
			EG += 20
		# Check diagonal mobility
		mobility = chess.popcount(board.attacks_mask(square))
		if mobility > 7:  # Bishop has good mobility
			MG += 15
			EG += 30
		return MG, EG
	
	# Color Weakness evaluation
	def evaluate_color_weakness():
		MG, EG = 0, 0		
		enemy_bishops = board.pieces(chess.BISHOP, not is_white)
		
		if enemy_bishops:
			enemy_bishop_square = next(iter(enemy_bishops))
			is_enemy_dark = (enemy_bishop_square + (enemy_bishop_square // 8)) % 2 == 1
			
			# Penalty for bishops of same color
			if is_enemy_dark == is_dark_squared:
				MG -= 20
				EG -= 40
				
				# Extra penalty if we have no knights to complement
				if not board.pieces(chess.KNIGHT, is_white):
					MG -= 15
					EG -= 30
		
		return MG, EG
	
	# Calculate all components
	MGC1, EGC1 = evaluate_bad_bishop()
	
	MGC2, EGC2 = evaluate_bishop_pair()
	
	MGC3, EGC3 = evaluate_bishop_vs_knight()
	
	MGC4, EGC4 = evaluate_color_weakness()
	
	MG = MGC1 + MGC2 + MGC3 + MGC4
	EG = EGC1 + EGC2 + EGC3 + EGC4

	# Penalty for undefended minor piece
	if not board.attackers_mask(is_white, square):
		MG -= 15
		EG -= 10
	
	# Mobility bonus
	mobility = chess.popcount(board.attacks_mask(square))
	MG += mobility * 5
	EG += mobility * 3
	
	# Bonus for controlling long diagonals
	long_diagonals = {
		chess.A1, chess.B2, chess.C3, chess.D4, chess.E5, chess.F6, chess.G7, chess.H8,
		chess.H1, chess.G2, chess.F3, chess.E4, chess.D5, chess.C6, chess.B7, chess.A8
	}
	if square in long_diagonals:
		MG += 15
		EG += 10
		
	# Bonus for attacking center squares
	center_attack = board.attacks_mask(square) & (
		chess.BB_SQUARES[chess.D4] | chess.BB_SQUARES[chess.E4] |
		chess.BB_SQUARES[chess.D5] | chess.BB_SQUARES[chess.E5]
	)
	if center_attack:
		MG += 20
		EG += 10
	
	return MG, EG

def evaluate_queen(board: chess.Board, square: chess.Square, is_white: bool):
	MG, EG = 0, 0
	
	# Penalty for early development
	# Penalize if queen is developed before minor pieces
	if is_white:
		if board.piece_at(chess.B1) != WKNIGHT or board.piece_at(chess.G1) != WKNIGHT:
			MG -= 40
		if board.piece_at(chess.C1) != WBISHOP or board.piece_at(chess.F1) != WBISHOP:
			MG -= 50
	else:
		if board.piece_at(chess.B8) != BKNIGHT or board.piece_at(chess.G8) != BKNIGHT:
			MG += 40
		if board.piece_at(chess.C8) != BBISHOP or board.piece_at(chess.F8) != BBISHOP:
			MG += 50

	# Queen mobility dont use so that use king tropism
	enemy_king_square = board.king(not is_white)
	distance_to_king = chess.square_distance(square, enemy_king_square)
	MG += (8 - distance_to_king) * 15
	EG += (8 - distance_to_king) * 10

	# Penalty for undefended piece
	if not board.attackers_mask(is_white, square):
		MG -= 30
		EG -= 40

	return MG, EG

def evaluate_king_safe(board: chess.Board, isWhite: bool) -> int:
	#find pos of pawn shield king
	def cal_pawn_shield_pos(color, rank, king_bb):
		pawn_shield_mask = 0
		if color == chess.WHITE:
			if rank < 7:  
				pawn_shield_mask = chess.shift_up(king_bb) | chess.shift_up_left(king_bb) | chess.shift_up_right(king_bb)
		else:  # black
			if rank > 0:
				pawn_shield_mask = chess.shift_down(king_bb) | chess.shift_down_left(king_bb) | chess.shift_down_right(king_bb)
		return pawn_shield_mask
	color = isWhite
	king_square = board.king(color)
	king_safety_score = 0
	king_bb = chess.BB_SQUARES[king_square]
	
	#evaluate king not near corner
	rank, file = divmod(king_square, 8)
	center_bonus = max(3 - abs(3.5 - rank), 3 - abs(3.5 - file))  # Kings closer to the edge get less bonus
	king_safety_score += center_bonus * 10

	# Evaluate the pawn shield
	pawn_shield_score = 0
	pawn_positions_mask = cal_pawn_shield_pos(color, rank, king_bb)
	pawns_bb = board.pieces(chess.PAWN, color)
	pawns_in_shield =  pawns_bb & pawn_positions_mask #return list bitboard 1 if in 3 pos front of King has pawn same color
	total_shield = chess.popcount(pawns_in_shield) #count number bit 1 
	pawn_shield_score += 15 * total_shield
	pawn_shield_score -= 10 * (3 - total_shield) 
	king_safety_score += pawn_shield_score

	# Piece proximity evaluation (if chess same team near by King or != color)
	king_zone_mask = board.attacks_mask(king_square)
	king_safety_score += 5 * chess.popcount(king_zone_mask & board.occupied_co[color])

	# Open file penalty
	file_mask = chess.BB_FILES[chess.square_file(king_square)]
	if not (board.occupied_co[isWhite] & file_mask):
		king_safety_score -= 20

	#King zone being attacked

	board.turn = not board.turn
	attacker = board.occupied_co[not color]
	attack_count = len( [move for move in board.generate_legal_moves(attacker, king_zone_mask)] )

	board.turn = not board.turn
	defender = board.occupied_co[color] & ~chess.BB_SQUARES[king_square] #remove king not count as a piece for defend
	def_count = len( [move for move in board.generate_legal_moves(defender, king_zone_mask)] )

	king_safety_score -= attack_count*15
	king_safety_score += def_count*10
	# above code is using bitboard improve from below old code
	# for square in king_zone:
	# 	attack_count += chess.popcount(board.attackers_mask(not color, square))
	# 	def_count += chess.popcount(board.attackers_mask(color, square))
	

	#King Tropism
	opponent_queens = board.pieces(chess.QUEEN, not color)
	opponent_knights = board.pieces(chess.KNIGHT, not color)
	opponent_bioshops = board.pieces(chess.BISHOP, not color)
	opponent_rooks = board.pieces(chess.ROOK, not color)
	for square in opponent_queens | opponent_knights | opponent_bioshops | opponent_rooks:
		distance = chess.square_distance(king_square, square)
		penalty = 5
		if square in opponent_queens:
			penalty = 15
		elif square in opponent_rooks:
			penalty = 10
		elif square in opponent_bioshops:
			penalty = 7
		king_safety_score -= penalty * (8 - distance)  # Closer is more dangerous

	# King's position evaluation castling
	if board.has_kingside_castling_rights(color) or board.has_queenside_castling_rights(color):
		king_safety_score += 20  # Bonus for having castling rights

	#check phase endgame or middle|open the value king safe is less important than at late game
	#phase = 32 = total number pieces on board
	# 27 < x <=32 is opengame (full score)
	# 16 <= x <= 27 is middlegame (reduce % per 1)
	# < 16 endgame (always = 0, score defend king is not important)
	phase_total = chess.popcount(board.occupied)
	phase_total = max(0, min(1, (phase_total - 16) / 11))
	king_safety_score = int(king_safety_score * (1 - phase_total))
	return king_safety_score

#trapped on fix
# def evaluate_trapped_piece(board, piece, square):
# 	if piece in {chess.PAWN, chess.KING.}
# 		return 0

#     legal_moves = list(board.legal_moves)
#     piece_moves = [move for move in legal_moves if move.from_square == square]
	
#     if len(piece_moves) == 0 and piece.piece_type:
#     	return - VALUES[]
#     elif len(piece_moves) < 3 and piece.piece_type != ches.KNIGHT:
#     	return -50
#     elif len(piece_moves) <

# def evaluate_trapped_piece(board: chess.Board, piece: chess.Piece, square: chess.Square) -> int:
# 	if not any(board.attacks(square) & board.legal_moves):
# 		# The piece is trapped if it has no legal moves
# 		return -150  # Arbitrary penalty for trapped piece
# 	return 0

# def evaluate_mobility(board: chess.Board, square : chess.Square, color: bool):
# 	# Mobility
# 	score = 2*chess.popcount(board.attacks_mask(square) & ~board.occupied_co[color]) (~ = remove)
# 	return score

def evaluate_piece(board: chess.Board, piece: chess.Piece, square: chess.Square, w_pawns: chess.SquareSet, b_pawns: chess.SquareSet, is_white: bool) -> int:
	MG = 0
	EG = 0
	score = 0
	if piece.piece_type == chess.PAWN:
		MG, EG = evaluate_pawn(board, square, w_pawns if is_white else b_pawns, is_white)
	elif piece.piece_type == chess.ROOK:
		score = evaluate_rook(board, square, is_white)
	elif piece.piece_type == chess.BISHOP:
		MG, EG = evaluate_bishop(board, square, is_white) #====
	elif piece.piece_type == chess.QUEEN:
		MG, EG = evaluate_queen(board, square, is_white) #====
	elif piece.piece_type == chess.KING:
		score = evaluate_king_safe(board, is_white)
		score += evaluate_escape_of_king(board, is_white)
	elif piece.piece_type == chess.KNIGHT:
		MG, EG = evaluate_knight(board, square, is_white)
	#score += evaluate_trapped_piece(board, piece, square)
	return MG, EG, score

central_squares = {chess.D4, chess.E4, chess.D5, chess.E5}
semi_central_squares = {chess.C3, chess.C4, chess.C5, chess.C6, chess.F3, chess.F4, chess.F5, chess.F6}

def evaluate_positional(board: chess.Board) -> int:
	#eval kngiht, bishop, pawn | QUEEN and ROOK dont use because of variaties VERTICAL or HORIZONTAL
	# Minor pieces developed
	MG = 0
	if board.piece_at(chess.B1) != WKNIGHT: MG += 25
	if board.piece_at(chess.C1) != WBISHOP: MG += 25
	if board.piece_at(chess.F1) != WBISHOP: MG += 25
	if board.piece_at(chess.G1) != WKNIGHT: MG += 25
	if board.piece_at(chess.B8) != BKNIGHT: MG -= 25
	if board.piece_at(chess.C8) != BBISHOP: MG -= 25
	if board.piece_at(chess.F8) != BBISHOP: MG -= 25
	if board.piece_at(chess.G8) != BKNIGHT: MG -= 25

	# Trapped bishop
	if board.piece_at(chess.A7) == WBISHOP  and  board.piece_at(chess.B6) == BPAWN: MG -= 120
	if board.piece_at(chess.H7) == WBISHOP  and  board.piece_at(chess.G6) == BPAWN: MG -= 120
	if board.piece_at(chess.A2) == BBISHOP  and  board.piece_at(chess.B3) == WPAWN: MG += 120
	if board.piece_at(chess.H2) == BBISHOP  and  board.piece_at(chess.G3) == WPAWN: MG += 120

	# Central pawn control
	for square in central_squares:
		if board.piece_at(square) and board.piece_at(square).piece_type == chess.PAWN:
			if board.piece_at(square).color == chess.WHITE:
				MG += 15  
			else:
				MG -= 15 
 
	# SEMI Central pawn control 
	for square in semi_central_squares:
		if board.piece_at(square) and board.piece_at(square).piece_type == chess.PAWN:
			if board.piece_at(square).color == chess.WHITE:
				MG += 10  # Bonus for controlling semi-central squares
			else:
				MG -= 10

	# Undevelopped central pawn
	if board.piece_at(chess.E2) == WPAWN  and  board.piece_at(chess.E3) != None: MG -= 15
	if board.piece_at(chess.D2) == WPAWN  and  board.piece_at(chess.D3) != None: MG -= 15
	if board.piece_at(chess.E7) == BPAWN  and  board.piece_at(chess.E6) != None: MG += 15
	if board.piece_at(chess.D7) == BPAWN  and  board.piece_at(chess.D6) != None: MG += 15

	return MG

def evaluate_king_pawn_tropism_endgame(board: chess.Board, color: bool):
	king_square = board.king(color)
	opponent_pawns = board.pieces(chess.PAWN, not color)
	tropism_score = 0
	for pawn_square in opponent_pawns:
		distance = chess.square_manhattan_distance(king_square, pawn_square)
		if distance <= 3:
			tropism_score += (8 - distance) * 15  #closer = bonus
		else:
			tropism_score -= distance * 5  #farer = penaly

	return tropism_score

def evaluate_force_enemy_king_corner_endgame(board: chess.Board, color: bool):
	#evaluate king near corner
	board.pin
	def score_center(king_square):
		rank = chess.square_rank(king_square)
		file = chess.square_file(king_square)
		near_corner_bonus = abs(3.5 - rank) + abs(3.5 - file) 
		return near_corner_bonus * 20
	
	#if enemy king near corner => bonus for my king
	return score_center(board.king(not color))
	
def evaluate_escape_of_king(board: chess.Board, color: bool):
	king_square = board.king(color)
	legal_move_king = len( [move for move in board.generate_legal_moves(chess.BB_SQUARES[king_square])] )
	return legal_move_king*20

def evaluate_penalty_king_being_xray_attack(board: chess.Board, square):
	score = 0
	penalty = 35
	if board.is_pinned(chess.WHITE, square):
		score -= penalty
	if board.is_pinned(chess.BLACK, square):
		score += penalty
	return score

def skill(value: float) -> int:
	if SKILL == 20:
		return (int(value) // GRAIN) * GRAIN
	return int(((value * SKILL) // (20 - SKILL + GRAIN)) * (20 - SKILL + GRAIN) / 20 + ((20 - SKILL) * random.random() * (SKILL - 20) * 200 - (SKILL - 20) * 100) / 20)

def evaluate_board(board: chess.Board) -> int:
	s2m = 1 if board.turn else -1
	late_eg = 0
	connectivity_count = 0
	score, phase, EG, MG = 0, 24, 0, 0
	w_pawns = board.pieces(chess.PAWN, chess.WHITE)
	b_pawns = board.pieces(chess.PAWN, chess.BLACK)

	#chess.SquareSet(board.occupied)
	#board.piece_map().keys()
	for square in chess.SquareSet(board.occupied):
		piece = board.piece_at(square)
		if piece is not None:
			phase -= VALUES[piece.piece_type][4]
			#xray attack if 1 piece cant move because shield King so penalty
			# bonus for Black if White is being Pin xray
			# bonus for White if Black is being Pin xray
			score += evaluate_penalty_king_being_xray_attack(board, square)
			if piece.color == chess.WHITE:
				MG += VALUES[piece.piece_type][0][chess.SQUARES_180[square]] + VALUES[piece.piece_type][2]
				EG += VALUES[piece.piece_type][1][chess.SQUARES_180[square]] + VALUES[piece.piece_type][3]
				ascore, AMG, AEG = evaluate_piece(board, piece, square, w_pawns, b_pawns, True)
				score += ascore
				MG += AMG
				EG += AEG
				connectivity_count += len(board.attackers(chess.WHITE, square))
			else:													#material score
				MG -= VALUES[piece.piece_type][0][square] + VALUES[piece.piece_type][2]
				EG -= VALUES[piece.piece_type][1][square] + VALUES[piece.piece_type][3]
				ascore, AMG, AEG = evaluate_piece(board, piece, square, w_pawns, b_pawns, False)
				score -= ascore
				MG -= AMG
				EG -= AEG
				connectivity_count -= len(board.attackers(chess.BLACK, square))

	# end game bonus score for force King enemy to corner
	# condition: pawn total on board  < 7
	# total value of chess (Queen 4, Knight 1, Rook 2, Bioshop 1) < 8 (24)
	if chess.popcount(board.pawns) < 7 and phase < 8:
		late_eg += mop_up(board)

	# when endgame 
	# + increase score when king near to enemy pawn
	# + increase score if enemy king near corner
	if phase < 9:
		late_eg += evaluate_king_pawn_tropism_endgame(board, chess.WHITE)
		late_eg -= evaluate_king_pawn_tropism_endgame(board, chess.BLACK)
		late_eg += evaluate_force_enemy_king_corner_endgame(board, chess.WHITE)
		late_eg -= evaluate_force_enemy_king_corner_endgame(board, chess.BLACK)

	# General positional eval (a type enchance from eval_mobility for some kind control)
	MG += evaluate_positional(board)
	MG += connectivity_count * 25
	EG += connectivity_count * 10

	# Tapered eval calc for opening/middle score + endgame score = score
	phase = (phase * 256 + 12) / 24
	score += ((MG * (256 - phase)) + (EG * phase)) / 256

	# #total space move can make
	# #if it blocked the space move will be less
	# score += evaluate_mobility(board)

	#Score connected by piece side (total move connect that piece can defend each other)
	score += connectivity_count * 10

	return skill(s2m * (score + late_eg))