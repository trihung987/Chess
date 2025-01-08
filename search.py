from enum import Enum
import chess
import evaluations
import time
import random
import prediction

# MVV_LVA table
MVV_LVA = [
	[0, 0, 0, 0, 0, 0, 0],  # victim K, attacker K, Q, R, B, N, P, None
	[50, 51, 52, 53, 54, 55, 0],  # victim Q, attacker K, Q, R, B, N, P, None
	[40, 41, 42, 43, 44, 45, 0],  # victim R, attacker K, Q, R, B, N, P, None
	[30, 31, 32, 33, 34, 35, 0],  # victim B, attacker K, Q, R, B, N, P, None
	[20, 21, 22, 23, 24, 25, 0],  # victim N, attacker K, Q, R, B, N, P, None
	[10, 11, 12, 13, 14, 15, 0],  # victim P, attacker K, Q, R, B, N, P, None
	[0, 0, 0, 0, 0, 0, 0],  # victim None, attacker K, Q, R, B, N, P, None
]

def mvv_lva(board: chess.Board, move: chess.Move):
	if board.is_en_passant(move):
		return 100
	victim = board.piece_at(move.to_square)
	aggressor = board.piece_at(move.from_square)
	if victim and aggressor:
		return MVV_LVA[victim.piece_type][aggressor.piece_type]
	return 0

class ZobristHashing:
	def __init__(self):
		# Initialize a random bitstring for each piece on each square
		self.board_size = 64  # 8x8 chess board
		self.piece_types = [
			chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
		]
		self.colors = [chess.WHITE, chess.BLACK]
		
		# Create a 2D list for storing random hashes: piece + color + square
		self.zobrist_table = {}

		# Initialize the Zobrist table with random 64-bit integers for each piece, color, and square
		for piece in self.piece_types:
			for color in self.colors:
				for square in range(self.board_size):
					self.zobrist_table[(piece, color, square)] = random.getrandbits(64)
		
		# For castling rights, en-passant square, and turn
		self.castling_rights_table = {}
		self.en_passant_table = {}
		self.turn_table = {}

		# Random values for castling rights, en-passant, and turn
		self.initialize_misc()

	def initialize_misc(self):
		# Castling rights (white and black can each have 4 possible combinations)
		for color in self.colors:
			for rights in range(16):  # 4 bits for each side (KQkq)
				self.castling_rights_table[(color, rights)] = random.getrandbits(64)

		# En passant target square
		for square in range(self.board_size):
			self.en_passant_table[square] = random.getrandbits(64)

		# Turn
		self.turn_table = {
			chess.WHITE: random.getrandbits(64),
			chess.BLACK: random.getrandbits(64),
		}

	def compute_hash(self, board: chess.Board):
		hash_value = 0

		# Loop over each square and compute the hash based on the piece on that square
		for square in range(self.board_size):
			piece = board.piece_at(square)
			if piece:
				color = board.color_at(square)
				# XOR the hash for the piece and square
				hash_value ^= self.zobrist_table[(piece.piece_type, color, square)]

		# Add the hash for castling rights
		white_castling = board.castling_rights & chess.WHITE
		black_castling = board.castling_rights & chess.BLACK
		hash_value ^= self.castling_rights_table[(chess.WHITE, white_castling)]
		hash_value ^= self.castling_rights_table[(chess.BLACK, black_castling)]

		# Add the hash for en passant square
		if board.ep_square:
			hash_value ^= self.en_passant_table[board.ep_square]

		# Add the hash for the turn (white or black)
		hash_value ^= self.turn_table[board.turn]

		return hash_value


	def update_hash(self, current_hash, board: chess.Board, move: chess.Move):
		# Update the hash when a move is made by XORing the hash for the piece from its old and new squares
		from_square = move.from_square
		to_square = move.to_square
		moving_piece = board.piece_at(from_square)

		# Remove piece from the old square
		color = board.color_at(from_square)
		current_hash ^= self.zobrist_table[(moving_piece.piece_type, color, from_square)]
		
		# Add the piece to the new square
		current_hash ^= self.zobrist_table[(moving_piece.piece_type, color, to_square)]
		
		# Handle en-passant, castling, and turn
		if move.promotion:
			# Handle promotion (this is simplified)
			current_hash ^= self.zobrist_table[(chess.QUEEN, color, to_square)]
		
		# Update the turn
		current_hash ^= self.turn_table[board.turn]
		
		return current_hash

zobrist = ZobristHashing()

class NodeType(Enum):
	EXACT = 0    # Exact score
	LOWERBOUND = 1    # Beta cutoff (fail-high)
	UPPERBOUND = 2    # Alpha cutoff (fail-low)

class TranspositionEntry:
	def __init__(self, key, depth, score, node_type, best_move):
		self.key = key
		self.depth = depth
		self.score = score
		self.node_type = node_type
		self.best_move = best_move

class TranspositionTable:
	def __init__(self, size_mb):
		# Calculate number of entries based on size in MB
		self.size = (size_mb * 1024 * 1024) // 32  # Assuming each entry is ~32 bytes
		self.table = {}
	
	def store(self, key, depth, score, node_type, best_move):
		# Replace if new position is searched to greater or equal depth
		if key in self.table:
			if self.table[key].depth <= depth:
				self.table[key] = TranspositionEntry(key, depth, score, node_type, best_move)
		else:
			if len(self.table) >= self.size:
				# Simple replacement strategy: remove a random entry
				self.table.pop(next(iter(self.table)))
			self.table[key] = TranspositionEntry(key, depth, score, node_type, best_move)
	
	def lookup(self, key):
		return self.table.get(key)

class TriangularPVTable:
	def __init__(self, max_depth):
		self.max_depth = max_depth
		self.table: List[List[Optional[chess.Move]]] = [[None for _ in range(max_depth+1)] for _ in range(max_depth+1)]
	
	def store(self, depth, ply, move):
		self.table[ply][ply] = move
		for next_ply in range(ply + 1, ply + depth):
			self.table[ply][next_ply] = self.table[ply + 1][next_ply]
	
	def get(self, depth, ply):
		return self.table[depth][ply]

def order_moves(moves, board, ply, killer_moves, history_table, pv_table, tt_move=None):
	moves = list(moves)
	move_scores = []
	for move in moves:
		score = 0
		# TT move gets highest priority
		if tt_move and move == tt_move:
			score += 20000
		# # PV move gets next highest priority
		# elif move in pv_table.table[0]:
		# 	score += 10000
		# Killer moves get third priority
		elif move in killer_moves[ply]:
			score += 8500
		# Captures get base score based on MVV-LVA
		elif board.is_capture(move):
			score += 8000 + mvv_lva(board, move)
		# Promotions
		elif move.promotion:
			score += 7000 
		# Check moves
		elif board.gives_check(move):
			score += 6000
		# History heuristic
		score += history_table.get((board.turn, move.from_square, move.to_square), 0)
			
		move_scores.append((move, score))
	
	return [move for move, score in sorted(move_scores, key=lambda x: x[1], reverse=True)]

def negamax(board: chess.Board, depth, alpha, beta, color, 
			killer_moves, history_table, pv_table, tt_table, ply=0, is_pv_node=False, type=0):
	# Early returns
	if depth == 0 and not board.is_game_over():
		return quiescence(board, alpha, beta, color, killer_moves, history_table, ply, type=type), None

	mate_value = 30000
	if board.is_checkmate():
		return -mate_value, None
	if board.is_stalemate():
		return 0, None

	legal_moves = board.legal_moves

	
		
	# Transposition table lookup
	hash_board = zobrist.compute_hash(board)
	tt_entry = tt_table.lookup(hash_board)
	if tt_entry and tt_entry.depth >= depth and not is_pv_node:
		if tt_entry.node_type == NodeType.EXACT:
			return tt_entry.score, tt_entry.best_move
		elif tt_entry.node_type == NodeType.LOWERBOUND:
			alpha = max(alpha, tt_entry.score)
		elif tt_entry.node_type == NodeType.UPPERBOUND:
			beta = min(beta, tt_entry.score)
		if alpha >= beta:
			return tt_entry.score, tt_entry.best_move
	
	# Futility Pruning
	stand_pat = 0
	if type == 0:
		stand_pat = evaluations.evaluate_board(board) 
	else:
		stand_pat = prediction.predict(type, board, prediction.device)

	if stand_pat >= beta and ply > 0 :
		return beta + (stand_pat-beta)//13, None 
	if stand_pat <= alpha and ply > 0 :
		return alpha, None
	# if depth <= 2 and abs(stand_pat) < 120 and ply > 0 :  # Threshold for futility pruning
	# 	return stand_pat, None

	# Null move pruning
	null_move_reduction = 2
	if beta-alpha==1 and depth > null_move_reduction and not board.is_check() and len(list(legal_moves)) > 0 and board.move_stack[-1] != chess.Move.null():
		board.push(chess.Move.null())
		value = -negamax(board, depth - 1 - null_move_reduction, -beta, -beta + 1, -color,
						 killer_moves, history_table, pv_table, 
						 tt_table, ply + 1, False, type = type)[0]
		board.pop()
		if value >= beta:
			return value, None

	# Mate distance pruning
	alpha = max(alpha, -mate_value + ply)
	beta = min(beta, mate_value - ply + 1)
	if alpha >= beta:
		return alpha, None

	best_value = float('-inf')
	best_move = None
	original_alpha = alpha
	

	ordered_moves = order_moves(legal_moves, board, ply, killer_moves, history_table, pv_table, 
							  tt_entry.best_move if tt_entry else None)
	for index, move in enumerate(ordered_moves):
		board.push(move)
		value = 0
		# Apply Late Move Reductions (LMR)
		if (index > 0 and depth > 2 and not board.is_check()):
			reduced_depth = depth - 1
			value = -negamax(board, reduced_depth - 1, -alpha - 1, -alpha, -color, 
						   killer_moves, history_table, pv_table, tt_table,
						   ply + 1, False, type = type)[0]
			if alpha < value < beta:
				value = -negamax(board, depth - 1, -beta, -alpha, -color, 
							   killer_moves, history_table, pv_table, tt_table,
							   ply + 1, True, type = type)[0]
		else:
			value = -negamax(board, depth - 1, -beta, -alpha, -color, 
						   killer_moves, history_table, pv_table, tt_table,
						   ply + 1, is_pv_node, type = type)[0]
		
		board.pop()
		
		if value > best_value:
			best_value = value
			best_move = move
			# pv_table.store(depth, ply, best_move)
			
		alpha = max(alpha, value)
		if alpha >= beta:
			#Store killer move
			if not board.is_capture(move):
				if move not in killer_moves[ply]:
					killer_moves[ply][1] = killer_moves[ply][0]
					killer_moves[ply][0] = move
			# Update history table
			history_table[(board.turn, move.from_square, move.to_square)] = (
				history_table.get((board.turn, move.from_square, move.to_square), 0) + depth * depth
			)
			# Store position in transposition table
			tt_table.store(hash_board, depth, beta, NodeType.LOWERBOUND, move)
			break
	
	# Store position in transposition table
	if best_move:
		node_type = (
			NodeType.UPPERBOUND if best_value <= original_alpha else
			NodeType.LOWERBOUND if best_value >= beta else
			NodeType.EXACT
		)
		tt_table.store(hash_board, depth, best_value, node_type, best_move)
	
	# Store PV move
	# if best_move and best_value > original_alpha:
	# 	print("store",depth,ply)
	# 	pv_table.store(depth, ply, best_move)
	
	return best_value, best_move

def quiescence(board, alpha, beta, color, killer_moves, history_table, ply, type=0):
	MATE_VALUE = 30000
	
	if board.is_insufficient_material() and alpha < 0 :
		alpha = 0
		if alpha >= beta:
			return alpha

	# Mate dis pruning
	if ply > 0 :
		alpha = max(-MATE_VALUE + ply, alpha)
		beta  = min(MATE_VALUE - ply + 1, beta)
		if alpha >= beta :
			return alpha

	stand_pat = 0
	if type == 0:
		stand_pat = evaluations.evaluate_board(board) 
	else:
		stand_pat = prediction.predict(type, board, prediction.device)

	if stand_pat >= beta:
		return beta
	if alpha < stand_pat:
		alpha = stand_pat


	 # Delta pruning
	delta = 975
	if chess.QUEEN in [move.promotion for move in board.legal_moves] :
		delta += 775
	if stand_pat < alpha - delta :
		return alpha
	moves = order_moves(board.generate_legal_captures(), board, ply, killer_moves, history_table, None, None )
	for move in moves:
		if see(board, move) < 0:
			continue
		board.push(move)
		value = -quiescence(board, -beta, -alpha, -color, killer_moves, history_table, ply, type=type)
		board.pop()
		if value >= beta:
			return beta
		alpha = max(alpha, value)
	
	if board.is_checkmate():
		alpha = -float('inf')
	elif board.is_stalemate():
		alpha = 0
	
	return alpha

def see(board: chess.Board, move: chess.Move):
	from_square = move.from_square
	to_square = move.to_square
	piece = board.piece_at(from_square)
	if not piece:
		return 0
		
	# Store the initial captured piece value
	target_piece = board.piece_at(to_square)
	gain = [evaluations.VALUES[target_piece.piece_type][2] if target_piece else 0]
	
	board.push(move)
	attackers = board.attackers(not piece.color, to_square)
	color = piece.color
	
	# Keep track of number of moves made
	moves_made = 1

	while attackers:
		attacker_square = min(attackers, key=lambda sq: evaluations.VALUES[board.piece_at(sq).piece_type][2])
		attacker = board.piece_at(attacker_square)
		gain.append(evaluations.VALUES[board.piece_at(to_square).piece_type][2] - gain[-1])
		board.push(chess.Move(attacker_square, to_square))
		moves_made += 1
		attackers = board.attackers(color, to_square)
		color = not color

	# Pop all moves made during SEE calculation
	for _ in range(moves_made):
		board.pop()

	# Minimize the gains for each player
	for i in range(len(gain) - 1, 0, -1):
		gain[i-1] = min(-gain[i], gain[i-1])

	return gain[0]

# Initialize tables and killer moves (add this where you initialize your search)
def initialize_search(max_depth, tt_size_mb=64):
	killer_moves = [[None, None] for _ in range(max_depth + 1)]
	history_table = {}
	pv_table = TriangularPVTable(max_depth)
	tt_table = TranspositionTable(tt_size_mb)
	return killer_moves, history_table, pv_table, tt_table
def retrieve_principal_variation(board, tt_table, max_depth):
	"""
	Retrieve the principal variation by following the transposition table.
	"""
	pv = []
	for _ in range(max_depth):
		hash_board = zobrist.compute_hash(board)
		tt_entry = tt_table.lookup(hash_board)
		if not tt_entry or not tt_entry.best_move:
			break
		pv.append(tt_entry.best_move.uci())
		board.push(tt_entry.best_move)
	for _ in range(len(pv)):
		board.pop()

	return pv

def iterative_deepening(board, max_depth, alpha, beta, color, tt_size_mb=64, timeend=4, type=0):
	# Initialize tables and killer moves
	killer_moves, history_table, pv_table, tt_table = initialize_search(max_depth, tt_size_mb)

	best_move = None
	best_value = float('-inf')
	endlimit = 0
	start = time.time()
	endlimit = timeend
	for depth in range(1, max_depth + 1):
		print(f"Starting search at depth {depth}...")
		value, move = negamax(board, depth, alpha, beta, color,
							  killer_moves, history_table, pv_table, tt_table, type=type)
		if move:
			best_move = move
			best_value = value

		mate_value = 30000
		if abs(value) >= mate_value:
			print("Mate found, stopping search.")
			pv_moves = retrieve_principal_variation(board, tt_table, max_depth)
			print(f"Depth {depth} complete. Best move: {best_move}, Value: {best_value}, {pv_moves}")
			break
		pv_moves = retrieve_principal_variation(board, tt_table, max_depth)
		print(f"Depth {depth} complete. Best move: {best_move}, Value: {best_value}, {pv_moves}")
		if time.time()-start >=endlimit:
			break

	return pv_moves, best_value, best_move

# Example usage
board = chess.Board("r1b1kbnr/pppp1ppp/8/4n1q1/5Q2/8/PPP1PPPP/RNB1KBNR w KQkq - 0 5")
max_depth = 9
alpha = float('-inf')
beta = float('inf')
times = time.time()
moves, best_value, best_move = iterative_deepening(board, max_depth, alpha, beta, 1, timeend=3, type=1)
print(f"Best move: {best_move}, Best value: {best_value}")
end = time.time()-times
print(f"Time: {end}")
