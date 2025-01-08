import search

from fastapi import FastAPI
import chess
import time

app = FastAPI()

class chessAPI():
	def __init__(self):
		self.board = chess.Board()
chessapi = chessAPI()

@app.get("/new")
def create_new():
	chessapi.board = chess.Board()


@app.get("/legalmoves")
def legal_moves():
	moves = [move.uci() for move in chessapi.board.legal_moves]
	return moves

@app.get("/legalmovessq")
def legal_movessq():
	moves = [(move.from_square, move.to_square) for move in chessapi.board.legal_moves]
	return moves

@app.get("/legalmovesfromsq/{square}")
def legal_movesfromsq(square: int):
	moves = [move.to_square for move in chessapi.board.legal_moves if move.from_square==square]
	return moves

@app.get("/incheck")
def incheck():
	return chessapi.board.is_check()

@app.get("/ischeckmate")
def ischeckmate():
	return chessapi.board.is_checkmate()

@app.get("/searchbestmove/{typeeval}")
def searchbestmove(typeeval: int):
	alpha = float('-inf')
	beta = float('inf')
	if typeeval == 0:
		start = time.time()
		moves, value, best_move = search.iterative_deepening(chessapi.board, 7, alpha, beta, 1, timeend=3, type=0)
		moves2 = [(chess.Move.from_uci(m).from_square, chess.Move.from_uci(m).to_square) for m in moves]
		print("search type 0")
		return moves2,'|',moves,'|',best_move.from_square, best_move.to_square,'|',time.time()-start
	elif typeeval == 1:
		start = time.time()
		moves, value, best_move = search.iterative_deepening(chessapi.board, 6, alpha, beta, 1, timeend=2, type=1)
		moves2 = [(chess.Move.from_uci(m).from_square, chess.Move.from_uci(m).to_square) for m in moves]
		print("search type 1")
		return moves2,'|',moves,'|',best_move.from_square, best_move.to_square,'|',time.time()-start
	elif typeeval == 2:
		start = time.time()
		moves, value, best_move = search.iterative_deepening(chessapi.board, 3, alpha, beta, 1, timeend=2, type=2)
		moves2 = [(chess.Move.from_uci(m).from_square, chess.Move.from_uci(m).to_square) for m in moves]
		print("search type 2")
		return moves2,'|',moves,'|',best_move.from_square, best_move.to_square,'|',time.time()-start
	elif typeeval == 3:
		start = time.time()
		moves, value, best_move = search.iterative_deepening(chessapi.board, 1, alpha, beta, 1, timeend=2, type=3)
		moves2 = [(chess.Move.from_uci(m).from_square, chess.Move.from_uci(m).to_square) for m in moves]
		print("search type 3")
		return moves2,'|',moves,'|',best_move.from_square, best_move.to_square,'|',time.time()-start 

@app.get("/pushmove")
def push_move(from_square: int, to_square: int):
	move = chess.Move(from_square, to_square)
	if chessapi.board.piece_type_at(from_square) == 1 and (move.uci().endswith("8") or move.uci().endswith("1")):
		move.promotion = chess.QUEEN
	if move in chessapi.board.legal_moves:
		chessapi.board.push(move)
		print("Push move",move.uci())
		return True
	else:
		print("Move not exist",move)
	return False