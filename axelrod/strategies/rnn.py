from typing import List, Tuple, Dict

import numpy as np
from axelrod.action import Action
from axelrod.evolvable_player import (
    EvolvablePlayer,
    InsufficientParametersError,
    crossover_lists,
)
from axelrod.player import Player

C, D = Action.C, Action.D

def gen_rnn_params(i=2, o=2, h=20, scale=.1) -> Dict:
  params = {
      'Whx': np.random.normal(size=(i, h)) * scale,
      'Whh': np.random.normal(size=(h, h)) * scale,
      'Woh': np.random.normal(size=(h, o)) * scale,
      'h0': np.random.normal(size=(h)) * scale,
      'bh': np.random.normal(size=h) * scale,
      'bo': np.random.normal(size=o) * scale
  }
  return params

def rnn_cell(params: Dict, h: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  h_t = np.tanh(np.dot(x, params['Whx']) + np.dot(params['Whh'], h) + params['bh'])
  o_t = np.dot(h_t, params['Woh']) + params['bo']
  return h_t, o_t

class RNN(Player):
  name = "RNN"
  classifier = {
      "memory_depth": float("inf"),
      "stochastic": False,
      "inspects_source": False,
      "makes_use_of": set(),
      "manipulates_source": False,
      "manipulates_state": False,
      "long_run_time": False,
  }
  def __init__(self, params=None, param_gen_args=None):
    Player.__init__(self)
    if params:
      self.params = params
    else:
      if param_gen_args:
        self.params = gen_rnn_params(**param_gen_args)
      else:
        self.params = gen_rnn_params()
    self.h = self.params['h0']

  def __repr__(self):
    return self.name
  
  def __str__(self):
    return self.name
      
  
  def strategy(self, opponent: Player) -> Action:
    if not self.history:
      return C
    own_move, opponent_move = self.history[-1], opponent.history[-1]
    features = np.array([own_move.value, opponent_move.value])
    h, output = rnn_cell(self.params, self.h, features)
    self.h = h
    action_probs = np.exp(output) / np.sum(np.exp(output))
    if action_probs[0] < .5:
      return C
    else:
      return D