from typing import List, Tuple, Dict

import numpy as np
from axelrod.action import Action
from axelrod.evolvable_player import (
    EvolvablePlayer,
    InsufficientParametersError,
    crossover_dictionaries,
)
from axelrod.player import Player

C, D = Action.C, Action.D

def gen_rnn_params(i: int = 2, o: int = 2, h: int = 20, scale: float = .1) -> Dict:
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
  def __init__(self, params : Dict = None, param_gen_args: Dict = None) -> None:
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

class EvolvableRNN(RNN, EvolvablePlayer):
    """Evolvable version of RNN."""
    name = "EvolvableRNN"

    def __init__(
        self,
        params : Dict = None,
        param_gen_args: Dict = None,
        mutation_probability: float = None,
        mutation_distance: int = 5,
        seed: int = None
    ) -> None:
        EvolvablePlayer.__init__(self, seed=seed)
        RNN.__init__(self,
                     params=params,
                     param_gen_args=param_gen_args
                     )
        self.mutation_probability = mutation_probability
        self.mutation_distance = mutation_distance
        self.overwrite_init_kwargs(
            params=params,
            param_gen_args=param_gen_args,
            mutation_probability=mutation_probability)

    def mutate_params(self, params: Dict, mutation_probability: float, mutation_distance: int):
      """Return new params"""
      new_params = {}
      for key, value in params.copy().items():
        mask = np.random.binomial(n = 1, p = mutation_probability, size=value.shape)
        noise = np.random.uniform(-1, 1, size=value.shape)
        new_params[key] = value + mutation_distance * noise * mask
      return new_params

    def mutate(self):
        params = self.mutate_params(
          self.params,
          self.mutation_probability,
          self.mutation_distance
        )
        return self.create_new(params=params)

    def crossover(self, other):
        if other.__class__ != self.__class__:
            raise TypeError("Crossover must be between the same player classes.")
        params = crossover_dictionaries(self.params, other.params, self._random)
        return self.create_new(params=params)