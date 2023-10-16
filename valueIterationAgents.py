# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        num_iterations = self.iterations

        for _ in range(num_iterations):
            all_states = self.mdp.getStates()
            temp_values = util.Counter()

            for curr_state in all_states:
                if (not self.mdp.isTerminal(curr_state)):

                    best_action = self.computeActionFromValues(curr_state)
                    best_q_value = self.computeQValueFromValues(curr_state, best_action)
                    temp_values[curr_state] = best_q_value
        
            self.values = temp_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        transition_states = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0
        for next_state,prob in transition_states:
            q_value += prob * ((self.mdp.getReward(state,action,next_state)) + self.discount*self.values[next_state])
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)
        best_action = actions[0] if len(actions) > 0 else None
        best_value = self.computeQValueFromValues(state,actions[0]) if len(actions) > 0 else float('-inf')
        for action in actions:
            cur_q_value = self.computeQValueFromValues(state,action)
            if cur_q_value > best_value:
                best_value = cur_q_value
                best_action = action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
    
        for i in range(self.iterations):
            all_states = self.mdp.getStates()
            curr_state = all_states[i % len(all_states)]
            if (not self.mdp.isTerminal(curr_state)):
                best_action = self.computeActionFromValues(curr_state)
                best_q_value = self.computeQValueFromValues(curr_state, best_action)
                self.values[curr_state] = best_q_value
        

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        all_states = self.mdp.getStates()
        predecessors = dict()
        for curr_state in all_states:
            possible_actions = self.mdp.getPossibleActions(curr_state)
            for action in possible_actions:
                transition_states = self.mdp.getTransitionStatesAndProbs(curr_state, action)
                for trans_state in transition_states:
                    current_transition_state = trans_state[0]
                    if current_transition_state not in predecessors:
                        predecessors[current_transition_state] = []
                    if curr_state not in predecessors[current_transition_state]:
                        predecessors[current_transition_state].append(curr_state)

        errors = util.PriorityQueue()
        for curr_state in all_states:
            if (not self.mdp.isTerminal(curr_state)):
                curr_value = self.values[curr_state]
                best_action = self.computeActionFromValues(curr_state)
                best_q_value = self.computeQValueFromValues(curr_state, best_action)
                diff =  abs(best_q_value - curr_value)
                errors.push(curr_state, -diff)
        
        
        for i in range(self.iterations):
            if errors.isEmpty():
                break
            s = errors.pop()
            best_action = self.computeActionFromValues(s)
            best_q_value = self.computeQValueFromValues(s, best_action)
            self.values[s] = best_q_value
            for predecessor in predecessors[s]:
                curr_predecessor_value = self.values[predecessor]
                best_action = self.computeActionFromValues(predecessor)
                best_q_value = self.computeQValueFromValues(predecessor, best_action)
                diff =  abs(best_q_value - curr_predecessor_value)
                if diff > self.theta:
                    errors.update(predecessor, -diff)


