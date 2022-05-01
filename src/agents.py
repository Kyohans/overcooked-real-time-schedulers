from collections import deque
import numpy as np
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.agents.agent import Agent, GreedyHumanModel as GreedyAgent

class EDFAgent(GreedyAgent):
    """
    An agent that chooses an action plan according to their earliest deadline
    from the current state.
    """
    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        min_deadline = np.Inf
        earliest_action, earliest_goal = None, None

        for goal in motion_goals:
            action_plan, _, _ = self.mlam.motion_planner.get_plan(start_pos_and_or, goal)
            plan_deadline = len(action_plan)

            if plan_deadline < min_deadline:
                min_deadline = plan_deadline
                earliest_action = action_plan[0]
                earliest_goal = goal

        return earliest_goal, earliest_action

class LLFAgent(GreedyAgent):
    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        min_laxity = np.Inf
        best_action, best_goal = None, None

        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(start_pos_and_or, goal)
            plan_deadline = len(action_plan)

            plan_laxity = abs(plan_cost - plan_deadline)
            if plan_laxity < min_laxity:
                min_laxity = plan_laxity
                best_action = action_plan[0]
                best_goal = goal

        return best_goal, best_action

class FIFOAgent(Agent):
    """
    An agent that takes the first motion action available to them
    and follows that path. Interact actions are prioritized.
    """
    def __init__(self):
        self.action_queue = deque()

    def action(self, state):
        if not self.action_queue:
            legal_actions = list(Action.MOTION_ACTIONS)
            for action in legal_actions:
                if action not in self.action_queue:
                    self.action_queue.append(action)

        action = self.action_queue.popleft()
        return action, {}
    
    def actions(self, states):
        return [self.action(state) for state in states]
