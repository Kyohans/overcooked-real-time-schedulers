import numpy as np

# Each agent here will inherit the Greedy model to avoid writing repeating code for the action and ml_action functions
from overcooked_ai_py.agents.agent import GreedyHumanModel as GreedyAgent

class EDFAgent(GreedyAgent):
    """
    An agent that chooses an action plan according to their earliest deadline
    from the current state.
    """
    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        return self.get_earliest_deadline_action_and_goal(start_pos_and_or, motion_goals)

    def get_earliest_deadline_action_and_goal(self, start_pos_and_or, motion_goals):
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
    """
    Least-Laxity-First Agent that calculates the laxity of each action (cost - deadline)
    and selects the action with the least laxity
    """
    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        return self.get_least_laxity_action_and_goal(start_pos_and_or, motion_goals)

    def get_least_laxity_action_and_goal(self, start_pos_and_or, motion_goals):
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

class FIFOAgent(GreedyAgent):
    """
    An agent that takes the first motion action available to them
    and follows that path. Interact actions are prioritized.
    """
    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        return self.get_fifo_action_and_goal(start_pos_and_or, motion_goals)

    def get_fifo_action_and_goal(self, start_pos_and_or, motion_goals):
        first_plan[0], first_goal = self.mlam.motion_planner.get_plan(start_pos_and_or, motion_goals[0]), motion_goals[0]
        if 'interact' in first_plan:
            return first_goal, first_plan
        else:
            for goal in motion_goals:
                action_plan, _, _ = self.mlam.motion_planner.get_plan(start_pos_and_or, goal)
                if 'interact' in action_plan:
                    first_plan = action_plan[0]
                    first_goal = goal
                    
                    break

        return first_goal, first_plan

