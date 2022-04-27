import numpy as np
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.agents.agent import Agent

class EDFAgent(Agent):
    """
    An agent that choses the first motion action available that
    has the lowest cost to the total deadline
    """
    def __init__(self, mlam):
        self.mlam = mlam
        self.mdp = self.mlam.mdp

    def action(self, state):
        """
        Performs actions according to EDF algorithm
        """
        possible_motion_goals = self.get_motion_goals(state)

        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        chosen_goal, chosen_action, action_probs = self.choose_motion_goal(start_pos_and_or, possible_motion_goals)

        return chosen_action, {'action_probs': action_probs}

    def get_motion_goals(self, state):
        """
        Finds available motion goals for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]
        """
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]
        
        counter_objects = self.mlam.mdp.get_counter_objects_dict(state, list(self.mlam.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.mlam.mdp.get_pot_states(state)

        if not player.has_object():
            motion_goals = self.mlam.pickup_dish_actions(counter_objects)
        else:
            player_obj = player.get_object()

            match player_obj.name:
                case 'onion':
                    motion_goals = self.mlam.put_onion_in_pot_actions(pot_states_dict)
                case 'tomato':
                    motion_goals = self.mlam.put_tomato_in_pot_actions(pot_states_dict)
                case 'dish':
                    motion_goals = self.mlam.pickup_soup_with_dish_actions(pot_states_dict, only_nearby_ready = True)
                case 'soup':
                    motion_goals = self.mlam.deliver_soup_actions()
                case _:
                    raise ValueError()

        motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
        if len(motion_goals) == 0:
            motion_goals = self.mlam.go_to_closest_feature_actions(player)
            motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
            assert len(motion_goals) != 0

        return motion_goals

    def choose_motion_goal(self, start_pos_and_or, motion_goals):
        # Finds action plan that has the lowest path (earliest deadline)
        chosen_goal, chosen_goal_action = self.get_earliest_action_plan(start_pos_and_or, motion_goals)
        action_probs = self.a_probs_from_action(chosen_goal_action)

        print(chosen_goal_action)

        return chosen_goal, chosen_goal_action, action_probs
    
    def get_earliest_action_plan(self, start_pos_and_or, motion_goals):
        """
        Finds the nearest action plan that has a deadline earliest of other available action plans from current state.
        Similar to get_lowest_cost_action_and_goal from GreedyHumanModel, but also selects the action plan that can be completed the earliest
        """
        min_len = np.Inf
        min_cost = np.Inf

        earliest_action, earliest_goal = None, None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(start_pos_and_or, goal)

            if plan_cost < min_cost or len(action_plan) < min_len:
                earliest_action = action_plan[0]
                earliest_goal = goal

                if len(action_plan) < min_len:
                    min_len = len(action_plan)

                if plan_cost < min_cost:
                    min_cost = plan_cost

        return earliest_goal, earliest_action

class LLFAgent(Agent):
    pass

class FIFOAgent(Agent):
    """
    An agent that takes the first motion action available to them
    and follows that path. Interact actions are prioritized.
    """
