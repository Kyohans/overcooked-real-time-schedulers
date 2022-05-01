from src.agents import *
from overcooked_ai_py.agents.agent import AgentPair, GreedyHumanModel as GreedyAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator, LayoutGenerator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

def generate_layout(layout_name=None, gen_params=None):
    """
    Generates layout for simulation.
    :param string: layout_name: pre-defined layout name supplied by overcooked_ai
    :param dict: gen_params: Parameters for LayoutGenerator. Possible parameters are as follows.
        - inner_shape (int, int): layout size
        - prop_empty (float): proportion of empty space in layout
        - prop_feats (float): proportion of counters with features on them
        - start_all_orders (list[dict]): list of possible recipe combinations
        - recipe_values (list[int]): Score values of corresponding recipes in recipe list
        - recipe_times (list[int]): Total time to cook for each corresponding recipe in recipe list
    """
    if layout_name and not gen_params:
        return LayoutGenerator.mdp_gen_fn_from_dict({"layout_name": layout_name})
    elif not layout_name and gen_params:
        return LayoutGenerator.mdp_gen_fn_from_dict(gen_params)
    else:
        return LayoutGenerator.mdp_gen_fn_from_dict({"layout_name": "cramped_room"})

def generate_agent_evaluator(layout_mdp, horizon):
    """
    Generates an AgentEvaluator object for the given layout and horizon value
    :params mdp_fn (OvercookedGridWorld): OvercookedGridWorld object created from LayoutGenerator
    :params horizon (int): Indicates how many timesteps will be made in each episode of evaluation
    """
    return AgentEvaluator(env_params={"horizon": horizon}, mdp_fn=mdp_fn)

if __name__ == '__main__':
    mdp_fn = generate_layout()
    agent_eval = generate_agent_evaluator(mdp_fn, 100)

    edf_edf_pair = AgentPair(EDFAgent(agent_eval.env.mlam), EDFAgent(agent_eval.env.mlam))
    trajectories_edf_pair = agent_eval.evaluate_agent_pair(edf_edf_pair, num_games=500)

    print(trajectories_edf_pair["ep_returns"])
    print(trajectories_edf_pair["ep_rewards"])
    # img_dir_path = StateVisualizer().display_rendered_trajectory(trajectories_edf_pair, trajectory_idx=0, ipython_display=True)
