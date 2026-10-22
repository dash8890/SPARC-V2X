"""The frozen admission contract: None denotes an infeasible edge job."""


def feasible_actions(k, helper_count):
    return range(k, helper_count + 1)


def action_mask(k, helper_count, action_dim):
    """Network index n means n workers; an infeasible job has an empty mask."""
    if action_dim <= helper_count:
        raise ValueError("Network output must cover every feasible worker count")
    return [k <= n <= helper_count for n in range(action_dim)]
