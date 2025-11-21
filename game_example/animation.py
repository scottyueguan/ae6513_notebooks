import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches

def peg_trajectory_simulation(peg, init_pos_p=(0,0), init_pos_e=(7,7), policy_p=None, policy_e=None):
    state = peg.pos2game_state(pos_1=init_pos_p, pos_2=init_pos_e)         # initial game state

    state_trj = [state]
    pursuer_pos_trj, evader_pos_trj = [init_pos_p], [init_pos_e]

    while not peg.is_terminal(state):
        # get action distributions
        action_dist_p, action_dist_e = policy_p[state], policy_e[state]

        # selection actions
        action_space_p, action_space_e = peg.action_mappings_1[state], peg.action_mappings_2[state]
        action_index_p = np.random.choice(a=list(range(len(action_space_p))), p=action_dist_p)
        action_index_e = np.random.choice(a=list(range(len(action_space_e))), p=action_dist_e)
        action_p, action_e = action_space_p[action_index_p], action_space_e[action_index_e]

        # update positions and state
        init_pos_p = (init_pos_p[0] + action_p[0], init_pos_p[1] + action_p[1])
        init_pos_e = (init_pos_e[0] + action_e[0], init_pos_e[1] + action_e[1])
        state = peg.pos2game_state(init_pos_p, init_pos_e)

        # store the trajectory
        state_trj.append(state)
        pursuer_pos_trj.append(init_pos_p)
        evader_pos_trj.append(init_pos_e)

    print("Simulation completed. Total steps: ", len(state_trj))
    return state_trj, pursuer_pos_trj, evader_pos_trj

def peg_animate_per_frame(ax, peg, occupancy_matrix, pos_p, pos_e):
    n_rows, n_cols = occupancy_matrix.shape
    plt.cla()
    ax.axis([-0.1, n_rows+0.1, -0.1, n_cols +0.1])
    ax.axis("off")
    ax.set_aspect('equal', adjustable='box')

    # render obstacles and grids
    for i in range(n_rows):
        for j in range(n_cols):
            if occupancy_matrix[i, j] == 1:
                ax.add_patch(patches.Rectangle((j, i), 1.0, 1.0, facecolor="gray"))
    for i in range(n_rows + 1):
        ax.add_artist(lines.Line2D([0, n_cols], [i, i], color="k", linestyle=":", linewidth=0.5))
    for j in range(n_cols + 1):
        ax.add_artist(lines.Line2D([j, j], [n_rows, 0], color="k", linestyle=":", linewidth=0.5))

    # render pursuer
    ax.add_artist(patches.Circle((pos_p[1] + 0.5, pos_p[0] + 0.5), radius=0.2, color='r'))

    # render capture radius
    row, col = pos_p[0] + 0.5, pos_p[1] + 0.5
    radius = peg.capture_radius + 0.5
    row_start, row_end = row - radius, row + radius
    col_start, col_end = col - radius, col + radius
    # row lines
    ax.add_artist(lines.Line2D([col_start, col_end], [row_start, row_start], color='r', linestyle="--",
                                       linewidth=1))
    ax.add_artist(lines.Line2D([col_start, col_end], [row_end, row_end], color='r', linestyle="--",
                                       linewidth=1))
    # col lines
    ax.add_artist(lines.Line2D([col_start, col_start], [row_start, row_end], color='r', linestyle="--",
                                       linewidth=1))
    ax.add_artist(lines.Line2D([col_end, col_end], [row_start, row_end], color='r', linestyle="--",
                                       linewidth=1))

    # render evader
    ax.add_artist(patches.Circle((pos_e[1] + 0.5, pos_e[0] + 0.5), radius=0.2, color='b'))