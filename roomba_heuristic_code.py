from time import perf_counter
import numpy as np
import matplotlib.pyplot as pt
from matplotlib import animation
from queue_search_code import *

WALL, CHARGER, CLEAN, DIRTY = list(range(4))
SIZE = 7

class RoombaDomain:
    def __init__(self):

        # deterministic grid world
        num_rows, num_cols = SIZE, SIZE
        grid = CLEAN*np.ones((num_rows, num_cols), dtype=int)
        grid[SIZE//2, 1:SIZE-1] = WALL
        grid[1:SIZE//2+1,SIZE//2] = WALL
        grid[0,0] = CHARGER
        grid[0,-1] = CHARGER
        grid[-1,SIZE//2] = CHARGER
        max_power = 2*SIZE + 1
        
        self.grid = grid
        self.max_power = max_power

    def pack(self, g, r, c, p):
        return (g.tobytes(), r, c, p)
    def unpack(self, state):
        grid, r, c, p = state
        grid = np.frombuffer(grid, dtype=int).reshape(self.grid.shape).copy()
        return grid, r, c, p

    def initial_state(self, roomba_position, dirty_positions):
        r, c = roomba_position
        grid = self.grid.copy()
        for dr, dc in dirty_positions: grid[dr, dc] = DIRTY
        return self.pack(grid, r, c, self.max_power)

    def render(self, ax, state, x=0, y=0):
        grid, r, c, p = self.unpack(state)
        num_rows, num_cols = grid.shape
        ax.imshow(grid, cmap='gray', vmin=0, vmax=3, extent=(x-.5,x+num_cols-.5, y+num_rows-.5, y-.5))
        for col in range(num_cols+1): pt.plot([x+ col-.5, x+ col-.5], [y+ -.5, y+ num_rows-.5], 'k-')
        for row in range(num_rows+1): pt.plot([x+ -.5, x+ num_cols-.5], [y+ row-.5, y+ row-.5], 'k-')
        pt.text(c-.25, r+.25, str(p), fontsize=24)
        pt.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    def valid_actions(self, state):

        
        grid, r, c, p = self.unpack(state)
        num_rows, num_cols = grid.shape
        actions = []


        if(c < num_cols-1 and grid[r,c+1] != WALL):     #Move Right
            actions.append(((0,1),1))

        if(c > 0 and grid[r,c-1] != WALL):            #Move Left
            actions.append(((0,-1),1))
        
        if(r > 0 and grid[r-1,c] != WALL):            #Move up
            actions.append(((-1,0),1))

        if(r < num_rows-1 and grid[r+1,c] != WALL):      #Move Down
            actions.append(((1,0),1))

        if(r > 0 and c < num_cols-1 and grid[r-1, c+1] != WALL ):   # Top Right
            actions.append(((-1,1),1))

        
        if(r > 0 and c > 0 and grid[r-1, c-1] != WALL):      # Top Left
            actions.append(((-1,-1),1))

        if(r < num_rows-1 and c < num_cols-1 and grid[r+1,c+1] != WALL):       # Bottom Right
            actions.append(((1,1),1))

        if(r < num_rows-1 and c > 0 and grid[r+1,c-1] !=WALL):             #Bottom Left
            actions.append(((1,-1),1))

        if(p == 0):
            actions = [((0,0),1)]

        
        actions.append(((0,0),1))

        return actions
    
    def perform_action(self, state, action):
        grid, r, c, p = self.unpack(state)
        dr, dc = action

        
        #r, c = r + dr, c + dc
        r += dr
        c += dc
        if (dr,dc) == (0,0):

            if grid[r,c] == DIRTY and p > 0:
                p -= 1
                grid[r,c] = CLEAN

                
            if grid[r,c] == CHARGER: 
                if p < self.max_power:
                    p += 1
            
        else: p -= 1
        
        new_state = self.pack(grid, r, c, p)
        return new_state

    def is_goal(self, state):
        grid, r, c, p = self.unpack(state)

        result = (grid != DIRTY).all()

        result = result and (grid[r,c] == CHARGER)

        return result

    def simple_heuristic(self, state):
        grid, r, c, p = self.unpack(state)

        # get list of dirty positions
        # dirty[k] has the form (i, j)
        # where (i, j) are the row and column position of the kth dirty cell
        dirty = list(zip(*np.nonzero(grid == DIRTY)))

        # if no positions are dirty, estimate zero remaining cost to reach a goal state
        if len(dirty) == 0: return 0

        # otherwise, get the distance from the roomba to each dirty square
        dists = [max(np.fabs(dr-r), np.fabs(dc-c)) for (dr, dc) in dirty]

        # estimate the remaining cost to goal as the largest distance to a dirty position
        return int(max(dists))

    def better_heuristic(self, state):

        ### TODO: Implement a "better" heuristic than simple_heuristic
        # "Better" means more memory-efficient (fewer popped nodes during A* search)

        

        return 0

if __name__ == "__main__":


    # set up initial state by making five random open positions dirty
    domain = RoombaDomain()
    init = domain.initial_state(
        roomba_position = (0, 0),
        dirty_positions = np.random.permutation(list(zip(*np.nonzero(domain.grid == CLEAN))))[:5])

    problem = SearchProblem(domain, init, domain.is_goal)

    # compare runtime and node count for BFS, A* with simple heuristic, and A* with better heuristic
    start = perf_counter()
    plan, node_count = breadth_first_search(problem)
    bfs_time = perf_counter() - start
    print("bfs_time", bfs_time)
    print("node count", node_count)

    start = perf_counter()
    plan, node_count = a_star_search(problem, domain.simple_heuristic)
    astar_time = perf_counter() - start
    print("astar_time", astar_time)
    print("node count", node_count)

    start = perf_counter()
    plan, node_count = a_star_search(problem, domain.better_heuristic)
    astar_time = perf_counter() - start
    print("better heuristic:")
    print("astar_time", astar_time)
    print("node count", node_count)

    # reconstruct the intermediate states along the plan
    states = [problem.initial_state]
    for a in range(len(plan)):
        states.append(domain.perform_action(states[-1], plan[a]))

    # Animate the plan

    fig = pt.figure(figsize=(8,8))

    def drawframe(n):
        pt.cla()
        domain.render(pt.gca(), states[n])

    # blit=True re-draws only the parts that have changed.
    anim = animation.FuncAnimation(fig, drawframe, frames=len(states), interval=500, blit=False)
    pt.show()