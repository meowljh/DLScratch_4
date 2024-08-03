import numpy as np
import matplotlib.pyplot as plt
import matplotlib

'''render.py
- function for visualization of the agent's actions
'''

class Renderer:
    def __init__(self, reward_map, goal_state, wall_state):
        self.reward_map = reward_map
        self.goal_state = goal_state
        self.wall_state = wall_state
        self.ys = len(self.reward_map)
        self.xs = len(self.reward_map[0])
        
        self.ax = None
        self.fig = None
        self.first_fig = True
    
    def save_figure(self, fpath):
        self.fig.savefig(fpath)
        
    def set_figure(self, figsize=None):
        fig = plt.figure(figsize=figsize)
        self.fig = fig
        self.ax = fig.add_subplot(111)
        ax = self.ax
        ax.clear()
        ax.set_title("Policy Evaluation Result")
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        ax.set_xticks(range(self.xs))
        ax.set_yticks(range(self.ys))
        ax.set_xlim(0, self.xs)
        ax.set_ylim(0, self.ys)
        ax.grid(True)
    
    def render_v(self, v=None, policy=None, print_value=True, fpath=None):
        '''
        cost function을 건네면 각 위치의 가치 함수의 값이 해당 칸의 우측 상단에 표시 됨
        가치 함수의 값의 크기에 따라서 색을 달리 하여 grid world를 heatmap의 형태로 나타내어,
        값이 작을수록 붉은색이 짙어지고 클수록 초록색이 짙어지게 된다.
        '''
        
        self.set_figure()
        ax = self.ax
        ys, xs = self.ys, self.xs 
        
        if v is not None:
            v_dict = v
            v = np.zeros((self.reward_map.shape))
            # 정책 함수가 별도로 주어진 경우에 #
            for state, reward in v_dict.items():
                v[state] = reward
            
            vmax, vmin = v.max(), v.min()
            vmax = max(vmax, abs(vmin))
            vmin = -1 * vmax
            vmax = np.clip(vmax, vmax, 1)
            vmin = np.clip(vmin, -1, vmin)
            
            # color_list = ['red', 'green', 'white']   
            ## cmap 정의 할 때 색의 순서가 실제 값의 최소 ~ 최대까지의 color range를 의미 ##
            color_list = ['red', 'white', 'green']     
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'colormap_name', color_list
            )
            
            ax.pcolormesh(np.flipud(v), cmap=cmap, vmin=vmin, vmax=vmax)
            
        for y in range(self.ys):
            for x in range(self.xs):
                state = (y, x)
                r = self.reward_map[state]
                if r != 0 and r is not None:
                    txt = f"R {r}"
                    if state == self.goal_state:
                        txt = f"{txt} (GOAL)"
                    ax.text(x + 0.1, ys-y-0.9, txt)
                if (v is not None) and state != self.wall_state:
                    if print_value:
                        offsets = [(0.4, -0.15), (-0.15, -0.3)]
                        key = 0
                        if v.shape[0] > 7:
                            key = 1
                        offset = offsets[key]
                        ax.text(x + offset[0], ys-y+offset[1], "{:12.2f}".format(v[y, x]))
                        
                # policy가 None인 경우에는 
                if (policy is not None) and state != self.wall_state:
                    actions = policy[state]
                    # policy에는 특정 state에서 action을 하였을 때의 reward가 저장 #
                    max_actions = [kv[0] for kv in actions.items() if kv[1] == max(actions.values())]
                    arrows = ["↑", "↓", "←", "→"]
                    offsets = [(0, 0.1), (0, -0.1), (-0.1, 0), (0.1, 0)]
                    for action in max_actions:
                        arrow = arrows[action]
                        offset = offsets[action]
                        if state == self.goal_state:
                            continue
                        ax.text(x + 0.45+offset[0], ys-y-0.5+offset[1], arrow)
                    
                if state == self.wall_state:
                    ax.add_patch(plt.Rectangle((x, ys-y-1), 1, 1, fc=(0.4, 0.4, 0.4, 1)))
        plt.show()
        
        if fpath is not None:
            self.save_figure(fpath)

                    