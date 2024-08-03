import numpy as np
import os, sys
from .grid_world_renderer import Renderer

class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3] # 행동 가능한 공간 #
        self.action_meaning = {
            0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"
        }
        self.reward_map = np.array( # 보상 map (각 좌표의 보상 값) #
            [[0, 0, 0, 1.0],
             [0, None, 0, -1.0],
             [0, 0, 0, 0]]
        )
        
        self.goal_state = (0, 3)  # 목표 좌표 #
        self.wall_state = (1, 1) # 벽 상태 좌표 #
        self.start_state = (2, 0) # 시작 상태 좌표 #
        self.agent_state = self.start_state # agent의 초기 상태 좌표 #
        
    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape
    
    def actions(self):
        return self.action_space
    
    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)
    
    def next_state(self, state, action):
        cur_y, cur_x = state
        DX, DY = [0, 0, -1, 1], [-1, 1, 0, 0]
        nx, ny = cur_x + DX[action], cur_y + DY[action]
        next_state = (ny, nx)
        if (nx < 0) or (ny < 0) or (nx >= self.width) or (ny >= self.height):
            next_state = state
        # elif nx == self.wall_state[1] and ny == self.wall_state[0]:
        elif next_state == self.wall_state:
            next_state = state
        else:
            next_state = (ny, nx)
        return next_state

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]
    
    def reset(self):
        # agent를 초기 상태롤 돌려주는 method #
        self.agent_state = self.start_state
        return self.agent_state
    
    def step(self, action):
        # agent에게 action이라는 행동을 시켜서 시간을 한 단계 진행 시킴 #
        # DX, DY = [0, 0, -1, 1], [-1, 1, 0, 0]
        # dx, dy = DX[action], DY[action]
        next_state = self.next_state(self.agent_state, action)
        reward = self.reward(self.agent_state, action, next_state)
        is_done = (next_state == self.goal_state) ## 목표를 달성 하였는지 확인 ##
        self.agent_state = next_state
        
        return next_state, reward, is_done

    def render_v(self, v=None, policy=None, print_value=True, fpath=None):
        # 상태 가치 함수, 즉 v가 따로 주어지면 나중에 이걸 사용해서 reward_map 대신에 가치 계산에 사용 #
        renderer = Renderer(self.reward_map,
                                          self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value, fpath)
        
