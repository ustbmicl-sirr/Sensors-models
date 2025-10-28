"""
RLlib多智能体泊位分配环境
"""
from typing import Dict, Optional, Tuple
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import sys
import os

# 添加父目录到路径以导入现有模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.vessel import Vessel, VesselGenerator
from environment.shore_power import ShorePowerManager
from rewards.reward_calculator import RewardCalculator


class BerthAllocationMultiAgentEnv(MultiAgentEnv):
    """
    泊位分配多智能体环境 - RLlib适配

    特性:
    - 每艘船是一个智能体
    - 连续动作空间 (3维): [position, wait_time, shore_power_prob]
    - 部分可观测 (POMDP): 每艘船只看到局部信息
    - 个体奖励: 每艘船独立的奖励函数
    """

    def __init__(self, config: dict):
        super().__init__()

        # 环境参数
        self.berth_length = config.get('berth_length', 2000)
        self.max_vessels = config.get('max_vessels', 20)
        self.planning_horizon = config.get('planning_horizon', 168)  # 7天 x 24小时
        self.max_wait_time = config.get('max_wait_time', 48.0)  # 48小时

        # 船舶生成器配置
        self.generation_mode = config.get('generation_mode', 'simple')
        vessel_gen_config = {
            'berth_length': self.berth_length,
            'max_vessels': self.max_vessels,
            'planning_horizon': self.planning_horizon,
        }
        self.vessel_generator = VesselGenerator(vessel_gen_config)

        # 岸电管理器
        if config.get('shore_power_enabled', True):
            # 生成岸电段配置
            num_segments = config.get('num_shore_power_segments', 5)
            segment_length = self.berth_length / num_segments
            capacity_per_segment = config.get('shore_power_capacity', 500)

            segments = []
            for i in range(num_segments):
                segments.append({
                    'start': i * segment_length,
                    'end': (i + 1) * segment_length,
                    'capacity': capacity_per_segment,
                })

            shore_power_config = {
                'segments': segments,
                'emission_factor_ship': 700.0,  # kg CO2/MWh
                'emission_factor_shore': 500.0,  # kg CO2/MWh
            }
            self.shore_power_manager = ShorePowerManager(shore_power_config)
        else:
            self.shore_power_manager = None

        # 奖励计算器
        reward_config = {
            'berth_length': self.berth_length,
            'planning_horizon': self.planning_horizon / 24,  # 转换为天
            'max_wait_time': self.max_wait_time,
            'safe_distance': config.get('safe_distance', 20.0),
            'rewards': config.get('reward_weights', {}),
            'shore_power': {
                'emission_factor_ship': 700.0,
                'emission_factor_shore': 500.0,
            }
        }
        self.reward_calculator = RewardCalculator(reward_config)

        # 环境状态
        self.vessels = []
        self.allocations = []
        self.current_step = 0
        self.current_time = 0.0

        # 定义观测空间 (17维局部观测)
        self._obs_space = Box(
            low=-1.0,
            high=1.0,
            shape=(17,),
            dtype=np.float32
        )

        # 定义动作空间 (3维连续动作)
        # [position, wait_time, shore_power_prob]
        self._action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # RLlib要求的智能体ID集合
        self._agent_ids = set()

        # 记录已分配的智能体
        self.allocated_agents = set()

    @property
    def observation_space(self):
        """返回观测空间"""
        return self._obs_space

    @property
    def action_space(self):
        """返回动作空间"""
        return self._action_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """
        重置环境

        Returns:
            observations: Dict[agent_id, obs]
            infos: Dict[agent_id, info]
        """
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)

        # 生成船舶
        if options and 'vessels' in options:
            self.vessels = options['vessels']
        else:
            # 根据generation_mode调用正确的方法
            planning_horizon_days = self.planning_horizon / 24.0

            if self.generation_mode == 'realistic':
                self.vessels = self.vessel_generator.generate_realistic(
                    num_vessels=self.max_vessels,
                    planning_horizon_days=planning_horizon_days
                )
            else:
                self.vessels = self.vessel_generator.generate_simple(
                    num_vessels=self.max_vessels,
                    planning_horizon_days=planning_horizon_days
                )

        # 重置状态
        self.allocations = []
        self.current_step = 0
        self.current_time = 0.0
        self.allocated_agents = set()

        # 设置智能体ID（每艘船一个ID）
        self._agent_ids = {f"vessel_{i}" for i in range(len(self.vessels))}

        # 获取初始观测
        observations = self._get_observations()
        infos = {agent_id: {} for agent_id in self._agent_ids}

        return observations, infos

    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        执行动作

        Args:
            action_dict: Dict[agent_id, action]
                action: [position, wait_time, shore_power_prob]
                       范围: [-1, 1]

        Returns:
            observations: Dict[agent_id, obs]
            rewards: Dict[agent_id, float]
            terminateds: Dict[agent_id, bool]
            truncateds: Dict[agent_id, bool]
            infos: Dict[agent_id, dict]
        """
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}

        # 处理每个智能体的动作
        for agent_id, action in action_dict.items():
            if agent_id in self.allocated_agents:
                continue  # 跳过已分配的智能体

            vessel_idx = int(agent_id.split('_')[1])
            vessel = self.vessels[vessel_idx]

            # 解码动作 (从[-1,1]映射到实际范围)
            position = self._decode_position(action[0])
            wait_time = self._decode_wait_time(action[1])
            shore_power_prob = self._decode_shore_power(action[2])

            # 决定是否使用岸电
            uses_shore_power = (
                vessel.can_use_shore_power and
                np.random.random() < shore_power_prob
            )

            # 计算实际泊位时间
            berthing_time = vessel.arrival_time + wait_time
            departure_time = berthing_time + vessel.operation_time

            # 创建分配
            allocation = {
                'vessel': vessel,
                'vessel_id': vessel.id,
                'position': position,
                'berthing_time': berthing_time,
                'departure_time': departure_time,
                'uses_shore_power': uses_shore_power,
                'waiting_time': wait_time,
                'shore_power_segments': [],  # 稍后填充
            }

            # 添加到分配列表
            self.allocations.append(allocation)
            self.allocated_agents.add(agent_id)

            # 计算奖励
            env_state = {
                'allocations': self.allocations,
                'current_time': self.current_time,
                'shore_power_manager': self.shore_power_manager,
            }
            reward = self.reward_calculator.calculate(
                vessel,
                allocation,
                env_state
            )

            # 获取观测（下一状态）
            obs = self._get_observation(vessel_idx)

            # 填充返回值
            observations[agent_id] = obs
            rewards[agent_id] = float(reward)
            terminateds[agent_id] = True  # 该智能体已完成决策
            truncateds[agent_id] = False
            infos[agent_id] = {
                'allocated': True,
                'position': float(position),
                'berthing_time': float(berthing_time),
                'departure_time': float(departure_time),
                'uses_shore_power': bool(uses_shore_power),
                'waiting_time': float(wait_time),
            }

        # 更新时间步
        self.current_step += 1

        # 检查是否所有船舶都已分配
        if len(self.allocated_agents) >= len(self.vessels):
            # Episode结束
            terminateds['__all__'] = True
            truncateds['__all__'] = False
        else:
            terminateds['__all__'] = False
            truncateds['__all__'] = False

        return observations, rewards, terminateds, truncateds, infos

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """获取所有智能体的观测"""
        observations = {}
        for i in range(len(self.vessels)):
            agent_id = f"vessel_{i}"
            if agent_id not in self.allocated_agents:
                observations[agent_id] = self._get_observation(i)
        return observations

    def _get_observation(self, vessel_idx: int) -> np.ndarray:
        """
        获取单个智能体的观测 (17维)

        观测内容:
        - 静态特征 (4): 船长, 到港时间, 优先级, 岸电能力
        - 动态特征 (3): 当前时间, 等待时间, 操作时间
        - 岸电信息 (6): 5段使用率 + 总使用率
        - 泊位信息 (4): 左右邻近距离, 可用空间, 占用率
        """
        vessel = self.vessels[vessel_idx]

        # 静态特征（归一化到[-1, 1]或[0, 1]）
        vessel_length = (vessel.length / self.berth_length) * 2 - 1  # [-1, 1]
        arrival_time = (vessel.arrival_time / self.planning_horizon) * 2 - 1
        priority = (vessel.priority / 4.0) * 2 - 1
        shore_power_cap = 1.0 if vessel.can_use_shore_power else -1.0

        # 动态特征
        current_time = (self.current_time / self.planning_horizon) * 2 - 1
        waiting_time = max(0, self.current_time - vessel.arrival_time) / self.max_wait_time * 2 - 1
        operation_time = (vessel.operation_time / self.planning_horizon) * 2 - 1

        # 岸电使用率（归一化到[-1, 1]）
        if self.shore_power_manager:
            try:
                shore_power_usage = self.shore_power_manager.get_usage_rates()
                shore_power_usage = [u * 2 - 1 for u in shore_power_usage]  # [0, 1] -> [-1, 1]
            except:
                shore_power_usage = [-1.0] * 6
        else:
            shore_power_usage = [-1.0] * 6

        # 泊位信息（简化版）
        left_distance = 0.0   # TODO: 计算左侧最近船舶距离
        right_distance = 0.0  # TODO: 计算右侧最近船舶距离
        available_space = 1.0 - (len(self.allocations) / max(self.max_vessels, 1)) * 2  # [-1, 1]
        berth_occupancy = (len(self.allocations) / max(self.max_vessels, 1)) * 2 - 1

        # 组合观测
        obs = np.array([
            vessel_length,
            arrival_time,
            priority,
            shore_power_cap,
            current_time,
            waiting_time,
            operation_time,
            *shore_power_usage,  # 6个值
            left_distance,
            right_distance,
            available_space,
            berth_occupancy,
        ], dtype=np.float32)

        # 确保观测在[-1, 1]范围内
        obs = np.clip(obs, -1.0, 1.0)

        return obs

    def _decode_position(self, action_value: float) -> float:
        """解码位置动作"""
        # action_value ∈ [-1, 1]
        # position ∈ [0, berth_length]
        position = (action_value + 1) * self.berth_length / 2
        return float(np.clip(position, 0, self.berth_length))

    def _decode_wait_time(self, action_value: float) -> float:
        """解码等待时间动作"""
        # action_value ∈ [-1, 1]
        # wait_time ∈ [0, max_wait_time]
        wait_time = (action_value + 1) * self.max_wait_time / 2
        return float(np.clip(wait_time, 0, self.max_wait_time))

    def _decode_shore_power(self, action_value: float) -> float:
        """解码岸电概率"""
        # action_value ∈ [-1, 1]
        # probability ∈ [0, 1]
        prob = (action_value + 1) / 2
        return float(np.clip(prob, 0, 1))

    def render(self, mode='human'):
        """渲染环境（可选）"""
        if mode == 'human':
            print(f"\n=== 步骤 {self.current_step} ===")
            print(f"已分配船舶: {len(self.allocations)}/{len(self.vessels)}")
            if self.allocations:
                last_alloc = self.allocations[-1]
                print(f"最新分配: 船舶{last_alloc['vessel_id']} -> "
                      f"位置{last_alloc['position']:.1f}m, "
                      f"等待{last_alloc['waiting_time']:.1f}h")

    def close(self):
        """清理资源"""
        pass
