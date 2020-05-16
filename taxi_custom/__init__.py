import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='TaxiCustom-v0',
    entry_point='taxi_custom.envs:TaxiCustomEnv',
    reward_threshold=8, # optimum = 8.46
    max_episode_steps=200,
)
