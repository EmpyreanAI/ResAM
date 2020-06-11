from gym.envs.registration import register

register(
    id='MarketEnv-v0',
    entry_point='gym_market.envs:MarketEnv',
)
# register(
#     id='market-extrahard-v0',
#     entry_point='gym_market.envs:MarketExtraHardEnv',
# )
