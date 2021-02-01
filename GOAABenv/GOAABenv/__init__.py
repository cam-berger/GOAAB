from gym.envs.registration import register

register(
     id='GOAABenv-v0',
     entry_point='GOAABenv.envs.GOAABenv:GOAABenv',
     max_episode_steps=1000,
     kwargs={'rows' : 1000, 'file_name' : "Botnet2014.csv"}
)
