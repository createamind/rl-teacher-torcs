import math
from multiprocessing import Pool
import numpy as np
import gym.spaces.prng as space_prng
from autodrive.agent.torcs2 import AgentTorcs2
# from rl_teacher.envs import get_timesteps_per_episode
from drlutils.utils import logger
def _slice_path(path, segment_length, start_pos=0):
    seg={
        k: np.asarray(v[start_pos:(start_pos + segment_length)])
        for k, v in path.items()
        if k in ['obs', "actions", 'original_rewards', 'human_obs','distances']}
    seg['maxdistance']=seg['distances'].max()
    return seg

def create_segment_q_states(segment):
    obs_Ds = segment["obs"]
    act_Ds = segment["actions"]
    return np.concatenate([obs_Ds, act_Ds], axis=1)

def sample_segment_from_path(path, segment_length):
    """Returns a segment sampled from a random place in a path. Returns None if the path is too short"""
    path_length = len(path["obs"])
    if path_length < segment_length:
        return None

    start_pos = np.random.randint(0, path_length - segment_length + 1)

    # Build segment
    segment = _slice_path(path, segment_length, start_pos)

    # Add q_states
    segment["q_states"] = create_segment_q_states(segment)
    return segment

def random_action(env, ob):
    """ Pick an action by uniformly sampling the environment's action space. """
    mu, sigma = 0, 1
    steering = env._rng.normal(loc=mu, scale=sigma, size=1)
    # steering = np.random.normal(loc=mu, scale=sigma, size=1)
    acc=np.array([0.5])
    action=np.hstack([steering,acc ])

    return action

def do_rollout(env, action_function):
    """ Builds a path by running through an environment using a provided function to select actions. """
    obs, rewards, actions, human_obs ,distances= [], [], [], [],[]
    max_timesteps_per_episode = 10000
    raw_ob = env.reset()
    logger.info('get obs {}'.format(raw_ob.shape))
    ob=raw_ob[:-1]
    distance=raw_ob[-1]
    # Primary environment loop
    for i in range(max_timesteps_per_episode):
        action = action_function(env, ob)
        obs.append(ob)
        raw_ob, action, reward, done = env.step((action, 0., [0., 0.], [0., 0.]))
        ob=raw_ob[0:-1]
        distance=raw_ob[-1]
        # logger.info('agent {} running at distance {}'.format(env._agentIdent,distance))
        actions.append(action)
        rgb=env._cur_screen

        # logger.info("[{:04d}: step".format(env._agentIdent))
        rewards.append(reward)
        human_obs.append(rgb)
        distances.append(distance)
        if done:
            break
    # Build path dictionary
    path = {
        "obs": np.array(obs),
        "original_rewards": np.array(rewards),
        "actions": np.array(actions),
        "human_obs": np.array(human_obs),
         'distances':np.array((distances))
         }
    return path

def basic_segments_from_rand_rollout(n_desired_segments, clip_length_in_seconds,
    # These are only for use with multiprocessing
    env_id=0, _verbose=True, _multiplier=1
):
    """ Generate a list of path segments by doing random rollouts. No multiprocessing. """
    segments = []

    env = AgentTorcs2(env_id, bots=['scr_server'], track='road/g-track-1', text_mode=False, laps=3,
                            torcsIdxOffset=0, screen_capture=True)
        # agent = AgentTorcs2(aidx, bots=['scr_server', 'olethros', 'berniw', 'bt', 'damned'], track='road/g-track-1', text_mode=True)
    env.reset()


    segment_length = int(clip_length_in_seconds * 24)
    while len(segments) < n_desired_segments:
        path = do_rollout(env, random_action)
        # Calculate the number of segments to sample from the path
        # Such that the probability of sampling the same part twice is fairly low.
        segments_for_this_path = max(1, int(0.25 * len(path["obs"]) / segment_length))
        for _ in range(segments_for_this_path):
            segment = sample_segment_from_path(path, segment_length)
            if segment:
                segments.append(segment)

            if _verbose and len(segments) % 10 == 0 and len(segments) > 0:
                print("Collected %s/%s segments" % (len(segments) * _multiplier, n_desired_segments * _multiplier))

    if _verbose:
        print("Successfully collected %s segments" % (len(segments) * _multiplier))
    return segments

def segments_from_rand_rollout( n_desired_segments, clip_length_in_seconds, workers):
    """ Generate a list of path segments by doing random rollouts. Can use multiple processes. """
    if workers < 2:  # Default to basic segment collection
        return basic_segments_from_rand_rollout( n_desired_segments, clip_length_in_seconds)

    pool = Pool(processes=workers)
    segments_per_worker = int(math.ceil(n_desired_segments / workers))
    # One job per worker. Only the first worker is verbose.
    jobs = [
        ( segments_per_worker, clip_length_in_seconds, i, i == 0, workers)
        for i in range(workers)]
    results = pool.starmap(basic_segments_from_rand_rollout, jobs)
    pool.close()
    return [segment for sublist in results for segment in sublist]
