import pathlib
from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags
from tqdm import trange

import os
import cv2
import datetime

import ogbench.manipspace  # noqa
from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle
from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
from ogbench.manipspace.oracles.plan.drawer_plan import DrawerPlanOracle
from ogbench.manipspace.oracles.plan.window_plan import WindowPlanOracle

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-single-v0', 'Environment name.')
# flags.DEFINE_string('dataset_type', 'play', 'Dataset type.')
# flags.DEFINE_string('save_path', None, 'Save path.')
flags.DEFINE_float('noise', 0.0, 'Action noise level.')
flags.DEFINE_float('noise_smoothing', 0.5, 'Action noise smoothing level for PlanOracle.')
# flags.DEFINE_float('min_norm', 0.4, 'Minimum action norm for MarkovOracle.')
# flags.DEFINE_float('p_random_action', 0, 'Probability of selecting a random action.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes.')
flags.DEFINE_integer('max_episode_steps', 1001, 'Number of episodes.')
flags.DEFINE_bool('render', False, 'Whether to render and save videos.')


def main(_):
    # assert FLAGS.dataset_type in ['play', 'noisy']
    # 'play': Use a non-Markovian oracle (PlanOracle) that follows a pre-computed plan.
    # 'noisy': Use a Markovian, closed-loop oracle (MarkovOracle) with Gaussian action noise.

    # Initialize environment.
    env = gymnasium.make(
        FLAGS.env_name,
        terminate_at_goal=False,
        mode='data_collection',
        max_episode_steps=FLAGS.max_episode_steps,
    )

    # Initialize oracles.
    oracle_type = 'plan'
    has_button_states = hasattr(env.unwrapped, '_cur_button_states')
    agents = {
        'cube': CubePlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
    }

    # Collect data.
    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10
    frames_traj = []
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        # Have an additional while loop to handle rare cases with undesirable states (for the Scene environment).
        while True:
            ob, info = env.reset()

            # input exact states
            custom_ee_pos = np.array([0.425, 0.0, 0.2])
            custom_ee_yaw = 0.0  # Radians
            custom_cubes = [np.array([0.425, 0.1, 0.02]),
                            np.array([0.35, -0.1, 0.02]),
                            np.array([0.35, -0.1, 0.06])]
            ob, info = env.unwrapped.reset_to_custom_state(
                ee_pos=custom_ee_pos,
                ee_yaw=custom_ee_yaw,
                cube_poses=custom_cubes
            )
            agent_ob, agent_info = env.unwrapped.set_custom_target(
                custom_target_block=0,
                custom_tar_pos=np.array([0.35, -0.1, 0.10]),
                custom_tar_ori=[1.0, 0.0, 0.0, 0.0]  # Identity quaternion (w, x, y, z)
            )

            # Set the cube stacking probability for this episode.
            if 'single' in FLAGS.env_name:
                p_stack = 0.0
            elif 'double' in FLAGS.env_name:
                p_stack = np.random.uniform(0.0, 0.25)
            elif 'triple' in FLAGS.env_name:
                p_stack = np.random.uniform(0.05, 0.35)
            elif 'quadruple' in FLAGS.env_name:
                p_stack = np.random.uniform(0.1, 0.5)
            elif 'octuple' in FLAGS.env_name:
                p_stack = np.random.uniform(0.0, 0.35)
            else:
                p_stack = 0.5

            if oracle_type == 'markov':
                # Set the action noise level for this episode.
                xi = np.random.uniform(0, FLAGS.noise)

            agent = agents[info['privileged/target_task']]
            agent.reset(agent_ob, agent_info)

            done = False
            step = 0
            ep_qpos = []

            custom_target = True

            while not done:
                if FLAGS.render:  # and (step % FLAGS.video_frame_skip == 0):
                    frame = env.render().copy()
                    frames_traj.append(frame)

                action = agent.select_action(ob, info)
                action = np.array(action)
                action = np.clip(action, -1, 1)
                next_ob, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if agent.done:
                    # Set a new task when the current task is done.

                    # input exact states
                    custom_ee_pos = np.array([0.425, 0.0, 0.2])
                    custom_ee_yaw = 0.0  # Radians
                    custom_cubes = [np.array([0.425, 0.1, 0.02]),
                                    np.array([0.35, -0.1, 0.02]),
                                    np.array([0.35, -0.1, 0.06])]

                    # Call your new custom reset method
                    ob, info = env.unwrapped.reset_to_custom_state(
                        ee_pos=custom_ee_pos,
                        ee_yaw=custom_ee_yaw,
                        cube_poses=custom_cubes
                    )
                    if custom_target:
                        agent_ob, agent_info = env.unwrapped.set_new_target(p_stack=p_stack)
                        custom_target = False
                    else:
                        agent_ob, agent_info = env.unwrapped.set_custom_target(
                            custom_target_block=0,
                            custom_tar_pos=np.array([0.35, -0.1, 0.10]),
                            custom_tar_ori=[1.0, 0.0, 0.0, 0.0]  # Identity quaternion (w, x, y, z)
                        )
                        custom_target = True
                    agent = agents[agent_info['privileged/target_task']]
                    agent.reset(agent_ob, agent_info)

                dataset['observations'].append(ob)
                dataset['actions'].append(action)
                dataset['terminals'].append(done)
                dataset['qpos'].append(info['prev_qpos'])
                dataset['qvel'].append(info['prev_qvel'])
                if has_button_states:
                    dataset['button_states'].append(info['prev_button_states'])
                ep_qpos.append(info['prev_qpos'])

                ob = next_ob
                step += 1

            break

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step

    print('Total steps:', total_steps)
    #
    # train_path = FLAGS.save_path
    # val_path = FLAGS.save_path.replace('.npz', '-val.npz')
    # pathlib.Path(train_path).parent.mkdir(parents=True, exist_ok=True)

    # Split the dataset into training and validation sets.
    # train_dataset = {}
    # val_dataset = {}
    # for k, v in dataset.items():
    #     if 'observations' in k and v[0].dtype == np.uint8:
    #         dtype = np.uint8
    #     elif k == 'terminals':
    #         dtype = bool
    #     elif k == 'button_states':
    #         dtype = np.int64
    #     else:
    #         dtype = np.float32
    #     train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
    #     val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)
    #
    # for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
    #     np.savez_compressed(path, **dataset)

    if FLAGS.render:
        videos_dir = os.path.join('./videos')
        os.makedirs(videos_dir, exist_ok=True)
        stamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        episode_path = os.path.join(videos_dir, f'video_{FLAGS.env_name}.mp4')
        height, width, _ = frames_traj[0].shape
        video_writer = cv2.VideoWriter(episode_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        # breakpoint()

        for i in range(len(frames_traj)):
            # episode_frames = frames_traj[i]
            # episode_name = (f'epi{i}_{stamp_str}.mp4')
            # episode_path = os.path.join(videos_dir, stamp_str)
            # height, width, _ = frames_traj[i].shape
            # video_writer = cv2.VideoWriter(episode_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
            # for frame in episode_frames:
            video_writer.write(cv2.cvtColor(frames_traj[i], cv2.COLOR_RGB2BGR))
        video_writer.release()
        print('Saved video to: ', videos_dir)


if __name__ == '__main__':
    app.run(main)
