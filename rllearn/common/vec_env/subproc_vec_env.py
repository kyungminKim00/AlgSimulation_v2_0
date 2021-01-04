from multiprocessing import Process, Pipe

import numpy as np

from rllearn.common.vec_env import VecEnv, CloudpickleWrapper
from rllearn.common.tile_images import tile_images


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments

    :param env_fns: ([Gym Environment]) Environments to run in subprocesses
    """

    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.processes = [Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                          for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for process in self.processes:
            process.daemon = True  # if the main process crashes, we should not cause things to hang
            process.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self, mode='human', *args, **kwargs):
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', (args, {'mode': 'rgb_array', **kwargs})))
        imgs = [pipe.recv() for pipe in self.remotes]
        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        for pipe in self.remotes:
            pipe.send(('render', {"mode": 'rgb_array'}))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def env_method(self, method_name, *method_args, **method_kwargs):
        """
        Provides an interface to call arbitrary class methods of vectorized environments

        :param method_name: (str) The name of the env class method to invoke
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items retured by each environment's method call
        """

        for remote in self.remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in self.remotes]

    def get_attr(self, attr_name):
        """
        Provides a mechanism for getting class attribues from vectorized environments
        (note: attribute value returned must be picklable)

        :param attr_name: (str) The name of the attribute whose value to return
        :return: (list) List of values of 'attr_name' in all environments
        """

        for remote in self.remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in self.remotes]

    def set_attr(self, attr_name, value, indices=None):
        """
        Provides a mechanism for setting arbitrary class attributes inside vectorized environments
        (note:  this is a broadcast of a single value to all instances)
        (note:  the value must be picklable)

        :param attr_name: (str) Name of attribute to assign new value
        :param value: (obj) Value to assign to 'attr_name'
        :param indices: (list,tuple) Iterable containing indices of envs whose attr to set
        :return: (list) in case env access methods might return something, they will be returned in a list
        """

        if indices is None:
            indices = range(len(self.remotes))
        elif isinstance(indices, int):
            indices = [indices]
        for remote in [self.remotes[i] for i in indices]:
            remote.send(('set_attr', (attr_name, value)))
        return [remote.recv() for remote in [self.remotes[i] for i in indices]]
