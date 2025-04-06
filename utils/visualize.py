# coding:utf8

import time

import numpy as np
import visdom


class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    """

    def __init__(self, env='default', **kwargs):
        self.vis = None
        try:
            self.vis = visdom.Visdom(env=env, **kwargs)
            print("Visdom connected successfully")
        except Exception as e:
            print(f"Could not connect to Visdom server: {e}")
            print("Training will continue without visualization")

    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        依次plot多个
        :param d: dict (name, values)
        :return:
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss', 1.00)
        """
        if self.vis is None:
            return
        try:
            x = np.array([time.time()])
            y = np.array([y])
            self.vis.line(Y=y, X=x,
                         win=name,
                         opts=dict(title=name),
                         update=None if not hasattr(self, name) else 'append',
                         **kwargs
                         )
            setattr(self, name, True)
        except Exception as e:
            print(f"Failed to plot {name}: {e}")

    def img(self, name, img_, **kwargs):
        self.vis.image(img_.cpu().numpy(),
                       win=(name),
                       opts=dict(title=name),
                       **kwargs
                       )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """
        if self.vis is None:
            print(info)
            return
        try:
            self.vis.text(info, win)
        except Exception as e:
            print(f"Failed to log: {e}")
            print(info)

    def __getattr__(self, item):
        return getattr(self.vis, item)
