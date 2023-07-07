import numpy as np
import pandas as pd
from functools import reduce
import plotly.express as px

"""
 Wilton Beltré
"""
class VAD:
    def __init__(self, minmax=[1, 5], increase_interv=0.):
        """
        Simple mapper class from vad model to categorical.
        based on paper: Evidence for a Three-Factor Theory of Emotions.

        """
        data = np.genfromtxt('./categorial_vad.csv', delimiter=',', usecols=(2, 3, 4, 5, 6, 7))
        self.terms = np.genfromtxt('./categorial_vad.csv', delimiter=',', usecols=(0), dtype=None, encoding='utf-8-sig')

        self.valence_mean = data[:, 0]
        self.valence_std = data[:, 1]
        self.arousal_mean = data[:, 2]
        self.arousal_std = data[:, 3]
        self.dominance_mean = data[:, 4]
        self.dominance_std = data[:, 5]

        self.minmax = minmax
        self.increase_interv = increase_interv
        self.to_plot = None

    def plot(self):
        df = self.to_plot
        fig = px.scatter_3d(df, x='Valence', y='Arousal', z='Dominance',
                            color='Terms', symbol='Terms', text='Closest', size='ivClosest')
        fig.show()

    def __normalize(self, value: int) -> set:
        """
        normalize interval [a, b] to [-1, 1] to easy map to vad model.
        also add min interval, max interval proportional
        :param value:
        :param minmax:
        :return: min: interval, norm: middle interval, max: inverval
        """
        norm = 2 * ((value - self.minmax[0]) / (self.minmax[1] - self.minmax[0])) - 1
        max = norm + (1./(self.minmax[1]-1)) + (norm * self.increase_interv)
        min = norm - (1./(self.minmax[1]-1)) - (norm * self.increase_interv)
        return min, norm, max

    def __intervals(self, v: int, a: int, d: int) -> []:
        """
        prepare V,A,D intervals and normalized value
        :param v:
        :param a:
        :param d:
        :return: v_inter, a_inter, d_inter
        """
        v_inter: set = self.__normalize(v)
        a_inter: set = self.__normalize(a)
        d_inter: set = self.__normalize(d)
        return [v_inter, a_inter, d_inter]

    def vad2categorical(self, v: int, a: int, d: int):
        """
        Query intervals and return a list or posible mapping.
        Experimental: closeness to centroid mean
        :param v:
        :param a:
        :param d:
        :return:
        """
        interval = self.__intervals(v, a, d)

        valence = np.argwhere(
            (self.valence_mean > interval[0][0]) &
            (self.valence_mean < interval[0][2])
        ).flatten()
        v_mask = np.abs(self.valence_mean[valence] - interval[0][1])
        # valence = valence[v_mask.argsort()]

        arousal = np.argwhere(
            (self.arousal_mean > interval[1][0]) &
            (self.arousal_mean < interval[1][2])
        ).flatten()
        a_mask = np.abs(self.arousal_mean[arousal] - interval[1][1])
        # arousal = arousal[a_mask.argsort()]

        dominan = np.argwhere(
            (self.dominance_mean > interval[2][0]) &
            (self.dominance_mean < interval[2][2])
        ).flatten()
        d_mask = np.abs(self.dominance_mean[dominan] - interval[2][1])
        # dominan = dominan[d_mask.argsort()]

        inter_set = reduce(np.intersect1d, (valence, arousal, dominan))

        # Sometime dominance doesn't apport nothing :X
        found_dominance = True
        if len(inter_set) == 0:
            inter_set = np.intersect1d(valence, arousal)
            found_dominance = False

        ranking = []
        for i in inter_set:
            # v_idx = np.argwhere(valence == i).flatten().item()
            # a_idx = np.argwhere(arousal == i).flatten().item()
            # v_val = v_mask[v_idx]
            # a_val = a_mask[a_idx]
            v_val = self.valence_mean[i]
            a_val = self.arousal_mean[i]
            d_val = 0
            if found_dominance:
                # d_idx = np.argwhere(dominan == i).flatten().item()
                # d_val = d_mask[d_idx]
                d_val = self.dominance_mean[i]
                vad_orig = np.array([interval[0][1], interval[1][1], interval[2][1]])
                vad_dest = np.array([v_val, a_val, d_val])
                closest = np.linalg.norm(vad_orig - vad_dest)
            else:
                vad_orig = np.array([interval[0][1], interval[1][1]])
                vad_dest = np.array([v_val, a_val])
                closest = np.linalg.norm(vad_orig - vad_dest)

            ranking.append({'term': self.terms[i], 'closest': closest, 'v': v_val, 'a': a_val, 'd': d_val})

        ranking = sorted(ranking, key=lambda x: x['closest'])

        self.to_plot = pd.DataFrame({
            'Terms': [v['term'] for v in ranking],
            'Valence': [v['v'] for v in ranking],
            'Arousal': [a['a'] for a in ranking],
            'Dominance': [d['d'] for d in ranking],
            'Closest': [np.round(c['closest'], 4) for c in ranking],
        })
        self.to_plot['ivClosest'] = self.to_plot[['Closest']][::-1].reset_index(drop=True)

        return ranking, {'using_dominance': found_dominance}

