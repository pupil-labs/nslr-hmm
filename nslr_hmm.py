# Copyleft 2017 Jami Pekkanen <jami.pekkanen@gmail.com>.
# Released under AGPL-3.0, see LICENSE.

import numpy as np
import scipy.stats
import nslr
import itertools

class ObservationModel:
    def __init__(self, dists):
        self.classidx = {}
        self.idxclass = []
        self.dists = []
        for i, (cls, dist) in enumerate(dists.items()):
            self.idxclass.append(cls)
            self.classidx[cls] = i
            self.dists.append(dist)
        self.idxclass = np.array(self.idxclass)
    
    def liks(self, d):
        scores = []
        scores = [dist.pdf(d) for dist in self.dists]
        return np.array(scores).T
    
    def classify(self, d):
        return np.argmax(self.liks(d), axis=1)

    def dist(self, cls):
        return self.dists[self.classidx[cls]]

FIXATION = 1
SACCADE = 2
PSO = 3
SMOOTH_PURSUIT = 4

def gaze_observation_model():
    # Estimated from data by doi:10.3758/s13428-016-0738-9.
    params = {
        FIXATION: [[0.6039844795867605, -0.7788440631929878], [[0.1651734722683456, 0.0], [0.0, 1.5875256060544993]]],
        SACCADE: [[2.3259276064858194, 1.1333265634427712], [[0.080879690559802, 0.0], [0.0, 2.0718979621084372]]],
        PSO: [[1.7511546389160744, -1.817487032170937], [[0.0752678429860497, 0.0], [0.0, 1.356411391040218]]],
        SMOOTH_PURSUIT: [[0.8175021916433242, 0.3047120126632254], [[0.13334607025750783, 0.0], [0.0, 2.5328705587328173]]]
    }
        
    dists = {
        cls: scipy.stats.multivariate_normal(m, c)
        for cls, (m, c) in params.items()
    }
    
    return ObservationModel(dists)


def gaze_transition_model():
    transitions = np.ones((4, 4))
    transitions[0, 2] = 0
    transitions[2, 1] = 0
    transitions[3, 2] = 0
    transitions[3, 0] = 0.5
    transitions[0, 3] = 0.5
    for i in range(len(transitions)):
        transitions[i] /= np.sum(transitions[i])
    return transitions

GazeObservationModel = gaze_observation_model()
GazeTransitionModel = gaze_transition_model()

def safelog(x):
    return np.log10(np.clip(x, 1e-6, None))

def viterbi(initial_probs, transition_probs, emissions):
    n_states = len(initial_probs)
    emissions = iter(emissions)
    emission = next(emissions)
    transition_probs = safelog(transition_probs)
    probs = safelog(emission) + safelog(initial_probs)
    state_stack = []
    
    for emission in emissions:
        emission /= np.sum(emission)
        trans_probs = transition_probs + np.row_stack(probs)
        most_likely_states = np.argmax(trans_probs, axis=0)
        probs = safelog(emission) + trans_probs[most_likely_states, np.arange(n_states)]
        state_stack.append(most_likely_states)
    
    state_seq = [np.argmax(probs)]

    while state_stack:
        most_likely_states = state_stack.pop()
        state_seq.append(most_likely_states[state_seq[-1]])

    state_seq.reverse()

    return state_seq

def forward_backward(transition_probs, observations, initial_probs=None):
    observations = np.array(list(observations))
    N = len(transition_probs)
    T = len(observations)
    if initial_probs is None:
        initial_probs = np.ones(N)
        initial_probs /= np.sum(initial_probs)
    
    forward_probs = np.zeros((T, N))
    backward_probs = forward_probs.copy()
    probs = initial_probs
    for i in range(T):
        probs = np.dot(probs, transition_probs)*observations[i]
        probs /= np.sum(probs)
        forward_probs[i] = probs
    
    probs = np.ones(N)
    probs /= np.sum(probs)
    for i in range(T-1, -1, -1):
        probs = np.dot(transition_probs, (probs*observations[i]).T)
        probs /= np.sum(probs)
        backward_probs[i] = probs
    
    state_probs = forward_probs*backward_probs
    state_probs /= np.sum(state_probs, axis=1).reshape(-1, 1)
    return state_probs, forward_probs, backward_probs

def dataset_segments(data, **nslrargs):
    segments = ((nslr.fit_gaze(ts, xs, **nslrargs), outliers) for ts, xs, outliers in data)
    features = [list(segment_features(s.segments, o)) for s, o in segments]
    return features

def transition_estimates(obs, trans, forward, backward):
    T, N = len(obs), len(trans)
    ests = np.zeros((T, N, N))
    for start, end, i in itertools.product(range(N), range(N), range(T)):
        if i == T - 1:
            b = 1/N
        else:
            b = backward[i+1, end]
        ests[i,start,end] = forward[i,start]*b*trans[start,end]
    return ests

def reestimate_observations(sessions,
        transition_probs=GazeTransitionModel,
        observation_model=GazeObservationModel,
        initial_probs=None):
    
    all_observations = np.vstack(sessions)
    
    import matplotlib.pyplot as plt
    CLASS_COLORS = {
    1: 'b',
    2: 'r',
    3: 'y',
    4: 'g',
    5: 'm',
    6: 'c',
    22: 'orange'
    }

    #plt.plot(all_observations[:,0], all_observations[:,1], '.', alpha=0.1, color='black')
    for iteration in range(100):
        all_state_probs = []
        all_transition_probs = []
        for features in sessions:
            liks = np.array([observation_model.liks(f) for f in features])
            probs, forward, backward = forward_backward(transition_probs, liks, initial_probs)
            all_state_probs.extend(probs)
            all_transition_probs.append(transition_estimates(liks, transition_probs, forward, backward))
        all_state_probs = np.array(all_state_probs)
        all_transition_probs = np.vstack(all_transition_probs)
        winner = np.argmax(all_state_probs, axis=1)
        for cls in np.unique(winner):
            my = winner == cls
            plt.plot(all_observations[my,0], all_observations[my,1], '.', alpha=0.1, color=CLASS_COLORS[cls+1])
        dists = {}
        for i, cls in enumerate(observation_model.idxclass):
            w = all_state_probs[:,i]
            wsum = np.sum(w)
            w /= wsum
            #for end in range(len(transition_probs)):
            #    transition_probs[i, end] = np.sum(all_transition_probs[:,i,end])/wsum
            #valid = np.ones(len(all_observations), dtype=bool)
            mean = np.average(all_observations, weights=w, axis=0)
            cov = np.cov(all_observations, aweights=w, rowvar=False)
            #cov = np.diag(np.diag(cov))
            #cov[1,1] = observation_model.dists[i].cov[1,1]
            #cov = observation_model.dists[i].cov
            #mean[1] = observation_model.dists[i].mean[1]
            plt.plot(mean[0], mean[1], 'o', color=CLASS_COLORS[cls])
            dists[cls] = scipy.stats.multivariate_normal(mean, cov)
        plt.pause(0.1)
        plt.cla()
        transition_probs /= np.sum(transition_probs, axis=1).reshape(-1, 1)
        print(transition_probs)
        observation_model=ObservationModel(dists)
    return observation_model

from sklearn.covariance import MinCovDet
def reestimate_observations(sessions,
        transition_probs=GazeTransitionModel,
        observation_model=GazeObservationModel,
        initial_probs=None):
    
    all_observations = np.vstack(sessions)
    
    import matplotlib.pyplot as plt
    CLASS_COLORS = {
    1: 'b',
    2: 'r',
    3: 'y',
    4: 'g',
    5: 'm',
    6: 'c',
    22: 'orange'
    }
    
    N = len(transition_probs)
    if initial_probs is None:
        initial_probs = np.ones(N)
        initial_probs /= np.sum(initial_probs)
    #plt.plot(all_observations[:,0], all_observations[:,1], '.', alpha=0.1, color='black')
    for iteration in range(100):
        all_states = []
        all_transitions = np.zeros((N, N))
        for features in sessions:
            liks = np.array([observation_model.liks(f) for f in features])
            states = viterbi(initial_probs, transition_probs, liks)
            for i in range(len(states) - 1):
                all_transitions[states[i], states[i+1]] += 1
            all_states.extend(states)
        all_states = np.array(all_states)
        for cls in np.unique(states):
            my = all_states == cls
            plt.plot(all_observations[my,0], all_observations[my,1], '.', alpha=0.1, color=CLASS_COLORS[cls+1])
        dists = {}
        print("ITER", iteration)
        for i, cls in enumerate(observation_model.idxclass):
            #for end in range(len(transition_probs)):
            #    transition_probs[i, end] = all_transitions[i,end]
            #valid = np.ones(len(all_observations), dtype=bool)
            my = all_states == i
            mean = np.average(all_observations[my], axis=0)
            #cov = np.cov(all_observations[my], rowvar=False)
            #mean[1] = observation_model.dists[i].mean[1]
            cov = observation_model.dists[i].cov
            
            robust = MinCovDet().fit(all_observations[my])
            mean = robust.location_
            covar = robust.covariance_
            print(cls, ':', [mean.tolist(), (covar).tolist()])
            plt.plot(mean[0], mean[1], 'o', color=CLASS_COLORS[cls])
            dists[cls] = scipy.stats.multivariate_normal(mean, cov)
        plt.pause(0.1)
        plt.cla()
        transition_probs /= np.sum(transition_probs, axis=1).reshape(-1, 1)
        observation_model=ObservationModel(dists)
    return observation_model


def segment_features(segments, outliers=None):
    prev_direction = np.array([0.0, 0.0])
    if outliers is None:
        outliers = np.zeros(segments[-1].i[-1], dtype=bool)
    for segment in segments:
        if np.any(outliers[segment.i[0]:segment.i[1]]): continue
        duration = float(np.diff(segment.t))
        speed = np.diff(segment.x, axis=0)/duration
        velocity = float(np.linalg.norm(speed))
        direction = speed/velocity
        cosangle = float(np.dot(direction, prev_direction.T))
        
        # Fisher transform, avoid exact |1|
        cosangle *= (1 - 1e-6)
        cosangle = np.arctanh(cosangle)
        if cosangle != cosangle:
            cosangle = 0.0
        
        yield safelog(velocity), cosangle
        prev_direction = direction

def classify_segments(segments,
        observation_model=GazeObservationModel,
        transition_model=GazeTransitionModel,
        initial_probabilities=None):
    if initial_probabilities is None:
        initial_probabilities = np.ones(len(transition_model))
        initial_probabilities /= np.sum(initial_probabilities)
    observation_likelihoods = (observation_model.liks(f) for f in segment_features(segments))

    path = viterbi(initial_probabilities, transition_model, observation_likelihoods)
    return observation_model.idxclass[path]
    
def classify_gaze(ts, xs, **kwargs):
    segmentation = nslr.fit_gaze(ts, xs, **kwargs)
    seg_classes = classify_segments(segmentation.segments)
    sample_classes = np.zeros(len(ts))
    for c, s in zip(seg_classes, segmentation.segments):
        start = s.i[0]
        end = s.i[1]
        sample_classes[start:end] = c

    return sample_classes, segmentation, seg_classes
