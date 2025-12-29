import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple

# Partially-overlapping but separable line priors
def line_profile_base(line_id: str) -> Dict:
    if line_id == 'L1':
        return dict(r_load=5.2, l_load=9.5e-3, harm3=0.040, harm5=0.020, harm7=0.010, phase_base=+0.010, curr_scale=1.00, volt_scale=1.00)
    if line_id == 'L2':
        return dict(r_load=4.8, l_load=8.5e-3, harm3=0.025, harm5=0.045, harm7=0.015, phase_base=-0.008, curr_scale=1.03, volt_scale=0.99)
    if line_id == 'L3':
        return dict(r_load=5.6, l_load=10.5e-3, harm3=0.030, harm5=0.030, harm7=0.030, phase_base=+0.000, curr_scale=0.97, volt_scale=1.01)
    return dict(r_load=5.0, l_load=9e-3, harm3=0.03, harm5=0.03, harm7=0.02, phase_base=0.0, curr_scale=1.0, volt_scale=1.0)

FREQ_BASE = 50.0
J_FREQ = (-0.25, 0.25)
J_R = (0.92, 1.08)
J_L = (0.92, 1.08)
J_PHASE = (-0.02, 0.02)
J_H3 = (-0.01, 0.01)
J_H5 = (-0.01, 0.01)
J_H7 = (-0.008, 0.008)
J_I_SCALE = (0.95, 1.05)
J_V_SCALE = (0.99, 1.01)
J_NOISE_I = (0.02, 0.05)
J_NOISE_V = (0.15, 0.28)

def jitter(a, b): return np.random.uniform(a, b)

def rl_phase_lag(r: float, l: float, freq: float) -> float:
    w = 2*np.pi*freq
    return np.arctan2(w*l, r)

class LTLineSimulator:
    def __init__(self, sample_rate=200, duration=2.0, seed=None):
        self.fs = sample_rate
        self.duration = duration
        if seed is not None:
            np.random.seed(seed)

    def _three_phase(self, amp, base_freq, t, phase_shift=0.0, h3=0.0, h5=0.0, h7=0.0):
        sig = np.zeros((len(t),3))
        for i in range(3):
            phase = i*2*np.pi/3 + phase_shift
            base = amp*np.sin(2*np.pi*base_freq*t - phase)
            s3 = h3*amp*np.sin(2*np.pi*3*base_freq*t - 3*phase)
            s5 = h5*amp*np.sin(2*np.pi*5*base_freq*t - 5*phase)
            s7 = h7*amp*np.sin(2*np.pi*7*base_freq*t - 7*phase)
            sig[:,i] = base + s3 + s5 + s7
        return sig

    def _transient_burst(self, span: int, fs: int) -> np.ndarray:
        t = np.arange(span)/fs
        f1 = np.random.uniform(180, 300); f2 = np.random.uniform(500, 850)
        sweep = np.sin(2*np.pi*(f1 + (f2-f1)*t/t[-1])*t)
        carrier = np.sin(2*np.pi*np.random.uniform(220,650)*t)
        return (0.5*carrier + 0.5*sweep) * np.hanning(span)

    def generate_sequence(self, event_type='normal', baseline_current=20.0, baseline_voltage=230.0,
                          line_id='L1', segment_id='S1-F1') -> Tuple[pd.DataFrame, str, float, str, str]:
        fs = self.fs
        t = np.arange(0, self.duration, 1.0/fs)
        prof = line_profile_base(line_id)

        freq = FREQ_BASE + jitter(*J_FREQ)
        r = prof['r_load'] * jitter(*J_R)
        l = prof['l_load'] * jitter(*J_L)
        phase_base = prof['phase_base'] + jitter(*J_PHASE)
        h3 = max(0.0, prof['harm3'] + jitter(*J_H3))
        h5 = max(0.0, prof['harm5'] + jitter(*J_H5))
        h7 = max(0.0, prof['harm7'] + jitter(*J_H7))
        i_scale = prof['curr_scale'] * jitter(*J_I_SCALE)
        v_scale = prof['volt_scale'] * jitter(*J_V_SCALE)
        n_i = jitter(*J_NOISE_I)
        n_v = jitter(*J_NOISE_V)

        iv_lag = rl_phase_lag(r, l, freq)

        I = self._three_phase(baseline_current*i_scale, freq, t, phase_shift=phase_base, h3=h3, h5=h5, h7=h7)
        V = self._three_phase(baseline_voltage*v_scale, freq, t, phase_shift=phase_base - iv_lag, h3=h3/2, h5=h5/2, h7=h7/3)
        A = np.zeros(len(t))

        I += np.random.normal(0, n_i*baseline_current, I.shape)
        V += np.random.normal(0, n_v*0.01*baseline_voltage, V.shape)

        idx = int(np.random.uniform(0.2*self.duration, 0.75*self.duration)*fs)
        idx = max(5, min(idx, len(t)-20))

        if event_type == 'switch_off':
            decay = int(np.random.uniform(0.10, 0.22)*fs)
            end = min(len(t), idx+decay)
            for ph in range(3):
                I[idx:end,ph] = np.linspace(I[idx,ph], 0.01, end-idx)
                I[end:,ph] = 0.01

        elif event_type == 'line_break':
            for ph in range(3):
                I[idx:,ph] = 0.01
            spike_len = min(len(t)-idx, int(np.random.uniform(0.03,0.06)*fs))
            V[idx:idx+spike_len,:] += np.linspace(np.random.uniform(0.06,0.12)*baseline_voltage, 0, spike_len)[:,None]
            vib_len = max(1, min(int(np.random.uniform(8,16)), len(t)-idx))
            A[idx:idx+vib_len] += np.linspace(np.random.uniform(1.6,2.6), 0.0, vib_len)

        elif event_type == 'transient':
            n_bursts = np.random.choice([1,2,3], p=[0.55,0.35,0.10])
            start = idx
            for _ in range(n_bursts):
                span = max(5, min(int(np.random.uniform(0.02,0.08)*fs), len(t)-start-1))
                burst = self._transient_burst(span, fs)
                V[start:start+span,:] += 0.09*baseline_voltage*burst[:,None]
                I[start:start+span,:] += 0.05*baseline_current*np.random.randn(span,3)
                start = min(len(t)-1, start + span + int(np.random.uniform(5,20)))

        elif event_type == 'normal':
            if np.random.rand() < 0.15:
                span = max(5, min(int(np.random.uniform(0.015,0.04)*fs), len(t)-idx-1))
                V[idx:idx+span,:] += np.random.normal(0, 0.006*baseline_voltage, (span,3))
                I[idx:idx+span,:] += np.random.normal(0, 0.015*baseline_current, (span,3))

        df = pd.DataFrame({
            'time': t,
            'Ia': I[:,0], 'Ib': I[:,1], 'Ic': I[:,2],
            'Va': V[:,0], 'Vb': V[:,1], 'Vc': V[:,2],
            'accel': A
        })
        return df, event_type, t[idx], line_id, segment_id

def build_dataset(outdir='dataset_raw', n_sequences=6000, seed=42):
    os.makedirs(outdir, exist_ok=True)
    sim = LTLineSimulator(sample_rate=200, duration=2.0, seed=seed)
    meta = []
    event_types = ['normal','switch_off','line_break','transient']
    line_ids = ['L1','L2','L3']
    segment_ids = {'L1':['S1-F1','F1-C1'], 'L2':['S2-F3','F3-C6'], 'L3':['X1-Y1','Y1-Z1']}
    probs = [0.25,0.20,0.25,0.30]
    for i in range(n_sequences):
        et = np.random.choice(event_types, p=probs)
        line = np.random.choice(line_ids)
        segment = np.random.choice(segment_ids[line])
        df, etype, etime, lineid, segid = sim.generate_sequence(event_type=et, line_id=line, segment_id=segment)
        fname = f"seq_{i:06d}.csv"
        df.to_csv(os.path.join(outdir, fname), index=False)
        meta.append({'seq_id':i,'file':fname,'event_type':etype,'event_time':etime,'line_id':lineid,'segment_id':segid})
    pd.DataFrame(meta).to_csv(os.path.join(outdir,'metadata.csv'), index=False)
    print(f"[OK] Generated {n_sequences} sequences into {outdir}")

if __name__ == "__main__":
    build_dataset(outdir='dataset_raw',     n_sequences=6000, seed=42)
    build_dataset(outdir='dataset_shifted', n_sequences=1500, seed=777)
