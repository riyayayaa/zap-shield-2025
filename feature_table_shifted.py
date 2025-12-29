  import os
  import numpy as np
  import pandas as pd
  import pywt
  from scipy.stats import skew, kurtosis
  from scipy.signal import hilbert

  def _as_float(x):
      arr = np.asarray(x)
      if arr.size == 0: return 0.0
      if np.isscalar(x): return float(x)
      return float(np.mean(arr))

  def fft_parts(x, fs):
      X = np.fft.rfft(x)
      freqs = np.fft.rfftfreq(len(x), 1/fs)
      mag = np.abs(X)
      mag2 = mag**2
      return freqs, mag, mag2

  def fft_band_energy_fraction(mag2, freqs, f0, f1):
      mask = (freqs >= f0) & (freqs < f1)
      total = mag2.sum() + 1e-12
      return _as_float(mag2[mask].sum()/total)

  def spectral_centroid_bandwidth(mag, freqs):
      p = mag + 1e-12
      sc = (freqs*p).sum()/p.sum()
      bw = np.sqrt(((freqs - sc)**2 * p).sum()/p.sum())
      return _as_float(sc), _as_float(bw)

  def thd_proxy(mag2, freqs, f0=50.0):
      def band(f_center, width=3.0):
          return (freqs >= f_center - width) & (freqs <= f_center + width)
      total = mag2.sum() + 1e-12
      fund = mag2[band(f0)].sum() + 1e-12
      h3 = mag2[band(3*f0)].sum()
      h5 = mag2[band(5*f0)].sum()
      h7 = mag2[band(7*f0)].sum()
      thd = (h3 + h5 + h7)/(fund + 1e-12)
      return _as_float(thd), _as_float(fund/total)

  def hilbert_env_stats(x):
      env = np.abs(hilbert(x))
      return _as_float(env.mean()), _as_float(env.std()), _as_float(env.max())

  def subwindow_phase_lag(I, V, fs, win=80, step=40):
      if len(I) < win: return 0.0, 0.0
      lags = []
      for i in range(0, len(I)-win+1, step):
          iwin = I[i:i+win]; vwin = V[i:i+win]
          Xi = np.fft.rfft(iwin); Xv = np.fft.rfft(vwin)
          if len(Xi) < 2 or len(Xv) < 2: continue
          lag = np.angle(Xi[1]) - np.angle(Xv[1])
          lags.append(np.arctan2(np.sin(lag), np.cos(lag)))
      if len(lags) == 0: return 0.0, 0.0
      lags = np.asarray(lags)
      return _as_float(lags.mean()), _as_float(lags.std())

  def extract_features_row(df, fs=200):
      feats = {}
      Ia, Ib, Ic = df['Ia'].values, df['Ib'].values, df['Ic'].values
      Va, Vb, Vc = df['Va'].values, df['Vb'].values, df['Vc'].values
      acc = df['accel'].values

      for name, arr in [('Ia',Ia),('Ib',Ib),('Ic',Ic)]:
          feats[f'{name}_mean'] = _as_float(arr.mean())
          feats[f'{name}_std']  = _as_float(arr.std())
          feats[f'{name}_skew'] = _as_float(skew(arr))
          feats[f'{name}_kurt'] = _as_float(kurtosis(arr))
          w = pywt.wavedec(arr, 'db4', level=3)
          for i,c in enumerate(w):
              feats[f'{name}_wmean_{i}'] = _as_float(np.mean(c))
              feats[f'{name}_wstd_{i}']  = _as_float(np.std(c))
          freqs, mag, mag2 = fft_parts(arr, fs)
          feats[f'{name}_b1'] = fft_band_energy_fraction(mag2, freqs, 0, 20)
          feats[f'{name}_b2'] = fft_band_energy_fraction(mag2, freqs, 20, 80)
          feats[f'{name}_b3'] = fft_band_energy_fraction(mag2, freqs, 80, 300)
          feats[f'{name}_b4'] = fft_band_energy_fraction(mag2, freqs, 300, 1000)
          sc, bw = spectral_centroid_bandwidth(mag, freqs)
          feats[f'{name}_sc'] = sc; feats[f'{name}_bw'] = bw
          thd, fund_frac = thd_proxy(mag2, freqs, f0=50.0)
          feats[f'{name}_thd'] = thd; feats[f'{name}_fund_frac'] = fund_frac
          env_m, env_s, env_x = hilbert_env_stats(arr)
          feats[f'{name}_env_mean'] = env_m; feats[f'{name}_env_std'] = env_s; feats[f'{name}_env_max'] = env_x

      I_means = np.array([Ia.mean(), Ib.mean(), Ic.mean()])
      I_stds  = np.array([Ia.std(),  Ib.std(),  Ic.std()])
      feats['I_ratio_a_b'] = _as_float((I_means[0]+1e-6)/(I_means[1]+1e-6))
      feats['I_ratio_b_c'] = _as_float((I_means[1]+1e-6)/(I_means[2]+1e-6))
      feats['I_ratio_c_a'] = _as_float((I_means[2]+1e-6)/(I_means[0]+1e-6))
      feats['I_std_ratio_a_b'] = _as_float((I_stds[0]+1e-6)/(I_stds[1]+1e-6))
      feats['I_std_ratio_b_c'] = _as_float((I_stds[1]+1e-6)/(I_stds[2]+1e-6))
      feats['I_std_ratio_c_a'] = _as_float((I_stds[2]+1e-6)/(I_stds[0]+1e-6))

      for phn, Iph, Vph in [('a',Ia,Va),('b',Ib,Vb),('c',Ic,Vc)]:
          m,s = subwindow_phase_lag(Iph, Vph, fs, win=80, step=40)
          feats[f'lag_{phn}_mean'] = m
          feats[f'lag_{phn}_std']  = s

      for name, arr in [('Va',Va),('Vb',Vb),('Vc',Vc)]:
          feats[f'{name}_mean'] = _as_float(arr.mean())
          feats[f'{name}_std']  = _as_float(arr.std())

      feats['corr_IaIb'] = _as_float(np.corrcoef(Ia,Ib)[0,1])
      feats['corr_IbIc'] = _as_float(np.corrcoef(Ib,Ic)[0,1])
      feats['corr_IaIc'] = _as_float(np.corrcoef(Ia,Ic)[0,1])

      a = np.exp(1j*2*np.pi/3)
      I0 = (Ia+Ib+Ic)/3
      I1 = (Ia + a*Ib + a*a*Ic)/3
      I2 = (Ia + a*a*Ib + a*Ic)/3
      feats['seq_I0_mag']   = _as_float(np.abs(I0))
      feats['seq_I1_mag']   = _as_float(np.abs(I1))
      feats['seq_I2_mag']   = _as_float(np.abs(I2))
      feats['seq_unbalance']= _as_float((np.abs(I2)+np.abs(I0))/(np.abs(I1)+1e-9))

      feats['accel_max'] = _as_float(acc.max())
      feats['accel_std'] = _as_float(acc.std())
      return feats

  def build_features_table(raw_dir='dataset_raw', out_csv='features_table_advanced.csv', fs=200):
      meta = pd.read_csv(os.path.join(raw_dir,'metadata.csv'))
      rows = []
      for _, r in meta.iterrows():
          df = pd.read_csv(os.path.join(raw_dir, r['file']))
          feats = extract_features_row(df, fs=fs)
          feats['seq_id'] = int(r['seq_id'])
          feats['fault_type'] = r['event_type']
          feats['line_id'] = r['line_id']
          feats['segment_id'] = r['segment_id']
          rows.append(feats)
      feat_df = pd.DataFrame(rows)
      for col in feat_df.columns:
          try:
              feat_df[col] = pd.to_numeric(feat_df[col])
          except Exception:
              pass
      feat_df.to_csv(out_csv, index=False)
      print(f"[OK] Wrote {out_csv} with {feat_df.shape[0]} rows and {feat_df.shape[1]} columns")
      return feat_df

  if __name__ == "__main__":
      build_features_table(raw_dir='dataset_raw', out_csv='features_table_advanced.csv', fs=200)
      build_features_table(raw_dir='dataset_shifted', out_csv='features_table_shifted.csv', fs=200)
