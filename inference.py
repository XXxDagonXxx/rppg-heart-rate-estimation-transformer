"""
rPPG Transformer v5 — Inference / Testing Script
Tests the trained model (best_rppg_v5.pth) on any video file.
All architecture, preprocessing, and signal-processing components
are preserved exactly as in the training notebook.
Usage:
python test_rppg_v5.py --video path/to/video.mp4
python test_rppg_v5.py --video path/to/video.mp4 --model best_rppg_v5.pth --plot
"""
import os, sys, cv2, math, random, time, hashlib, pickle, warnings, argparse
import numpy  as np
import torch
import torch.nn            as nn
import torch.nn.functional as F
import matplotlib.pyplot   as plt
import mediapipe           as mp
from pathlib     import Path
from scipy.signal import butter, filtfilt, detrend as sp_detrend
from scipy.signal import resample as scipy_resample
from scipy.stats  import pearsonr
warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  (must exactly match training)
# ═══════════════════════════════════════════════════════════════════════════════
class CFG:
    DEVICE     : str   = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE   : int   = 64
    SEQ_LEN    : int   = 256
    STRIDE     : int   = 64       # window hop for sliding-window inference
    D_MODEL    : int   = 128
    NHEAD      : int   = 8
    NUM_LAYERS : int   = 4
    DIM_FF     : int   = 512
    DROPOUT    : float = 0.1
    FPS        : float = 30.0
    MIN_FRAMES : int   = 16       # [UPDATED] Lowered to support short videos (padded to SEQ_LEN)
    GRAD_CKPT  : bool  = False    # disabled for inference
    USE_AMP    : bool  = torch.cuda.is_available()

cfg = CFG()

BPM_LOW,  BPM_HIGH  = 40.0,  200.0
FREQ_LOW, FREQ_HIGH = BPM_LOW / 60.0, BPM_HIGH / 60.0

# MediaPipe landmark index groups (identical to training)
FOREHEAD_IDXS = [10, 67, 103, 109, 338, 297, 332, 333, 334, 296]
LEFT_CHEEK    = [50, 187, 205, 207, 216, 206, 203]
RIGHT_CHEEK   = [280, 425, 411, 427, 436, 426, 423]

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING  (exact copy from training notebook)
# ═══════════════════════════════════════════════════════════════════════════════
def detrend_signal(signal: np.ndarray, n_segments: int = 8) -> np.ndarray:
    """Fast piecewise-linear detrending (O(N))."""
    N   = len(signal)
    out = signal.astype(np.float64)
    seg_len = max(N // n_segments, 1)
    result  = np.zeros_like(out)
    for i in range(0, N, seg_len):
        seg = out[i : i + seg_len]
        result[i : i + seg_len] = sp_detrend(seg, type="linear")
    result = sp_detrend(result, type="linear")
    return result.astype(np.float32)

def bandpass_filter(
    signal: np.ndarray,
    fs:     float = 30.0,
    lo:     float = FREQ_LOW,
    hi:     float = FREQ_HIGH,
    order:  int   = 4,
) -> np.ndarray:
    """Butterworth bandpass with automatic order reduction for short signals."""
    nyq = fs * 0.5
    lo, hi = max(lo, 0.01), min(hi, nyq * 0.99)
    if lo >= hi:
        return (signal - signal.mean()) / (signal.std() + 1e-8)
    
    min_len = 3 * (2 * order + 1)
    effective_order = order
    while len(signal) < min_len and effective_order > 1:
        effective_order -= 1
        min_len = 3 * (2 * effective_order + 1)

    if len(signal) < 15:
        sig = signal - signal.mean()
        return (sig / (sig.std() + 1e-8)).astype(np.float32)

    b, a  = butter(effective_order, [lo / nyq, hi / nyq], btype='band')
    filt  = filtfilt(b, a, signal)
    return ((filt - filt.mean()) / (filt.std() + 1e-8)).astype(np.float32)

def compute_bpm(signal: np.ndarray, fs: float = 30.0) -> float:
    signal = bandpass_filter(signal, fs)
    N = len(signal)
    
    # [FIX 2] Hybrid BPM: Use time-domain peak detection for short videos (< 4s)
    # FFT frequency resolution is too poor for short windows, causing massive errors.
    if N < fs * 4.0:
        from scipy.signal import find_peaks
        # Minimum distance between peaks = 0.3s (equivalent to max 200 BPM)
        min_distance = int(0.3 * fs)
        peaks, _ = find_peaks(signal, distance=min_distance)
        
        if len(peaks) >= 2:
            avg_interval_sec = np.mean(np.diff(peaks)) / fs
            estimated_bpm = 60.0 / avg_interval_sec
            if 40.0 <= estimated_bpm <= 200.0:
                return float(estimated_bpm)

    # Fallback to FFT for longer signals or if peak detection fails
    freqs   = np.fft.rfftfreq(N, d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(signal * np.hanning(N)))
    mask    = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
    if mask.sum() == 0:
        return 75.0
        
    fft_band   = fft_mag[mask]
    freqs_band = freqs[mask]
    peak_idx   = np.argmax(fft_band)

    if 0 < peak_idx < len(fft_band) - 1:
        alpha = fft_band[peak_idx - 1]
        beta  = fft_band[peak_idx]
        gamma = fft_band[peak_idx + 1]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-10:
            offset = 0.5 * (alpha - gamma) / denom
            bin_hz = (freqs_band[1] - freqs_band[0]) if len(freqs_band) > 1 else 0.0
            return float((freqs_band[peak_idx] + offset * bin_hz) * 60.0)

    return float(freqs_band[peak_idx] * 60.0)

def compute_snr(signal: np.ndarray, fs: float = 30.0) -> float:
    """SNR in physiological frequency band (dB)."""
    freqs   = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    fft_pow = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    band    = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
    p_sig   = fft_pow[band].sum()  + 1e-12
    p_noise = fft_pow[~band].sum() + 1e-12
    return float(10 * np.log10(p_sig / p_noise))

# ═══════════════════════════════════════════════════════════════════════════════
# ROI EXTRACTION  (exact copy from training notebook)
# ═══════════════════════════════════════════════════════════════════════════════
class LandmarkStabiliser:
    """EMA over landmark positions to reduce jitter."""
    def __init__(self, alpha: float = 0.6):
        self.alpha    = alpha
        self.prev_pts: dict = {}

    def smooth(self, idx: int, pts: np.ndarray) -> np.ndarray:
        if idx not in self.prev_pts:
            self.prev_pts[idx] = pts.astype(np.float32)
            return pts
        smoothed = self.alpha * pts + (1 - self.alpha) * self.prev_pts[idx]
        self.prev_pts[idx] = smoothed
        return smoothed.astype(np.int32)

def skin_mask_ycrcb(roi_bgr: np.ndarray) -> np.ndarray:
    """Refined YCrCb skin mask (Kovac et al.)."""
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    mask  = cv2.inRange(
        ycrcb,
        np.array([  0, 130,  75], dtype=np.uint8),
        np.array([255, 180, 135], dtype=np.uint8),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask

# [SPD-8] CLAHE instance pre-allocated once (not per-frame)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

def clahe_normalise(roi_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on L channel."""
    lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def roi_quality_score(roi_rgb: np.ndarray, skin_mask: np.ndarray) -> float:
    """[0, 1] quality score combining skin coverage and sharpness."""
    coverage  = (skin_mask > 0).mean()
    gray      = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    lap_var   = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(lap_var / 500.0, 1.0)
    return float(0.5 * coverage + 0.5 * sharpness)

def extract_roi(
    frame_rgb:       np.ndarray,
    landmarks,
    indices:         list,
    stabiliser:      LandmarkStabiliser = None,
    roi_idx:         int  = 0,
    apply_skin_mask: bool = True,
    apply_clahe:     bool = True,
    img_size:        int  = 64,
) -> tuple:
    """Extract face ROI from MediaPipe landmarks."""
    h, w, _ = frame_rgb.shape
    pts = np.array(
        [(int(landmarks.landmark[i].x * w),
          int(landmarks.landmark[i].y * h)) for i in indices],
        dtype=np.int32,
    )
    if stabiliser is not None:
        pts = stabiliser.smooth(roi_idx, pts)

    hull     = cv2.convexHull(pts)
    geo_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(geo_mask, hull, 255)

    x, y, bw, bh = cv2.boundingRect(hull)
    pad = max(3, min(bw, bh) // 8)
    x   = max(0, x - pad);  y = max(0, y - pad)
    bw  = min(w - x, bw + 2 * pad)
    bh  = min(h - y, bh + 2 * pad)

    if bw < 5 or bh < 5:
        return None, None, 0.0

    roi_rgb   = frame_rgb[y:y + bh, x:x + bw].copy()
    mask_crop = geo_mask[y:y + bh, x:x + bw]

    if apply_clahe:
        roi_rgb = clahe_normalise(roi_rgb)

    if apply_skin_mask:
        roi_bgr  = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
        skin_m   = skin_mask_ycrcb(roi_bgr)
        combined = cv2.bitwise_and(mask_crop, skin_m)
    else:
        combined = mask_crop
        skin_m   = mask_crop

    skin_pixels = roi_rgb[combined > 0]
    if len(skin_pixels) < 10:
        rgb_mean = roi_rgb.reshape(-1, 3).mean(axis=0).astype(np.float32)
        quality  = 0.1
    else:
        rgb_mean = skin_pixels.mean(axis=0).astype(np.float32)
        quality  = roi_quality_score(roi_rgb, skin_m)

    roi_resized = cv2.resize(roi_rgb, (img_size, img_size)).astype(np.float32) / 255.0
    return roi_resized, rgb_mean, quality

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE  (exact copy from training notebook)
# ═══════════════════════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al. 2018)."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * w

class SpatialAttention(nn.Module):
    """CBAM spatial attention."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1,  keepdim=True)[0]
        return x * self.sig(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAMBlock(nn.Module):
    """Combined channel + spatial CBAM attention."""
    def __init__(self, channels: int):
        super().__init__()
        self.channel = SEBlock(channels)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))

def depthwise_sep_conv(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Depthwise separable convolution."""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class AttentionPool2d(nn.Module):
    """Learned soft-attention spatial pooling (B, C, H, W) → (B, C)."""
    def __init__(self, channels: int):
        super().__init__()
        self.key = nn.Conv2d(channels, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.key(x).flatten(2), dim=-1)
        return (x.flatten(2) * w).sum(dim=-1)

class SpectralAttention1D(nn.Module):
    """1-D spectral attention gate over temporal features."""
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Xf   = torch.fft.rfft(x, dim=1)
        mag  = torch.abs(Xf).mean(dim=1)
        gate = self.fc(mag).unsqueeze(1)
        return x * gate

class MultiScaleSpatialEncoder(nn.Module):
    """Multi-scale spatial CNN encoder with CBAM attention."""
    def __init__(self, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        def conv_bn(ci, co, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(ci, co, k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            )

        self.stem   = nn.Sequential(conv_bn(3, 32), conv_bn(32, 32), nn.MaxPool2d(2))
        self.scale1 = nn.Sequential(depthwise_sep_conv(32, 32))
        self.cbam1  = CBAMBlock(32)
        self.pool1  = AttentionPool2d(32)
        self.scale2 = nn.Sequential(depthwise_sep_conv(32, 64), depthwise_sep_conv(64, 64), nn.MaxPool2d(2))
        self.cbam2  = CBAMBlock(64)
        self.pool2  = AttentionPool2d(64)
        self.scale3 = nn.Sequential(depthwise_sep_conv(64, 128), depthwise_sep_conv(128, 128), nn.MaxPool2d(2))
        self.cbam3  = CBAMBlock(128)
        self.pool3  = AttentionPool2d(128)
        self.proj   = nn.Sequential(
            nn.Linear(32 + 64 + 128, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s  = self.stem(x)
        f1 = self.pool1(self.cbam1(self.scale1(s)))
        s2 = self.scale2(s)
        f2 = self.pool2(self.cbam2(s2))
        f3 = self.pool3(self.cbam3(self.scale3(s2)))
        return self.proj(torch.cat([f1, f2, f3], dim=-1))

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        
        # [SPD-7] persistent=False: not saved/loaded in state_dict
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])

class RPPGTransformer(nn.Module):
    """Dual-Stream rPPG Transformer v5."""
    def __init__(self):
        super().__init__()
        self.motion_enc     = MultiScaleSpatialEncoder(cfg.D_MODEL, cfg.DROPOUT)
        self.appearance_enc = MultiScaleSpatialEncoder(cfg.D_MODEL, cfg.DROPOUT)

        self.stream_fuse = nn.Sequential(
            nn.Linear(cfg.D_MODEL * 2, cfg.D_MODEL),
            nn.LayerNorm(cfg.D_MODEL),
            nn.GELU(),
        )

        self.pos_enc = SinusoidalPositionalEncoding(
             cfg.D_MODEL, max_len=cfg.SEQ_LEN + 16, dropout=cfg.DROPOUT
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.D_MODEL,
            nhead=cfg.NHEAD,
            dim_feedforward=cfg.DIM_FF,
            dropout=cfg.DROPOUT,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.NUM_LAYERS,
            enable_nested_tensor=False,
        )

        self.spectral_attn = SpectralAttention1D(cfg.D_MODEL)

        # [ARC-1] Pre-norm residual temporal conv
        self.temporal_norm = nn.LayerNorm(cfg.D_MODEL)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(cfg.D_MODEL, cfg.D_MODEL, 5, padding=2, bias=False, groups=cfg.D_MODEL),
            nn.Conv1d(cfg.D_MODEL, cfg.D_MODEL, 1, bias=False),
            nn.BatchNorm1d(cfg.D_MODEL),
            nn.GELU(),
            nn.Conv1d(cfg.D_MODEL, cfg.D_MODEL, 3, padding=1, bias=False),
        )

        # [ARC-2] Dropout before final projection
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.D_MODEL),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.D_MODEL, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _encode_stream(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Batched CNN: (B, T, H, W, C) → (B, T, D)."""
        B, T, H, W, C = x.shape
        xt   = x.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
        feats = encoder(xt)
        return feats.view(B, T, -1)

    def forward(
        self,
        diff_frames:          torch.Tensor,
        raw_frames:           torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        m = self._encode_stream(self.motion_enc,     diff_frames)
        a = self._encode_stream(self.appearance_enc, raw_frames)

        x = self.stream_fuse(torch.cat([m, a], dim=-1))
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.spectral_attn(x)

        # [ARC-1] Pre-norm residual temporal conv
        xn = self.temporal_norm(x)
        xc = self.temporal_conv(xn.permute(0, 2, 1)).permute(0, 2, 1)
        x  = x + xc

        return self.head(x).squeeze(-1)

# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO PROCESSOR  (exact copy from training notebook, inference-only)
# ═══════════════════════════════════════════════════════════════════════════════
def get_face_mesh():
    """Create a MediaPipe FaceMesh instance for inference."""
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

def process_video(video_path: str) -> tuple:
    """Extract dual-stream face frames from a video file."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or cfg.FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video FPS : {actual_fps:.2f}  |  Total frames reported: {total_frames}")

    face_mesh  = get_face_mesh()
    stabiliser = LandmarkStabiliser(alpha=0.65)

    raw_frames_list, rgb_signals_list, quality_list = [], [], []
    frame_num = 0

    print("  Extracting ROIs... ", end="", flush=True)
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"{frame_num} ", end="", flush=True)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results   = face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0]
        roi_imgs, rgb_means, qualities = [], [], []

        for ridx, indices in enumerate([FOREHEAD_IDXS, LEFT_CHEEK, RIGHT_CHEEK]):
            roi_img, rgb_mean, quality = extract_roi(
                frame_rgb, landmarks, indices,
                stabiliser=stabiliser, roi_idx=ridx,
                img_size=cfg.IMG_SIZE,
            )
            if roi_img is not None:
                roi_imgs.append(roi_img)
                rgb_means.append(rgb_mean)
                qualities.append(quality)

        if not roi_imgs:
            continue

        face_img = np.mean(np.stack(roi_imgs),  axis=0).astype(np.float32)
        rgb_mean = np.mean(np.stack(rgb_means), axis=0).astype(np.float32)
        qual     = float(np.mean(qualities))

        raw_frames_list.append(face_img)
        rgb_signals_list.append(rgb_mean)
        quality_list.append(qual)

    cap.release()
    face_mesh.close()
    print()

    n = len(raw_frames_list)
    if n < cfg.MIN_FRAMES:
        raise ValueError(
            f"Only {n} valid frames detected (minimum {cfg.MIN_FRAMES}). "
            f"Check that the video contains a clear frontal face."
        )
        
    # [NEW] Warning for very short videos regarding FFT resolution limits
    if n < 90:  # Less than ~3 seconds at 30fps
        print(f"  ⚠️  Warning: Short video detected ({n} frames, ~{n/actual_fps:.1f}s). "
              f"BPM accuracy may be limited due to FFT frequency resolution constraints.")

    print(f"  Valid frames extracted: {n}")

    raw_arr  = np.array(raw_frames_list,  dtype=np.float32)
    rgb_arr  = np.array(rgb_signals_list, dtype=np.float32)
    qual_arr = np.array(quality_list,     dtype=np.float32)

    # Temporal difference stream (motion cues) — normalised per-video
    diff_arr = raw_arr[1:] - raw_arr[:-1]
    mean_d   = diff_arr.mean(axis=(0, 1, 2), keepdims=True)
    std_d    = diff_arr.std(axis=(0, 1, 2),  keepdims=True)
    diff_arr = (diff_arr - mean_d) / (std_d + 1e-8)

    raw_arr  = raw_arr[:-1]
    rgb_arr  = rgb_arr[:-1]
    qual_arr = qual_arr[:-1]

    trim_frames = 2
    if len(raw_arr) > 2 * trim_frames + 5:
        raw_arr  = raw_arr[trim_frames:-trim_frames]
        diff_arr = diff_arr[trim_frames:-trim_frames]
        rgb_arr  = rgb_arr[trim_frames:-trim_frames]
        qual_arr = qual_arr[trim_frames:-trim_frames]
        print(f"  Trimmed {trim_frames} frames from start/end to remove motion artifacts.")

    return diff_arr, raw_arr, rgb_arr, qual_arr, actual_fps

# ═══════════════════════════════════════════════════════════════════════════════
# [UPDATED] PHYSICALLY-CONSISTENT PADDING FOR rPPG
# ═══════════════════════════════════════════════════════════════════════════════
def pad_rppg_sequence(raw_slice: np.ndarray, diff_slice: np.ndarray, target_len: int) -> tuple:
    """
    Pads short rPPG sequences using a physically-consistent strategy optimal for rPPG:
    - raw_frames: 'edge' padding (repeats the last frame). Keeps the CNN in-distribution 
      and avoids artificial spatial edges.
    - diff_frames: 'zero' padding (no motion). Physically consistent with static raw frames.
    
    If the sequence is longer than target_len, it is cropped.
    """
    n = raw_slice.shape[0]
    if n >= target_len:
        return raw_slice[:target_len], diff_slice[:target_len]
    
    pad_time = target_len - n
    
    # Pad raw frames by repeating the last frame (edge mode)
    pw_raw = [(0, pad_time)] + [(0, 0)] * (raw_slice.ndim - 1)
    raw_padded = np.pad(raw_slice, pw_raw, mode='edge')
    
    # Pad diff frames with zeros (no motion)
    pw_diff = [(0, pad_time)] + [(0, 0)] * (diff_slice.ndim - 1)
    diff_padded = np.pad(diff_slice, pw_diff, mode='constant', constant_values=0.0)
    
    return raw_padded, diff_padded

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_model(checkpoint_path: str) -> nn.Module:
    # ... [previous code] ...
    ckpt = torch.load(checkpoint_path, map_location=cfg.DEVICE)

    if isinstance(ckpt, dict):
        # [FIXED] Added checks for 'ema_state' and 'model_state'
        if 'ema_state' in ckpt:
            state_dict = ckpt['ema_state']
            print("  Using EMA weights (ema_state)")
        elif 'model_state' in ckpt:
            state_dict = ckpt['model_state']
            print("  Using model_state")
        elif 'ema_state_dict' in ckpt:
            state_dict = ckpt['ema_state_dict']
            print("  Using EMA weights (ema_state_dict)")
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
            print("  Using model_state_dict")
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            print("  Using state_dict")
        else:
            state_dict = ckpt
            print("  Using raw dict as state_dict")

    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace('_orig_mod.', '')
        cleaned[new_k] = v

    model = RPPGTransformer().to(cfg.DEVICE)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    if missing:
        print(f"  ⚠️  Missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  ⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model loaded  |  Params: {total_params:,} ({total_params/1e6:.2f}M)  |  Device: {cfg.DEVICE.upper()}")
    return model

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDING-WINDOW INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
def pad_or_crop(arr: np.ndarray, length: int, axis: int = 0) -> np.ndarray:
    """Trim or zero-pad array to `length` along `axis`."""
    n = arr.shape[axis]
    if n >= length:
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(0, length)
        return arr[tuple(slices)]
    
    pw = [(0, 0)] * arr.ndim
    pw[axis] = (0, length - n)
    return np.pad(arr, pw, mode='constant', constant_values=0.0)

def run_inference(
    model:       nn.Module,
    diff_frames: np.ndarray,
    raw_frames:  np.ndarray,
    seq_len:     int = 256,
    stride:      int = 64,
    batch_size:  int = 4,
) -> np.ndarray:
    N = len(diff_frames)
    
    # [FIX 1] Dynamically adjust window size for short videos to avoid padding
    # If the video is 60 frames, we process 60 frames, not 256.
    current_seq_len = min(N, seq_len)
    
    full_pred = np.zeros(N, dtype=np.float64)
    weight    = np.zeros(N, dtype=np.float64)

    # Window weights for smooth blending at window boundaries
    win_weights = np.hanning(current_seq_len).astype(np.float64)

    starts = list(range(0, max(1, N - current_seq_len + 1), stride))
    if not starts or starts[-1] + current_seq_len > N:
        starts.append(max(0, N - current_seq_len))
    starts = sorted(set(starts))

    print(f"  Sliding-window inference: {len(starts)} windows "
          f"(window_size={current_seq_len}, stride={stride})")

    # Collect windows into batches
    diff_batch, raw_batch, mask_batch, pos_batch = [], [], [], []

    def _run_batch():
        if not diff_batch:
            return
        diff_t = torch.tensor(np.stack(diff_batch), dtype=torch.float32).to(cfg.DEVICE)
        raw_t  = torch.tensor(np.stack(raw_batch),  dtype=torch.float32).to(cfg.DEVICE)
        mask_t = torch.tensor(np.stack(mask_batch), dtype=torch.float32).to(cfg.DEVICE)
        pad_mask = (mask_t == 0)

        with torch.no_grad():
            if cfg.USE_AMP:
                with torch.amp.autocast(device_type='cuda'):
                    preds = model(diff_t, raw_t, src_key_padding_mask=pad_mask)
            else:
                preds = model(diff_t, raw_t, src_key_padding_mask=pad_mask)

        preds_np = preds.cpu().numpy()

        for i, (start, pred_w) in enumerate(zip(pos_batch, preds_np)):
            end   = start + current_seq_len
            # Overlap-add with hanning window weights
            full_pred[start:end] += pred_w * win_weights
            weight[start:end]    += win_weights

        diff_batch.clear(); raw_batch.clear()
        mask_batch.clear(); pos_batch.clear()

    for start in starts:
        end        = min(start + current_seq_len, N)
        diff_slice = diff_frames[start:end].copy()
        raw_slice  = raw_frames[start:end].copy()
        T_valid    = len(diff_slice)

        # Pad or crop to current_seq_len (This is a no-op if N < seq_len)
        diff_out = pad_or_crop(diff_slice, current_seq_len)
        raw_out  = pad_or_crop(raw_slice,  current_seq_len)
        
        mask     = np.zeros(current_seq_len, dtype=np.float32)
        mask[:min(T_valid, current_seq_len)] = 1.0

        diff_batch.append(diff_out)
        raw_batch.append(raw_out)
        mask_batch.append(mask)
        pos_batch.append(start)

        if len(diff_batch) >= batch_size:
            _run_batch()

    _run_batch()  # flush remaining

    # Normalise by accumulated window weights
    safe_weight = np.where(weight > 1e-10, weight, 1.0)
    full_signal = (full_pred / safe_weight).astype(np.float32)

    return full_signal

# ═══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING & METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def postprocess_signal(raw_pred: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """Post-process the raw model output."""
    detrended = detrend_signal(raw_pred)
    filtered  = bandpass_filter(detrended, fs=fs)
    return filtered

def rolling_bpm(signal: np.ndarray, fs: float = 30.0, window_sec: float = 10.0) -> tuple:
    """Compute BPM over a sliding window for temporal HR tracking."""
    window = int(window_sec * fs)
    hop    = max(1, window // 4)
    times, bpms = [], []

    for start in range(0, max(1, len(signal) - window + 1), hop):
        seg = signal[start : start + window]
        if len(seg) < 30:
            continue
        times.append((start + window / 2) / fs)
        bpms.append(compute_bpm(seg, fs))

    return np.array(times), np.array(bpms)

def summarise_results(signal: np.ndarray, fs: float, video_path: str):
    """Print a concise summary of inference results."""
    bpm  = compute_bpm(signal, fs)
    snr  = compute_snr(signal, fs)
    dur  = len(signal) / fs
    print("\n" + "=" * 55)
    print("  rPPG INFERENCE RESULTS")
    print("=" * 55)
    print(f"  Video          : {Path(video_path).name}")
    print(f"  Duration       : {dur:.1f} s  ({len(signal)} frames @ {fs:.1f} fps)")
    print(f"  Estimated HR   : {bpm:.1f} BPM")
    print(f"  Signal SNR     : {snr:.2f} dB")
    print("=" * 55)
    return bpm, snr

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════
def plot_results(
    raw_signal:  np.ndarray,
    proc_signal: np.ndarray,
    quality:     np.ndarray,
    fs:          float,
    video_path:  str,
    save_path:   str = None,
):
    """Generate a 4-panel diagnostic plot."""
    t = np.arange(len(proc_signal)) / fs
    times_bpm, bpms = rolling_bpm(proc_signal, fs)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle(
        f"rPPG Transformer v5 — {Path(video_path).name}",
        fontsize=13, fontweight='bold'
    )

    axes[0].plot(t, raw_signal, color='steelblue', lw=0.8, alpha=0.85)
    axes[0].set_title("Raw Model Output")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(t[0], t[-1])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, proc_signal, color='crimson', lw=0.9)
    axes[1].set_title("Post-processed rPPG Signal (detrended + bandpass filtered)")
    axes[1].set_ylabel("Amplitude (z-scored)")
    axes[1].set_xlim(t[0], t[-1])
    axes[1].grid(True, alpha=0.3)

    if len(times_bpm) > 0:
        axes[2].plot(times_bpm, bpms, color='darkorange', lw=1.5, marker='o',
                     markersize=3, label='Rolling BPM (10 s window)')
        axes[2].axhline(compute_bpm(proc_signal, fs), color='red',
                        linestyle='--', lw=1.2, label=f'Full-video HR = {compute_bpm(proc_signal, fs):.1f} BPM')
        axes[2].set_ylim(30, 220)
        axes[2].set_title("Rolling Heart Rate Estimate")
        axes[2].set_ylabel("BPM")
        axes[2].set_xlim(t[0], t[-1])
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)

    freqs = np.fft.rfftfreq(len(proc_signal), d=1.0 / fs)
    power = np.abs(np.fft.rfft(proc_signal * np.hanning(len(proc_signal)))) ** 2
    band  = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
    axes[3].fill_between(freqs[band] * 60, power[band], alpha=0.4,
                         color='green', label='Cardiac band')
    axes[3].plot(freqs * 60, power, color='navy', lw=0.8)
    peak_bpm = compute_bpm(proc_signal, fs)
    axes[3].axvline(peak_bpm, color='red', linestyle='--',
                    lw=1.5, label=f'Peak = {peak_bpm:.1f} BPM')
    axes[3].set_title("Power Spectrum (cardiac band highlighted)")
    axes[3].set_xlabel("Frequency (BPM)")
    axes[3].set_ylabel("Power")
    axes[3].set_xlim(0, 250)
    axes[3].legend(fontsize=9)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Test rPPG Transformer v5 on a video file."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--model", default="best_rppg_v5.pth", help="Path to the model checkpoint")
    parser.add_argument("--seq_len", type=int, default=256, help="Inference window length")
    parser.add_argument("--stride", type=int, default=64, help="Inference window stride")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--plot", action="store_true", help="Show/save diagnostic plots")
    parser.add_argument("--save_plot", type=str, default=None, help="Path to save the plot image")
    parser.add_argument("--save_signal", type=str, default=None, help="Path to save the predicted rPPG signal")
    parser.add_argument("--device", type=str, default=None, help="Override device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    if args.device:
        cfg.DEVICE = args.device

    print("=" * 55)
    print("  rPPG Transformer v5 — Inference")
    print("=" * 55)
    print(f"  Device   : {cfg.DEVICE.upper()}")
    print(f"  Video    : {args.video}")
    print(f"  Checkpoint: {args.model}")

    t0    = time.time()
    model = load_model(args.model)

    print("\nStep 1 / 3 — Extracting face ROIs from video ...")
    diff_frames, raw_frames, rgb_signals, quality, actual_fps = process_video(args.video)

    print("\nStep 2 / 3 — Running model inference ...")
    raw_signal = run_inference(
        model, diff_frames, raw_frames,
        seq_len=args.seq_len, stride=args.stride, batch_size=args.batch_size,
    )

    print("\nStep 3 / 3 — Post-processing signal ...")
    proc_signal = postprocess_signal(raw_signal, fs=actual_fps)

    bpm, snr = summarise_results(proc_signal, actual_fps, args.video)
    print(f"\n  Total inference time : {time.time() - t0:.1f} s")

    if args.save_signal:
        np.save(args.save_signal, proc_signal)
        print(f"  rPPG signal saved  → {args.save_signal}")

    if args.plot or args.save_plot:
        plot_results(
            raw_signal, proc_signal, quality,
            actual_fps, args.video,
            save_path=args.save_plot,
        )

    return proc_signal, bpm, snr

if __name__ == "__main__":
    main()
