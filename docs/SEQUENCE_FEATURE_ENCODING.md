# Sequence Feature Encoding (Transformer)

This document describes the sequence feature encoding for transformer models, implemented in `riichienv-core/src/observation/sequence_features.rs` with a Python wrapper at `riichienv-ml/src/riichienv_ml/features/sequence_features.py`.

The encoding design is based on [Kanachan v3](https://github.com/Cryolite/kanachan/wiki/%5Bv3%5DNotes-on-Training-Data) as a subset — `Room` (5 values) and `Grade` (4x16=64 values) are removed since they are online-platform-dependent and unavailable via MJAI protocol.

## Overview

Unlike the CNN encoder (`obs.encode()`) which produces spatial `(C, 34)` tensors, the sequence feature encoding produces **four heterogeneous feature groups** designed for embedding-based transformer architectures:

| Feature Group | Shape | Type | Description |
|---------------|-------|------|-------------|
| **Sparse** | `(25,)` | int64 | Categorical tokens for embedding lookup |
| **Numeric** | `(12,)` | float32 | Continuous scalar features |
| **Progression** | `(256, 5)` | int64 | Action history as 5-tuple sequences |
| **Candidates** | `(32, 4)` | int64 | Legal actions as 4-tuple sets |

Each variable-length group is padded to its maximum length, with accompanying boolean masks indicating real vs. padding entries.

## Tile Encodings

### kan37 (37 tiles, red fives distinct)

Used for discard, dora, drawn tile, daiminkan, and kakan encoding.

| Range | Tiles |
|-------|-------|
| 0 | Red 5m |
| 1-9 | 1m-9m |
| 10 | Red 5p |
| 11-19 | 1p-9p |
| 20 | Red 5s |
| 21-29 | 1s-9s |
| 30-36 | E, S, W, N, P (white), F (green), C (red) |

Conversion from 136-tile ID:
- `tile_id == 16` -> 0 (red 5m)
- `tile_id == 52` -> 10 (red 5p)
- `tile_id == 88` -> 20 (red 5s)
- Otherwise: `tile_type = tile_id / 4`, then `tile_type + 1` (man), `+2` (pin), `+3` (sou/honor)

### kan34 (34 tile types, no red five distinction)

Used for ankan encoding. Identity map: `tile_type (0-33)`.

### Relative Seat

`(target - actor + 3) % 4`:
- 0 = shimocha (right / downstream)
- 1 = toimen (across)
- 2 = kamicha (left / upstream)

## 1. Sparse Features

**Vocabulary size: 442, max tokens: 25, padding index: 441**

Each observation produces 5-25 sparse tokens. Each token is an index into an embedding table.

| Offset | Count | Feature | Source |
|--------|-------|---------|--------|
| 0-1 | 2 | Game style (0=tonpuusen, 1=hanchan) | parameter |
| 2-5 | 4 | Seat (`player_id`) | `obs.player_id` |
| 6-8 | 3 | Chang / round wind (E/S/W) | `obs.round_wind` |
| 9-12 | 4 | Ju / dealer round (0-3) | `obs.oya` |
| 13-82 | 70 | Tiles remaining (0-69) | derived from visible tiles |
| 83-267 | 185 | Dora indicators (5 slots x 37 tiles) | `obs.dora_indicators` |
| 268-403 | 136 | Hand tile instances (tile_id 0-135) | `obs.hands[player_id]` |
| 404-440 | 37 | Drawn tile (kan37 encoding) | last tsumo event |
| 441 | 1 | Padding | - |

**Token composition per observation:**
- 4 fixed tokens (game style + seat + round wind + dealer)
- 1 tiles-remaining token
- 1-5 dora indicator tokens
- ~13 hand tile tokens (varies with melds)
- 0-1 drawn tile token
- Total: typically 19-24 tokens

### Rust API

```rust
obs.encode_seq_sparse(game_style: u8) -> Vec<u16>
```

### Python API (raw)

```python
sparse_bytes = obs.encode_seq_sparse(game_style=1)
sparse = np.frombuffer(sparse_bytes, dtype=np.uint16)  # variable length
```

## 2. Numeric Features

**Fixed: 12 floats**

| Index | Feature | Source |
|-------|---------|--------|
| 0 | Honba (current) | `obs.honba` |
| 1 | Riichi deposits (current) | `obs.riichi_sticks` |
| 2 | Score (self) | `obs.scores[player_id]` |
| 3 | Score (right / shimocha) | `obs.scores[(player_id+1)%4]` |
| 4 | Score (across / toimen) | `obs.scores[(player_id+2)%4]` |
| 5 | Score (left / kamicha) | `obs.scores[(player_id+3)%4]` |
| 6 | Honba (round start) | `start_kyoku` event |
| 7 | Riichi deposits (round start) | `start_kyoku` event |
| 8-11 | Scores at round start (self-relative order) | `start_kyoku` event |

**Note:** Scores are raw values (e.g. 25000), not normalized. Normalization should be applied in the model or data pipeline as needed.

### Rust API

```rust
obs.encode_seq_numeric() -> [f32; 12]
```

### Python API (raw)

```python
numeric_bytes = obs.encode_seq_numeric()
numeric = np.frombuffer(numeric_bytes, dtype=np.float32)  # shape (12,)
```

## 3. Progression Features (Action History)

**5-tuple sequence, max 256 entries (default)**

Each action from the kyoku start to the current decision point is encoded as a 5-tuple `(actor, type, moqie, liqi, from)`.

### Tuple Fields

| Field | Vocab | Values |
|-------|-------|--------|
| actor | 5 | 0-3 (seats), 4 (padding/marker) |
| type | 277 | see table below |
| moqie | 3 | 0=tedashi (hand tile), 1=tsumogiri (drawn tile), 2=N/A |
| liqi | 3 | 0=no riichi, 1=with riichi declaration, 2=N/A |
| from | 5 | 0=shimocha, 1=toimen, 2=kamicha, 4=N/A |

**Padding tuple:** `(4, 276, 2, 2, 4)`

### Type Encoding (277 values)

| Range | Count | Action | Encoding |
|-------|-------|--------|----------|
| 0 | 1 | Beginning-of-round marker | Fixed value |
| 1-37 | 37 | Discard | `1 + kan37(tile)` |
| 38-127 | 90 | Chi | `38 + chi_encoding` |
| 128-167 | 40 | Pon | `128 + pon_encoding` |
| 168-204 | 37 | Daiminkan | `168 + kan37(tile)` |
| 205-238 | 34 | Ankan | `205 + kan34(tile)` |
| 239-275 | 37 | Kakan | `239 + kan37(tile)` |
| 276 | 1 | Padding | - |

### MJAI Event to Tuple Mapping

| Event | Tuple |
|-------|-------|
| `start_kyoku` | `(4, 0, 2, 2, 4)` |
| `dahai` | `(actor, 1+kan37, moqie, liqi, 4)` |
| `chi` | `(actor, 38+chi_enc, 2, 2, relative_from)` |
| `pon` | `(actor, 128+pon_enc, 2, 2, relative_from)` |
| `daiminkan` | `(actor, 168+kan37, 2, 2, relative_from)` |
| `ankan` | `(actor, 205+kan34, 2, 2, 4)` |
| `kakan` | `(actor, 239+kan37, 2, 2, 4)` |

- For `dahai`: `liqi=1` if preceded by a `reach` event from the same actor
- `tsumo`, `dora`, `reach_accepted` events are **not** included in progression

### Rust API

```rust
obs.encode_seq_progression() -> Vec<[u16; 5]>
```

### Python API (raw)

```python
prog_bytes = obs.encode_seq_progression()
prog = np.frombuffer(prog_bytes, dtype=np.uint16).reshape(-1, 5)  # variable length
```

## 4. Candidate Features (Legal Actions)

**4-tuple set, max 32 entries (default)**

Each legal action is encoded as a 4-tuple `(type, moqie, liqi, from)`.

### Tuple Fields

| Field | Vocab | Values |
|-------|-------|--------|
| type | 280 | see table below |
| moqie | 3 | 0=tedashi, 1=tsumogiri, 2=N/A |
| liqi | 3 | 0=no riichi, 1=with riichi, 2=N/A |
| from | 4 | 0=shimocha, 1=toimen, 2=kamicha, 3=self |

**Padding tuple:** `(279, 2, 2, 3)`

### Type Encoding (280 values)

| Range | Count | Action | Encoding |
|-------|-------|--------|----------|
| 0-36 | 37 | Discard | `kan37(tile)` |
| 37-70 | 34 | Ankan | `37 + kan34(tile)` |
| 71-107 | 37 | Kakan | `71 + kan37(tile)` |
| 108 | 1 | Tsumo (win) | Fixed |
| 109 | 1 | Kyushu kyuhai (9 terminals draw) | Fixed |
| 110 | 1 | Pass | Fixed |
| 111-200 | 90 | Chi | `111 + chi_encoding` |
| 201-240 | 40 | Pon | `201 + pon_encoding` |
| 241-277 | 37 | Daiminkan | `241 + kan37(tile)` |
| 278 | 1 | Ron (win) | Fixed |
| 279 | 1 | Padding | - |

**Note:** `Riichi` is not a separate candidate type. When riichi is available, the corresponding discard candidates should be interpreted with `liqi=1`.

### Rust API

```rust
obs.encode_seq_candidates() -> Vec<[u16; 4]>
```

### Python API (raw)

```python
cand_bytes = obs.encode_seq_candidates()
cand = np.frombuffer(cand_bytes, dtype=np.uint16).reshape(-1, 4)  # variable length
```

## Chi Encoding (90 patterns)

90 = 3 suits x 30 per suit.

Each suit has 7 possible sequence starts (rank 1-7, i.e. 1-2-3 through 7-8-9). For each sequence:
- 3 call positions (which tile in the sequence was called from discard)
- If the sequence contains a 5 tile: +3 red-five variants (one per call position)

Per suit: sequences not containing 5 have 3 patterns, sequences containing 5 have 6 patterns.

```
Rank 1-2-3 (no 5): 3 patterns  ->  3
Rank 2-3-4 (no 5): 3 patterns  ->  3
Rank 3-4-5 (has 5): 6 patterns ->  6
Rank 4-5-6 (has 5): 6 patterns ->  6
Rank 5-6-7 (has 5): 6 patterns ->  6
Rank 6-7-8 (no 5): 3 patterns  ->  3
Rank 7-8-9 (no 5): 3 patterns  ->  3
Total per suit:                    30
```

Suit offsets: manzu=0, pinzu=30, souzu=60.

## Pon Encoding (40 patterns)

40 = 3 suits x 11 + 7 honors.

Per suit (11 patterns):
- Ranks 1-4 (non-five): 4 patterns (one each)
- Rank 5 (five): 3 variants (no red, red-in-hand, red-called)
- Ranks 6-9 (non-five): 4 patterns (one each)

Honors: 7 patterns (E, S, W, N, P, F, C), one each.

Suit offsets: manzu=0, pinzu=11, souzu=22, honors=33.

## Python Wrapper: SequenceFeatureEncoder

`riichienv_ml.features.sequence_features.SequenceFeatureEncoder` provides padded torch tensors with masks.

### Usage

```python
from riichienv import RiichiEnv
from riichienv_ml.features.sequence_features import SequenceFeatureEncoder

env = RiichiEnv(game_mode="4p-red-half")
obs_dict = env.reset()
enc = SequenceFeatureEncoder(n_players=4, game_style=1)

for pid, obs in obs_dict.items():
    features = enc.encode(obs)
    # features["sparse"]      -- (25,) int64, padded with 441
    # features["numeric"]     -- (12,) float32
    # features["progression"] -- (256, 5) int64, padded with (4, 276, 2, 2, 4)
    # features["candidates"]  -- (32, 4) int64, padded with (279, 2, 2, 3)
    # features["sparse_mask"] -- (25,) bool, True for real tokens
    # features["prog_mask"]   -- (256,) bool, True for real entries
    # features["cand_mask"]   -- (32,) bool, True for real entries
```

### Constants

```python
SequenceFeatureEncoder.SPARSE_VOCAB_SIZE  # 442
SequenceFeatureEncoder.MAX_SPARSE_LEN     # 25
SequenceFeatureEncoder.MAX_PROG_LEN       # 256 (default; V1 compat: 512)
SequenceFeatureEncoder.MAX_CAND_LEN       # 32  (default; V1 compat: 64)
SequenceFeatureEncoder.NUM_NUMERIC         # 12
SequenceFeatureEncoder.PROG_DIMS           # (5, 277, 3, 3, 5)
SequenceFeatureEncoder.CAND_DIMS           # (280, 3, 3, 4)
```

## Implementation

| File | Package | Description |
|------|---------|-------------|
| `riichienv-core/src/observation/sequence_features.rs` | riichienv-core | Rust encoding logic (~470 lines) |
| `riichienv-core/src/observation/python.rs` | riichienv-core | PyO3 bindings (4 methods) |
| `riichienv-ml/src/riichienv_ml/features/sequence_features.py` | riichienv-ml | Python wrapper (~115 lines) |
