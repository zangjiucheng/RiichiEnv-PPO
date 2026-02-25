# Game Rules Configuration

Detailed game mechanics can be configured using the `GameRule` struct.

> [!NOTE]
> Configurable rule options are still under development and not yet exhaustive. If you find missing rules or have suggestions for additional configuration options, please report them via [GitHub Issues #43](https://github.com/smly/RiichiEnv/issues/43).

## Pao (Responsibility Payment)

When a winning hand includes a Pao-triggering Yakuman (Daisangen or Daisuushii), the Pao player shares payment responsibility with the deal-in player (Ron) or bears it alone (Tsumo).

For a **single Pao Yakuman** (e.g., Daisangen only), both platforms behave identically:

- **Ron**: The total is split 50/50 between the Pao player and the deal-in player.
- **Tsumo**: The Pao player pays the full amount. Other players pay nothing.

The `yakuman_pao_is_liability_only` flag only matters for **composite hands where Pao and non-Pao Yakumans coexist** (e.g., Daisangen + Tsuuiisou). In this case:

**Ron**: The flag controls the split between the Pao player and the deal-in player (both 4P and 3P):

- **Tenhou** (`false`): The **total** Ron amount is split 50/50 between the Pao player and the deal-in player.
- **Mahjong Soul** (`true`): Only the **Pao-triggering Yakuman portion** is split 50/50. The non-Pao Yakuman portion is paid entirely by the deal-in player.

**Tsumo**: The flag controls how the non-Pao Yakuman portion is settled (both 4P and 3P):

- **Tenhou** (`false`): Pao player pays the full Tsumo amount including non-Pao Yakumans. Other players pay nothing.
- **Mahjong Soul** (`true`): Pao player pays only the Pao-triggering Yakuman portion. The remaining Yakumans are split normally among all non-winning players.

| Flag | Description |
|------|-------------|
| `.yakuman_pao_is_liability_only` | Whether to limit Pao liability to the Pao-triggering Yakuman portion only when combined with non-Pao Yakumans (Mahjong Soul style). Affects both Ron and Tsumo settlement in both 4P and 3P. If false, Pao covers the full amount (Tenhou style). Does not affect single-Yakuman wins (both styles are identical). |

## Kuikae (Swap Calling) Restriction

Controls whether players are forbidden from discarding a tile that completes the same group they just called (including flank tiles for Chi). Both Tenhou and Mahjong Soul enforce this restriction.

| Flag | Description |
|------|-------------|
| `.kuikae_forbidden` | When `True`, kuikae is forbidden: after Chi/Pon, the called tile and flank tiles cannot be discarded. When `False`, no kuikae restriction applies. |

## Kokushi Musou Rules

| Flag | Description |
|------|-------------|
| `.allows_ron_on_ankan_for_kokushi_musou` | Whether to allow Ron on a closed Kan (Ankan) for Kokushi Musou (Chankan). |
| `.is_kokushi_musou_13machi_double` | Whether to treat Kokushi Musou 13-sided wait as a Double Yakuman. |

## Double Yakuman Pattern Rules

Controls whether specific Yakuman pattern variants are treated as Double Yakuman. Tenhou treats all pattern variants as single Yakuman, while Mahjong Soul treats them as Double Yakuman. Note that combinations of independent Yakuman (e.g., Tsuuiisou + Daisangen) always stack regardless of these flags.

| Flag | Description |
|------|-------------|
| `.is_suuankou_tanki_double` | Whether to treat Suuankou Tanki (四暗刻単騎) as a Double Yakuman. |
| `.is_junsei_chuurenpoutou_double` | Whether to treat Junsei Chuurenpoutou (純正九蓮宝燈) as a Double Yakuman. |
| `.is_daisuushii_double` | Whether to treat Daisuushii (大四喜) as a Double Yakuman. |

## Sanchaho (Triple Ron)

| Flag | Description |
|------|-------------|
| `.sanchaho_is_draw` | Whether triple ron (三家和, all non-discarders declaring Ron simultaneously) causes an abortive draw. When enabled (Tenhou), no scoring occurs and the round ends as a draw with renchan. When disabled (Mahjong Soul), all three Ron declarations are processed normally. |

## Open Kan Dora Reveal Timing

Controls when dora indicators are revealed after an open kan (Daiminkan/Kakan) declaration.

Ankan (closed kan) always reveals dora immediately before the rinshan tsumo, regardless of this flag.

| Flag | Description |
|------|-------------|
| `.open_kan_dora_after_discard` | Whether open kan (Daiminkan/Kakan) dora is revealed after the discard. When `True` (Tenhou / Mahjong Soul style), dora is revealed after the discard. When `False` (Mortal mjai protocol style), dora is revealed before the discard. |

### Event Order

When `open_kan_dora_after_discard = True` (Tenhou / Mahjong Soul):

**Ankan (Closed Kan)**:
```
ankan → dora → tsumo (rinshan) → dahai
```

**Kakan/Daiminkan (Open/Added Kan)**:
```
kakan → tsumo (rinshan) → dahai → dora
daiminkan → tsumo (rinshan) → dahai → dora
```

When `open_kan_dora_after_discard = False` (Mortal):

**Ankan (Closed Kan)** - same as above:
```
ankan → dora → tsumo (rinshan) → dahai
```

**Kakan/Daiminkan (Open/Added Kan)**:
```
kakan → tsumo (rinshan) → dora → dahai
daiminkan → tsumo (rinshan) → dora → dahai
```

Note: Rinshan kaihou (winning on rinshan draw) always includes the kan dora.

### Usage

```python
from riichienv import RiichiEnv, GameRule

# Mortal style: open kan dora before discard (default for default_mortal())
rule = GameRule(open_kan_dora_after_discard=False)
env = RiichiEnv(rule=rule)

# Tenhou / Mahjong Soul style: open kan dora after discard
rule = GameRule(open_kan_dora_after_discard=True)
env = RiichiEnv(rule=rule)
```

## Mortal Preset and Kan Dora Timing

The `default_mortal()` preset is designed for compatibility with Mortal's mjai protocol implementation. It uses the same rule flags as `default_tenhou()` except for `open_kan_dora_after_discard`.

In Tenhou's actual game rules, open kan (Kakan/Daiminkan) dora is revealed **after** the player's discard. However, Mortal's mjai protocol implementation (`libriichi/src/arena/board.rs`) encodes the dora event **before** the dahai event for open kans:

```rust
// Mortal board.rs — Kakan/Daiminkan handler
Event::Daiminkan { actor, .. } | Event::Kakan { actor, .. } => {
    // ...
    self.need_new_dora_at_discard = Some(());
    // ...
}

// Mortal board.rs — Dahai handler
Event::Dahai { actor, pai, .. } => {
    if self.need_new_dora_at_discard.take().is_some() {
        self.add_new_dora()?;   // dora event emitted BEFORE dahai
    }
    self.broadcast(&ev.event);  // then dahai
    self.add_log(ev.clone());
}
```

This produces the mjai event sequence `kakan → tsumo → dora → dahai`, where the dora event appears before the dahai in the protocol log. For ankan, Mortal reveals dora immediately after the kan declaration (before rinshan tsumo), which is the same as Tenhou:

```rust
// Mortal board.rs — Ankan handler
Event::Ankan { actor, .. } => {
    // ...
    self.add_new_dora()?;  // "Immediately add new dora"
    // ...
}
```

The `default_mortal()` preset (`open_kan_dora_after_discard = false`) matches this protocol behavior, enabling compatibility with Mortal for agent evaluation and validation.

## Platform-Specific Rule Presets

Differences in standard ranked match rules across major platforms and Mortal.

### 4-Player Presets

| Flag | `default_tenhou()` | `default_mjsoul()` | `default_mortal()` |
|------|--------|--------------|--------------|
| `.allows_ron_on_ankan_for_kokushi_musou` | `False` | `True` | `False` |
| `.is_kokushi_musou_13machi_double` | `False` | `True` | `False` |
| `.is_suuankou_tanki_double` | `False` | `True` | `False` |
| `.is_junsei_chuurenpoutou_double` | `False` | `True` | `False` |
| `.is_daisuushii_double` | `False` | `True` | `False` |
| `.yakuman_pao_is_liability_only` | `False` | `True` | `False` |
| `.sanchaho_is_draw` | `True` | `False` | `True` |
| `.open_kan_dora_after_discard` | `True` | `True` | `False` |

### 3-Player (Sanma) Presets

| Flag | `default_tenhou_sanma()` | `default_mjsoul_sanma()` | `default_mortal_sanma()` |
|------|--------|--------------|--------------|
| `.allows_ron_on_ankan_for_kokushi_musou` | `False` | `True` | `False` |
| `.is_kokushi_musou_13machi_double` | `False` | `True` | `False` |
| `.is_suuankou_tanki_double` | `False` | `True` | `False` |
| `.is_junsei_chuurenpoutou_double` | `False` | `True` | `False` |
| `.is_daisuushii_double` | `False` | `True` | `False` |
| `.yakuman_pao_is_liability_only` | `False` | `True` | `False` |
| `.open_kan_dora_after_discard` | `True` | `True` | `False` |
