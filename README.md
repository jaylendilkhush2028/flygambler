# 🪰🎲 FlyGambler — a real fruit-fly connectome that learns to gamble

This project takes the **real wiring diagram of a fruit fly's brain** (the FlyWire
connectome) and turns its learning circuit into a **gambler**: it sees betting
"cues", decides whether to bet, gets a squirt of **dopamine when it wins** and a
jolt of **pain when it loses**, and keeps playing until it either **goes broke
("kills itself" — gambler's ruin)** or **gets rich**. The point is to watch how a
*real* reward-learning nervous system reacts to winning and losing — a tiny,
honest window on the machinery behind human/animal gambling.

> **Why a fly?** The fly **mushroom body** is one of the best-understood learning
> circuits in all of biology, and it is *literally* a dopamine-based
> reinforcement-learning machine. So "a brain that gambles, feels dopamine on
> wins and pain on losses, and learns from it" is not sci-fi hand-waving — it
> maps onto real, published fly neuroscience.

## The core question: is the pain of losing *innate* or *learned*?

Run many **random** flies and you can pull the reaction to losing apart into two
pieces:

- **The pain signal is innate.** The PPL1 punishment neurons fire on *every*
  loss, from the very first bet, no matter what the fly has experienced.
- **The compulsion is learned.** The loss-chasing, the relapse-after-quitting,
  the spiral to ruin — that lives entirely in the dopamine-gated KC→MBON
  plasticity. Switch learning off and the *pain still fires*, but the
  self-destructive gambling pattern disappears.

That distinction is what makes "how do you stop the pain?" a tractable question
here. You can't (and shouldn't want to) delete the innate pain signal — it's the
honest feedback that losing is bad. But you *can* target the **learned** machinery
that turns that signal into a spiral. `scripts/04_interventions.py` runs
populations of random flies under mechanistic "treatments" — no-learning,
reward-blocked (wins stop reinforcing the chase), fast-forgetting (extinction),
and self-control (a higher betting threshold) — and measures which ones cut the
ruin rate and the total pain endured. These are **in-silico hypotheses about
mechanism**, not medical advice; real gambling harm is a clinical matter with
real help available.

---

## The honest truth (read this first)

This is a neuroscience **simulation**, and it's easy to over-claim. Here's the line
between what's real and what's metaphor:

**Real:**
- The **connectome is real** — 139,255 neurons and ~15 million synaptic
  connections reconstructed from an adult *Drosophila* brain (FlyWire).
- The **circuit is real** — we pull the actual mushroom-body learning ecosystem
  by cell type: **2,597 Kenyon cells**, **48 output neurons (MBONs)**, the
  **153-neuron PAM reward-dopamine cluster**, the **8-neuron PPL1 punishment
  cluster**, the **APL** sparseness neuron, and **344 projection neurons** as the
  cue input — with their **real synapses** (KC→MBON: ~24k connections in this
  hemisphere).
- The **spiking model is real** — leaky integrate-and-fire dynamics from the
  published whole-brain model (Shiu et al., *Nature* 2024).
- **Dopamine and "pain" are real events in the sim** — winning fires the real PAM
  neurons; losing fires the real PPL1 neurons.

**Our addition (clearly flagged):**
- The published model has **fixed weights and does not learn**. Gambling requires
  learning, so we add **dopamine-gated plasticity** at the KC→MBON synapses. This
  follows well-characterized fly biology (Hige, Cohn, Owald, Aso et al.: dopamine
  rewrites KC→MBON synapses to store valence), but the specific learning rule is
  ours, not the connectome's.

**Metaphor / not literal:**
- The fly **doesn't *feel* pleasure or pain** and doesn't *choose* to die.
  Reward/punishment signals that reshape behavior are the *mechanistic basis* of
  what we call those things — that's the interesting part — but no creature is
  harmed and nothing is conscious here.
- **No real money, no real betting.** "Sports betting" is just the
  reinforcement-learning task; "kills itself" is bankroll reaching \$0 (the
  classic *gambler's ruin*). This is not gambling advice or facilitation.

---

## The biology, in one paragraph

A cue (an odor, or here a "matchup") activates **projection neurons**, which fan
out onto a huge layer of **Kenyon cells (KCs)**. Only a few percent of KCs fire
for any given cue — a **sparse, high-dimensional code** — kept sparse by the
inhibitory **APL** neuron. Those KCs synapse onto a small set of **MBONs** whose
balance of activity drives approach vs. avoidance (here: bet vs. pass).
**Dopaminergic neurons** teach the circuit: the **PAM** cluster signals *reward*
and the **PPL1** cluster signals *punishment*, and they **rewrite the KC→MBON
synapses** of the KCs that were just active. That is the fly's reinforcement
learning — and it's exactly what we drive with wins and losses.

## One gambling round (what "thinking and acting" means here)

1. **See the cue** → drive real projection neurons → real wiring makes a sparse KC code.
2. **Think** → the LIF network spikes for ~100 ms; MBONs integrate KC input through the *current learned* weights.
3. **Act** → bet-drive `D` = total MBON activity → `P(bet) = sigmoid((D − D_ref)/T)` → bet or pass.
4. **Feel** → win → **PAM** dopamine fires; loss → **PPL1** pain fires (reward-prediction error sets the size).
5. **Learn** → dopamine-gated update rewrites KC→MBON weights of the most-active KCs (k-winner-take-all), with slow decay back to baseline (forgetting → relapse).
6. **Live or die** → bankroll updates; stop at **\$0 (ruin)** or the **target (rich)**.

## The two games

- **`RiggedBook`** — a real sportsbook: negative expected value (a house "vig"),
  and the cue tells you *nothing*. The only winning move is not to play. Does the
  fly chase losses (intermittent wins are a variable-ratio schedule — the classic
  driver of compulsive gambling), quit, or relapse after it forgets?
- **`LearnableBook`** — the cue genuinely predicts the outcome. A fly that learns
  to bet the good cues and pass the bad ones can actually **get rich**. Tests
  whether the real mushroom-body circuit can find a true edge.

---

## Architecture & performance (why two engines)

The full 140k-neuron model is real but **slow per run** in Brian2 (large fixed
per-`run()` overhead), and a gambling life is *thousands* of rounds. So:

- **Whole brain (Brian2):** used to **validate** that the real published model
  boots and spikes (`scripts/01_validate_wholebrain.py`).
- **Mushroom-body subnetwork (fast NumPy engine):** the interactive gambling loop
  runs on the ~3,150-neuron MB ecosystem extracted from *the same connectome*,
  using a lightweight integrator of the **identical LIF equations and identical
  connectome weights**. Validated against Brian2 (matching sparse KC codes;
  ~5–10% KC sparseness, as in real flies). This is what makes live play and whole
  populations feasible.

Both engines consume one shared `Circuit` (same wiring, same weights) — see
`flygambler/connectome.py`.

---

## Install & run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install brian2 numpy scipy pandas pyarrow matplotlib setuptools
```

The connectome data ships in the vendored model repo and is staged under `data/`
(`Connectivity_783.parquet`, `Completeness_783.csv`) plus FlyWire cell-type
annotations (`neuron_annotations.tsv`). See **Data & credits** below.

```bash
# 0) prove the real whole brain runs (all ~140k neurons)
.venv/bin/python scripts/01_validate_wholebrain.py

# 1) one fly's gambling life (random each run; writes runs/life_*.json + .png)
.venv/bin/python scripts/02_run_fly.py --book learnable
.venv/bin/python scripts/02_run_fly.py --book rigged

# 2) a population of random flies — the distribution of fates
.venv/bin/python scripts/03_run_population.py --book rigged --n 60

# 3) innate vs learned + "how do you stop the pain?" — the intervention study
.venv/bin/python scripts/04_interventions.py --n 12

# 4) temperaments — chase to ruin vs. get too sad and stop playing
.venv/bin/python scripts/05_temperaments.py --n 20

# 5) watch the REAL brain gamble FOREVER, live in your terminal
#    (respawns a new fly on every $0 or $100; archives each life to runs/lives.jsonl)
.venv/bin/python scripts/06_watch_forever.py --temperament withdrawer
```

Runs are **randomized each time** (pass `--seed` for reproducibility).

**The dashboard** (`dashboard/fly_gambler.html`) is a **live, forever** simulator:
it never stops — every time the fly hits **$0 or $100 a new one is born** — and it
shows the fly's **mood** driving whether it bets (watch "wants to bet" vs "actually
bets" diverge for a *withdrawer*). Life history is saved in your browser. It runs a
fast **behavioral model** of the circuit so it can play endlessly in a page; the
full **spiking connectome** runs forever in Python via `scripts/06` (archiving to
`runs/lives.jsonl`).

---

## Results

**The whole brain runs.** All 138,639 neurons and 15,091,983 synapses build and
simulate; stimulating 685 real projection neurons lights up ~10k neurons
brain-wide and drives the mushroom body — the published LIF model boots and
computes (fast engine: 60 ms of brain time in ~2 s).

**A beatable game → it learns and gets rich.** On the learnable book a fly starts
treading water, then its KC→MBON weights discover which cues pay: bet-rate on
good (+EV) cues climbs toward ~0.85 while bad (−EV) cues fall toward ~0.30, win
rate on its bets rises above 50%, and the bankroll climbs from \$25 to the \$100
target (**RICH**). See `runs/life_learnable.png`.

**A rigged game → it chases losses to ruin.** On the negative-EV book there is no
edge to find. A fly briefly quits, but an intermittent win streak (a variable-
ratio schedule — the same thing that hooks human gamblers) pulls it back into
compulsive betting, and the house edge grinds it to \$0 (**RUIN**). See
`runs/life_rigged.png`.

**Innate vs. learned, and what reduces harm.** Across random-fly populations on
the rigged book (`scripts/04_interventions.py`), the innate pain signal is
present in every condition, but the *path to ruin is learned*. The clearest
result: **blocking win-driven reinforcement takes ruin from ~40% to 0% and cuts
total pain by ~80%**, while merely adding self-control (a higher betting
threshold) doesn't help — the harm is a learned reinforcement loop, and breaking
that loop beats willpower. The same plasticity is exactly what lets a fly profit
on a beatable game. See `runs/interventions.png`.

**Get sad and stop, or chase to ruin?** A slow **despair state** builds as a fly
loses. Whether that leads to withdrawal or compulsion is a single temperament
knob (`withdrawal`): a *chaser* stays sad but keeps betting and goes broke; a
*withdrawer* gets too sad, disengages, and stops playing — surviving with money
left (the sadness is protective). This is the learned-helplessness / behavioral-
withdrawal side of losing, and flies really do show persistent negative internal
states. See `runs/temperaments.png` and `scripts/05_temperaments.py`.

_(This is a model: the numbers are hypotheses about mechanism, not claims about
real brains, and nothing here is medical advice — real gambling harm is a
clinical matter with real help available.)_

---

## Data & credits

This project stands on real data and a real published model. Please cite them:

- **Whole-brain LIF model:** Shiu, Sang, et al., *A leaky integrate-and-fire
  computational model based on the connectome of the entire adult Drosophila
  brain…*, **Nature** (2024). Code: <https://github.com/philshiu/Drosophila_brain_model>
  (vendored under `vendor/`, MIT-licensed).
- **FlyWire connectome:** Dorkenwald et al., *Neuronal wiring diagram of an adult
  brain*, **Nature** (2024).
- **Cell-type annotations:** Schlegel et al., *Whole-brain annotation and
  multi-connectome cell typing of Drosophila*, **Nature** (2024). Annotations from
  <https://github.com/flyconnectome/flywire_annotations>.
- **Mushroom-body learning biology:** Aso et al., *eLife* (2014); Hige et al.,
  *Neuron* (2015); Cohn et al., *Cell* (2015); Owald & Waddell.

The learning extension (dopamine-gated KC→MBON plasticity + the betting task) is
this project's own work, built on top of the above.

## Layout

```
flygambler/     connectome.py (load + Circuit), fastlif.py (fast engine),
                brain.py (learning circuit), game.py (books), simulate.py, plots.py
scripts/        01 validate whole brain · 02 one fly · 03 population
dashboard/      fly_gambler.html (watch it gamble)
data/           connectome + annotations (git-ignored; staged from the sources above)
vendor/         the published Brian2 model (reused for validation)
```
