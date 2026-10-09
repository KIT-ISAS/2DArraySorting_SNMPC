"""Fast PTCR activation-sequence tree search (with legacy fallback in the controller).

Optimizations (relative to the original Python BFS):

1. Path as an immutable ``tuple`` of actor indices (no list copies).
2. Scalar Conditions 3/4 (no NumPy max/min on tiny arrays).
3. Upstream check via a tight loop (no temporary boolean mask).
4. Compact leaf storage; reconstruct control sequences only at the end.
5. Optional Numba filter for candidate actors on each expand.
6. ``PTCRController`` selects fast vs. legacy via config ``tree_search.use_fast``.
   Numba expand is controlled by ``tree_search.use_numba`` (:func:`set_tree_numba_enabled`).

``PTCRController`` calls ``warmup_ptcr_tree_numba`` at construction when the fast
tree search with Numba expand is active.
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Sequence, Tuple

import numpy as np

from absl import logging

try:
    from numba import njit

    _NUMBA = True
except ImportError:  # pragma: no cover
    _NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def wrap(fn):
            return fn

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return wrap


_TREE_NUMBA_ENABLED = True


def set_tree_numba_enabled(enabled: bool) -> None:
    """
    Enables or disables Numba expand helpers for the fast tree.

    :param enabled: A Boolean. If False, the fast tree uses the pure-Python expand path.
    """
    global _TREE_NUMBA_ENABLED
    _TREE_NUMBA_ENABLED = bool(enabled)


def tree_numba_enabled() -> bool:
    """
    Whether Numba expand helpers for the fast tree are available and enabled.

    :returns: A Boolean, False if Numba is missing or disabled via
        :func:`set_tree_numba_enabled`.
    """
    if not _NUMBA:
        return False
    return _TREE_NUMBA_ENABLED


def warmup_ptcr_tree_numba(
    *,
    num_actors: int = 4,
    num_particles: int = 8,
    t_lead: float = 3.0,
    T_step: float = 9.0,
    t_cycle: float = 9.0,
) -> None:
    """
    Triggers JIT compilation of tree-search Numba kernels.

    No-op if ``tree_numba_enabled()`` is false. Uses small dummy arrays to exercise
    both empty and non-empty path prefixes.

    :param num_actors: An integer, the number of actors ``Na`` (compile-time size).
    :param num_particles: An integer, the dummy path / particle length (``>= 2``).
    :param t_lead: A float, the actor lead time used in the expand step.
    :param T_step: A float, the internal stage length.
    :param t_cycle: A float, the actor cycle time.
    """
    if not tree_numba_enabled():
        return
    logging.info(
        "Compiling PTCR tree-search Numba kernels (num_actors=%d)…",
        int(num_actors),
    )
    na = max(int(num_actors), 1)
    # Path / horizon length for assigned[] and delta_toa[:, p]; see docstring.
    np_particles = max(int(num_particles), 2)
    # Uniform finite deltas: upstream_ok only needs comparable floats, not physics.
    delta_toa = np.full((na, np_particles), 10.0, dtype=np.float64)
    valid = np.ones((na, np_particles), dtype=np.bool_)
    undisturbed = np.ones((na, np_particles), dtype=np.bool_)
    assigned = np.full(np_particles, -1, dtype=np.int32)
    select_child_actors_numba(
        0,
        assigned,
        delta_toa,
        valid,
        undisturbed,
        float(t_lead),
        float(T_step),
        float(t_cycle),
        na,
    )
    assigned[0] = 0
    select_child_actors_numba(
        1,
        assigned,
        delta_toa,
        valid,
        undisturbed,
        float(t_lead),
        float(T_step),
        float(t_cycle),
        min(2, na),
    )

# ---------------------------------------------------------------------------
# Scalar / Numba checks
# ---------------------------------------------------------------------------

@njit(cache=True)
def _upstream_ok_numba(
    actor_idx: int,
    depth: int,
    delta_toa: np.ndarray,
    undisturbed: np.ndarray,
) -> bool:
    toa = delta_toa[actor_idx, depth]
    na = delta_toa.shape[0]
    for a in range(na):
        if delta_toa[a, depth] <= toa and (not undisturbed[a, depth]):
            return False
    return True


@njit(cache=True)
def _free_and_once_ok_numba(
    actor_idx: int,
    depth: int,
    assigned: np.ndarray,
    delta_toa: np.ndarray,
    activation_time: float,
    t_lead: float,
    t_cycle: float,
    T_step: float,
) -> bool:
    """Conditions 3 + 4 using only the path prefix ``assigned[:depth]``."""
    current_step = int(activation_time / T_step)
    latest_prev = -np.inf
    next_follow = np.inf
    for p in range(depth):
        if assigned[p] != actor_idx:
            continue
        t_act = delta_toa[actor_idx, p] - t_lead
        if not math.isfinite(t_act):
            continue
        # Condition 4: at most one activation per internal time step
        if int(math.floor(t_act / T_step)) == current_step:
            return False
        if t_act <= activation_time:
            if t_act > latest_prev:
                latest_prev = t_act
        elif t_act < next_follow:
            next_follow = t_act
    if latest_prev + t_cycle > activation_time:
        return False
    if activation_time + t_cycle > next_follow:
        return False
    return True


@njit(cache=True)
def select_child_actors_numba(
    depth: int,
    assigned: np.ndarray,
    delta_toa: np.ndarray,
    valid_mask: np.ndarray,
    undisturbed: np.ndarray,
    t_lead: float,
    T_step: float,
    t_cycle: float,
    max_branches: int,
) -> np.ndarray:
    """
    Selects feasible child actors for one expand step of the fast tree.

    :param depth: An integer, the current tree depth (particle index).
    :param assigned: An integer np.array of shape ``[num_particles]``, the path of
        actor indices (``-1`` padding after the prefix of length ``depth``).
    :param delta_toa: A float np.array of shape ``[num_actors, num_particles]``,
        arrival-time deltas.
    :param valid_mask: A Boolean np.array of shape ``[num_actors, num_particles]``,
        the ready (condition-2) mask.
    :param undisturbed: A Boolean np.array of shape ``[num_actors, num_particles]``,
        the partial undisturbed (condition-1) mask.
    :param t_lead: A float, the actor lead time.
    :param T_step: A float, the internal stage length.
    :param t_cycle: A float, the actor cycle time.
    :param max_branches: An integer, the maximum number of child actors to return.

    :returns: An integer np.array of selected actor indices (length ``<= max_branches``),
        including the no-actor sentinel ``-1`` when offered.
    """
    na = delta_toa.shape[0]
    out = np.empty(na, dtype=np.int32)
    n_out = 0
    for a in range(na):
        if n_out >= max_branches:
            break
        if not valid_mask[a, depth]:
            continue
        toa = delta_toa[a, depth]
        act = toa - t_lead
        if not _upstream_ok_numba(a, depth, delta_toa, undisturbed):
            continue
        if not _free_and_once_ok_numba(
            a, depth, assigned, delta_toa, act, t_lead, t_cycle, T_step
        ):
            continue
        out[n_out] = a
        n_out += 1
    return out[:n_out]


def _upstream_ok_py(
    actor_idx: int,
    depth: int,
    delta_toa: np.ndarray,
    undisturbed: np.ndarray,
) -> bool:
    toa = delta_toa[actor_idx, depth]
    for a in range(delta_toa.shape[0]):
        if delta_toa[a, depth] <= toa and not undisturbed[a, depth]:
            return False
    return True


def _free_and_once_ok_py(
    actor_idx: int,
    depth: int,
    assigned: Sequence[int],
    delta_toa: np.ndarray,
    activation_time: float,
    t_lead: float,
    t_cycle: float,
    T_step: float,
) -> bool:
    current_step = int(activation_time / T_step)
    latest_prev = -math.inf
    next_follow = math.inf
    for p in range(depth):
        if assigned[p] != actor_idx:
            continue
        t_act = float(delta_toa[actor_idx, p] - t_lead)
        if not math.isfinite(t_act):
            continue
        if int(math.floor(t_act / T_step)) == current_step:
            return False
        if t_act <= activation_time:
            if t_act > latest_prev:
                latest_prev = t_act
        elif t_act < next_follow:
            next_follow = t_act
    if latest_prev + t_cycle > activation_time:
        return False
    if activation_time + t_cycle > next_follow:
        return False
    return True


def _select_child_actors_py(
    depth: int,
    assigned: Sequence[int],
    delta_toa: np.ndarray,
    valid_mask: np.ndarray,
    undisturbed: np.ndarray,
    t_lead: float,
    T_step: float,
    t_cycle: float,
    max_branches: int,
) -> List[int]:
    out: List[int] = []
    na = delta_toa.shape[0]
    for a in range(na):
        if len(out) >= max_branches:
            break
        if not valid_mask[a, depth]:
            continue
        act = float(delta_toa[a, depth] - t_lead)
        if not _upstream_ok_py(a, depth, delta_toa, undisturbed):
            continue
        if not _free_and_once_ok_py(
            a, depth, assigned, delta_toa, act, t_lead, t_cycle, T_step
        ):
            continue
        out.append(a)
    return out


# ---------------------------------------------------------------------------
# Control-sequence reconstruction (shared by fast path)
# ---------------------------------------------------------------------------

def control_sequence_from_assignment(
    assigned: Sequence[int],
    depth: int,
    delta_toa: np.ndarray,
    t_lead: float,
    T_step: float,
    N_horizon: int,
) -> np.ndarray:
    """
    Reconstructs a control-sequence tensor from a compact actor-assignment path.

    :param assigned: A sequence of integer actor indices of length ``depth``.
    :param depth: An integer, the number of assigned particles on the path.
    :param delta_toa: A float np.array of shape ``[num_actors, num_particles]``,
        arrival-time deltas for the filtered ejectable particles.
    :param t_lead: A float, the actor lead time.
    :param T_step: A float, the internal stage length.
    :param N_horizon: An integer, the control horizon length ``N``.

    :returns: A float np.array of shape ``[num_actors, N_horizon]``, the activation
        deltas w.r.t. the current time (``np.inf`` where unused).
    """
    na = delta_toa.shape[0]
    control_sequence = np.full((na, N_horizon), np.inf)
    for p in range(depth):
        a = int(assigned[p])
        if a < 0:
            continue
        t_act = float(delta_toa[a, p] - t_lead)
        if not math.isfinite(t_act):
            continue
        ts = int(math.floor(t_act / T_step))
        if ts >= N_horizon:
            continue
        control_sequence[a, ts] = t_act
    return control_sequence


# ---------------------------------------------------------------------------
# Fast BFS
# ---------------------------------------------------------------------------

def search_activation_sequences_fast(
    *,
    delta_toa: np.ndarray,
    particle_class: np.ndarray,
    t_act: np.ndarray,
    undisturbed_mask: np.ndarray,
    valid_mask: np.ndarray,
    t_lead: float,
    T_step: float,
    t_cycle: float,
    N_R: int,
    N_horizon: int,
    max_branches,
    num_actors: int,
    perform_rollout,
) -> np.ndarray:
    """
    BFS with compact paths; reconstructs (+ rollout) only for leaves.

    Inputs match the pre-filtered masks of the legacy search (ejectable /
    ready particles already applied to ``delta_toa`` and the masks).

    :param delta_toa: A float np.array of shape ``[num_actors, num_particles]``,
        arrival-time deltas.
    :param particle_class: An integer np.array, class labels for particles (used by rollout).
    :param t_act: A float np.array of shape ``[num_actors]``, current actor internal times.
    :param undisturbed_mask: A Boolean np.array of shape ``[num_actors, num_particles]``,
        the partial condition-1 mask.
    :param valid_mask: A Boolean np.array of shape ``[num_actors, num_particles]``,
        the ready (condition-2) mask.
    :param t_lead: A float, the actor lead time.
    :param T_step: A float, the internal stage length.
    :param t_cycle: A float, the actor cycle time.
    :param N_R: An integer, the full-search depth (number of particles expanded fully).
    :param N_horizon: An integer, the control horizon length ``N``.
    :param max_branches: An integer or ``inf``, the max branches per particle.
    :param num_actors: An integer, the number of actors.
    :param perform_rollout: A callable completing unassigned particles at leaves.

    :returns: A float np.array of shape ``[num_sequences, num_actors, N_horizon]``,
        the candidate control sequences as activation deltas.
    """
    delta_toa = np.ascontiguousarray(delta_toa, dtype=np.float64)
    undisturbed_mask = np.ascontiguousarray(undisturbed_mask, dtype=np.bool_)
    valid_mask = np.ascontiguousarray(valid_mask, dtype=np.bool_)
    na, np_particles = delta_toa.shape

    if max_branches is None or (isinstance(max_branches, float) and math.isinf(max_branches)):
        max_br = na  # cannot select more distinct actors than exist
        always_offer_no_actor = True
    else:
        max_br = int(max_branches)
        always_offer_no_actor = False

    use_numba = tree_numba_enabled()
    # Scratch assigned buffer for Numba (prefix + padding)
    assigned_buf = np.full(np_particles, -1, dtype=np.int32)

    # Queue entries: (depth, assigned_tuple)
    queue: deque[Tuple[int, Tuple[int, ...]]] = deque()
    queue.append((0, ()))
    leaves: List[Tuple[int, Tuple[int, ...]]] = []

    while queue:
        depth, assigned = queue.popleft()

        if depth == np_particles:
            leaves.append((depth, assigned))
            continue

        if depth > N_R - 1:
            leaves.append((depth, assigned))
            continue

        if use_numba:
            assigned_buf[:] = -1
            for i, a in enumerate(assigned):
                assigned_buf[i] = a
            child_actors = select_child_actors_numba(
                depth,
                assigned_buf,
                delta_toa,
                valid_mask,
                undisturbed_mask,
                float(t_lead),
                float(T_step),
                float(t_cycle),
                max_br,
            )
            child_actors_list = child_actors.tolist()
        else:
            child_actors_list = _select_child_actors_py(
                depth,
                assigned,
                delta_toa,
                valid_mask,
                undisturbed_mask,
                float(t_lead),
                float(T_step),
                float(t_cycle),
                max_br,
            )

        for a in child_actors_list:
            queue.append((depth + 1, assigned + (int(a),)))

        # "No actor" does not consume max_branches. Legacy skips it only when the
        # actor loop already hit max_branches before reaching the sentinel.
        if always_offer_no_actor or len(child_actors_list) < max_br:
            queue.append((depth + 1, assigned + (-1,)))

    if not leaves:
        return [np.full((num_actors, 1), np.inf)]

    sequences = []
    for depth, assigned in leaves:
        seq = control_sequence_from_assignment(
            assigned, depth, delta_toa, t_lead, T_step, N_horizon
        )
        if depth < np_particles:
            seq = perform_rollout(
                control_sequence=seq,
                assigned_particle_idxs=np.array([], dtype=int),
                delta_toa=delta_toa[:, depth:],
                particle_class=np.ones(np_particles - depth),
                t_act=t_act,
            )
        sequences.append(seq)

    return np.asarray(sequences)
