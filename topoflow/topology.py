"""Crossing-word topology validator.

For a polyline from start to goal, record every crossing of an obstacle cut in
order along the path: letter +(j+1) when crossing cut j left->right, -(j+1)
when crossing right->left. Free reduction (cancelling adjacent x x^-1) gives
the reduced word.

Interpretation: fixing a reference path (here: the path passing ABOVE every
obstacle, whose word is empty), fixed-endpoint homotopy classes of paths in
the plane minus the obstacles correspond one-to-one with reduced words, i.e.
with elements of the free group F_m. The class set itself is a torsor over
F_m; the reduced word is its coordinate relative to that reference path.

Side convention for points exactly on the line x = cx: they count as "right"
(half-open). This makes crossing detection deterministic for degenerate inputs.
"""
from __future__ import annotations

import numpy as np

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def crossing_sequence(env, path):
    """Unreduced list of signed letters (ints, +-(j+1)) in path order."""
    path = np.asarray(path, dtype=float)
    seq = []
    for k in range(len(path) - 1):
        p, q = path[k], path[k + 1]
        events = []
        for j, o in enumerate(env.obstacles):
            cx = o.cx
            sp, sq = p[0] >= cx, q[0] >= cx
            if sp == sq:
                continue
            s = (cx - p[0]) / (q[0] - p[0])
            y = p[1] + s * (q[1] - p[1])
            if y < o.cut_top():
                events.append((s, (j + 1) if not sp else -(j + 1)))
        events.sort()
        seq.extend(e for _, e in events)
    return seq


def reduce_word(seq):
    """Free reduction with a stack."""
    stack = []
    for x in seq:
        if stack and stack[-1] == -x:
            stack.pop()
        else:
            stack.append(x)
    return tuple(stack)


def crossing_word(env, path):
    return reduce_word(crossing_sequence(env, path))


def word_to_str(word):
    """(1, -2, 3) -> 'aB c' style: lowercase = generator, uppercase = inverse. Empty -> 'e'."""
    if not word:
        return "e"
    return "".join(LETTERS[abs(x) - 1] if x > 0 else LETTERS[abs(x) - 1].upper() for x in word)


def str_to_word(s):
    if s == "e":
        return ()
    return tuple((LETTERS.index(c) + 1) if c.islower() else -(LETTERS.index(c.lower()) + 1) for c in s)


def winding_vector(word, m):
    """Abelianization: exponent sum per generator. Loses order information."""
    v = np.zeros(m, dtype=int)
    for x in word:
        v[abs(x) - 1] += 1 if x > 0 else -1
    return v


def expected_word_for_code(code):
    """Route code -> word for an x-monotone left-to-right path.

    code[j] = 1 means 'pass below obstacle j' (crosses cut j once, left->right),
    code[j] = 0 means 'pass above' (no crossing). Obstacles are indexed in cut
    x-order, so the letters appear in increasing order.
    """
    return tuple(j + 1 for j, b in enumerate(code) if b)


def code_to_str(code):
    return "".join("B" if b else "A" for b in code)  # A = above, B = below
