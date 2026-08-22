"""Symbolic regression: discover compact mathematical expressions for precursors."""

from __future__ import annotations

import random
import numpy as np
from typing import Optional, Callable


class SymbolicSearch:
    """Search for symbolic expressions that predict degradation."""

    def __init__(self, feature_names: list[str], max_depth: int = 4,
                 population_size: int = 200, generations: int = 50):
        self.feature_names = feature_names
        self.max_depth = max_depth
        self.pop_size = population_size
        self.generations = generations
        self.rng = random.Random(42)

    def search(self, X: np.ndarray, y: np.ndarray) -> list[dict]:
        """Run genetic programming to find expressions."""
        # Initialize population with random expressions
        population = [self._random_expr() for _ in range(self.pop_size)]

        best_expr = None
        best_score = float("inf")
        history = []

        for gen in range(self.generations):
            # Evaluate all expressions
            scores = []
            for expr in population:
                try:
                    pred = self._evaluate(expr, X)
                    if pred is None or np.any(np.isnan(pred)):
                        scores.append(float("inf"))
                    else:
                        mse = np.mean((pred - y) ** 2)
                        complexity = self._complexity(expr)
                        # Penalize complexity
                        score = mse + 0.001 * complexity
                        scores.append(score)
                except:
                    scores.append(float("inf"))

            # Find best
            best_idx = np.argmin(scores)
            if scores[best_idx] < best_score:
                best_score = scores[best_idx]
                best_expr = population[best_idx]

            history.append({
                "generation": gen,
                "best_score": best_score,
                "avg_score": np.mean([s for s in scores if s < float("inf")]),
                "best_expr": str(best_expr),
            })

            # Selection + crossover + mutation
            new_pop = []
            # Elitism
            sorted_idx = np.argsort(scores)
            for i in range(self.pop_size // 10):
                new_pop.append(population[sorted_idx[i]])

            while len(new_pop) < self.pop_size:
                if self.rng.random() < 0.7:
                    p1 = self._tournament_select(population, scores)
                    p2 = self._tournament_select(population, scores)
                    child = self._crossover(p1, p2)
                else:
                    child = self._tournament_select(population, scores)
                child = self._mutate(child)
                new_pop.append(child)

            population = new_pop

        return [{"expr": best_expr, "score": best_score, "history": history}]

    def _random_expr(self, depth: int = 0) -> dict:
        if depth >= self.max_depth or (depth > 0 and self.rng.random() < 0.3):
            return {"type": "feature", "index": self.rng.randint(0, len(self.feature_names) - 1)}
        op = self.rng.choice(["add", "sub", "mul", "div", "sqrt", "abs", "feature"])
        if op == "feature":
            return {"type": "feature", "index": self.rng.randint(0, len(self.feature_names) - 1)}
        elif op in ("sqrt", "abs"):
            return {"type": op, "child": self._random_expr(depth + 1)}
        else:
            return {"type": op, "left": self._random_expr(depth + 1),
                    "right": self._random_expr(depth + 1)}

    def _evaluate(self, expr: dict, X: np.ndarray) -> Optional[np.ndarray]:
        if expr["type"] == "feature":
            idx = expr["index"]
            if idx < X.shape[1]:
                return X[:, idx].astype(float)
            return np.zeros(X.shape[0])
        elif expr["type"] == "add":
            l = self._evaluate(expr["left"], X)
            r = self._evaluate(expr["right"], X)
            return l + r if l is not None and r is not None else None
        elif expr["type"] == "sub":
            l = self._evaluate(expr["left"], X)
            r = self._evaluate(expr["right"], X)
            return l - r if l is not None and r is not None else None
        elif expr["type"] == "mul":
            l = self._evaluate(expr["left"], X)
            r = self._evaluate(expr["right"], X)
            return l * r if l is not None and r is not None else None
        elif expr["type"] == "div":
            l = self._evaluate(expr["left"], X)
            r = self._evaluate(expr["right"], X)
            if r is None or l is None:
                return None
            return l / (r + 1e-10)
        elif expr["type"] == "sqrt":
            c = self._evaluate(expr["child"], X)
            return np.sqrt(np.abs(c)) if c is not None else None
        elif expr["type"] == "abs":
            c = self._evaluate(expr["child"], X)
            return np.abs(c) if c is not None else None
        return None

    def _complexity(self, expr: dict) -> int:
        if expr["type"] == "feature":
            return 1
        elif expr["type"] in ("sqrt", "abs"):
            return 1 + self._complexity(expr["child"])
        else:
            return 1 + self._complexity(expr["left"]) + self._complexity(expr["right"])

    def _tournament_select(self, pop, scores, k=3):
        indices = self.rng.sample(range(len(pop)), min(k, len(pop)))
        best = min(indices, key=lambda i: scores[i])
        return pop[best]

    def _crossover(self, p1, p2) -> dict:
        if self.rng.random() < 0.5:
            return p1
        return p2

    def _mutate(self, expr: dict) -> dict:
        if self.rng.random() > 0.3:
            return expr
        return self._random_expr(depth=0)

    def format_expr(self, expr: dict) -> str:
        if expr["type"] == "feature":
            idx = expr["index"]
            return self.feature_names[idx] if idx < len(self.feature_names) else f"x{idx}"
        elif expr["type"] in ("sqrt", "abs"):
            return f"{expr['type']}({self.format_expr(expr['child'])})"
        else:
            return f"({self.format_expr(expr['left'])} {expr['type']} {self.format_expr(expr['right'])})"
