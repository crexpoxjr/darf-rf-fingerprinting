from __future__ import annotations

import math
from typing import Any, Dict, List

import torch


def _candidate_fitness(model, genome: Dict[str, torch.Tensor], train_loader, device: torch.device, max_batches: int | None) -> float:
    model.set_genome(genome)
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

    return float(correct / total) if total > 0 else 0.0


def _clone_genome(genome: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in genome.items()}


def _mutate_genome(genome: Dict[str, torch.Tensor], mutation_std: float, generator: torch.Generator) -> Dict[str, torch.Tensor]:
    mutated = {}
    for key, value in genome.items():
        noise = torch.randn(value.shape, generator=generator, dtype=value.dtype) * mutation_std
        mutated[key] = value + noise
    return mutated


def _crossover_genome(genome_a: Dict[str, torch.Tensor], genome_b: Dict[str, torch.Tensor], generator: torch.Generator) -> Dict[str, torch.Tensor]:
    child = {}
    for key in genome_a:
        mix = torch.rand(genome_a[key].shape, generator=generator, dtype=genome_a[key].dtype)
        child[key] = torch.where(mix > 0.5, genome_a[key], genome_b[key])
    return child


def evolve_model(model, train_loader, device: torch.device, training_cfg: Dict[str, Any], seed: int) -> tuple[list[float], Dict[str, Any]]:
    evolution_cfg = training_cfg.get("neuroevolution", {})
    population_size = int(evolution_cfg.get("population_size", 18))
    generations = int(evolution_cfg.get("generations", 12))
    elite_fraction = float(evolution_cfg.get("elite_fraction", 0.25))
    mutation_std = float(evolution_cfg.get("mutation_std", 0.12))
    crossover_rate = float(evolution_cfg.get("crossover_rate", 0.35))
    eval_batches = evolution_cfg.get("eval_batches")
    eval_batches = int(eval_batches) if eval_batches is not None else None

    if population_size < 2:
        raise ValueError("neuroevolution.population_size must be at least 2")
    if generations < 1:
        raise ValueError("neuroevolution.generations must be at least 1")

    elite_count = max(1, int(math.ceil(population_size * elite_fraction)))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    base_genome = model.clone_genome()
    population: List[Dict[str, torch.Tensor]] = [_clone_genome(base_genome)]
    for _ in range(population_size - 1):
        population.append(_mutate_genome(base_genome, mutation_std, rng))

    best_history: list[float] = []
    avg_history: list[float] = []
    best_genome = _clone_genome(base_genome)
    best_fitness = float("-inf")

    for generation_idx in range(generations):
        scored_population = []
        for genome in population:
            fitness = _candidate_fitness(model, genome, train_loader, device, eval_batches)
            scored_population.append((fitness, genome))

        scored_population.sort(key=lambda item: item[0], reverse=True)
        generation_best = float(scored_population[0][0])
        generation_avg = float(sum(score for score, _ in scored_population) / len(scored_population))
        best_history.append(generation_best)
        avg_history.append(generation_avg)
        print(
            f"Generation {generation_idx + 1}/{generations} | "
            f"best_fitness={generation_best:.4f} | avg_fitness={generation_avg:.4f}"
        )

        if generation_best > best_fitness:
            best_fitness = generation_best
            best_genome = _clone_genome(scored_population[0][1])

        elites = [_clone_genome(genome) for _, genome in scored_population[:elite_count]]
        next_population: List[Dict[str, torch.Tensor]] = elites[:]
        while len(next_population) < population_size:
            parent_a = elites[int(torch.randint(len(elites), (1,), generator=rng).item())]
            if len(elites) > 1 and float(torch.rand(1, generator=rng).item()) < crossover_rate:
                parent_b = elites[int(torch.randint(len(elites), (1,), generator=rng).item())]
                child = _crossover_genome(parent_a, parent_b, rng)
            else:
                child = _clone_genome(parent_a)
            next_population.append(_mutate_genome(child, mutation_std, rng))

        population = next_population

    model.set_genome(best_genome)
    model.to(device)
    return best_history, {
        "algorithm": "hypernea_prototype_evolution",
        "population_size": population_size,
        "generations": generations,
        "elite_fraction": elite_fraction,
        "mutation_std": mutation_std,
        "crossover_rate": crossover_rate,
        "eval_batches": eval_batches,
        "best_fitness": best_fitness,
        "avg_fitness_history": avg_history,
    }