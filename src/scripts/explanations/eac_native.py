"""EAC, "Explain Any Concept" (Sun et al., NeurIPS 2023), re-implemented on this repository's masked predictor
(TO_DO T3b). The original code (github.com/Jerry00917/samshap) is under an academic-only licence, so it is not
copied here; this module follows the published method and the released code's settings.

Pipeline, per image and explained class:
1. Concepts: one per label of the compact SAM map ("EAC-sharedmasks"), so the only difference from
   BPT+SAM_c and SAM_c+KernelSHAP is the estimator.
2. Per-Input Equivalence (PIE) surrogate, trained on `budget` real model queries: random concept subsets
   z ~ Bernoulli(0.5)^n, each evaluated with the shared masked predictor (same backgrounds as every other
   method). EAC's surrogate is a linear layer on z followed by the classifier's frozen head and softmax. Our
   runners treat the model as a black box (detectors, CLIP), so the head is replaced by a small trained MLP:
   Linear(n, hidden) -> tanh -> Linear(hidden, 1), fitted by mean squared error. This is the one deliberate
   deviation, and it is recorded in `info`.
3. Concept values on the surrogate, which costs no model queries. estimator='eac' (default) reproduces the
   released code: for each concept, the mean over Bernoulli(0.5) subsets of the surrogate's change when the
   concept is added, which is the Banzhaf value. estimator='shapley' uses sampled permutations instead.

Per-pixel convention: a concept's value is given to every pixel of the concept, as in
`utils_xai.kernel_shap_partition_regions` (no division by area).
"""
import numpy as np


def concept_masks(z, labels, concepts):
    """Boolean image masks for rows of concept indicators z (one column per concept)."""
    index = np.searchsorted(concepts, labels.ravel())
    return z[:, index].reshape((len(z), *labels.shape)).astype(bool)


def query(predict, target, z, labels, concepts, batch_size):
    scores = [np.asarray(predict(concept_masks(z[start:start + batch_size], labels, concepts)))[:, target]
              for start in range(0, len(z), batch_size)]
    return np.concatenate(scores).astype(float)


def fit_surrogate(z, y, hidden=64, epochs=300, lr=1e-2, seed=0, device='cpu'):
    """Small MLP from concept indicators to the model score; returns (predict function, R^2 on its training set)."""
    import torch
    torch.manual_seed(seed)
    x = torch.as_tensor(z, dtype=torch.float32, device=device)
    mean, scale = float(y.mean()), float(y.std()) or 1.0
    target = torch.as_tensor((y - mean) / scale, dtype=torch.float32, device=device)[:, None]
    net = torch.nn.Sequential(torch.nn.Linear(z.shape[1], hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, 1)).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(net(x), target)
        loss.backward()
        optimizer.step()
    net.eval()

    def surrogate(rows):
        with torch.no_grad():
            out = net(torch.as_tensor(rows, dtype=torch.float32, device=device))[:, 0].cpu().numpy()
        return out * scale + mean

    fitted = surrogate(z)
    r2 = 1 - float(((fitted - y) ** 2).sum()) / max(float(((y - y.mean()) ** 2).sum()), 1e-12)
    return surrogate, r2


def concept_values(surrogate, n, samples, estimator='eac', seed=0, chunk=200_000):
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    if estimator == 'eac':
        z = rng.binomial(1, .5, size=(samples, n)).astype(np.float32)  # Shared across concepts
        for i in range(n):
            with_i, without_i = z.copy(), z.copy()
            with_i[:, i], without_i[:, i] = 1, 0
            for start in range(0, samples, chunk):
                values[i] += (surrogate(with_i[start:start + chunk]) - surrogate(without_i[start:start + chunk])).sum()
        return values / samples
    if estimator == 'shapley':
        permutations = max(1, samples // n)
        for _ in range(permutations):
            order = rng.permutation(n)
            rows = np.zeros((n + 1, n), dtype=np.float32)
            for step, concept in enumerate(order, 1):
                rows[step] = rows[step - 1]
                rows[step, concept] = 1
            scores = surrogate(rows)
            values[order] += np.diff(scores)
        return values / permutations
    raise ValueError("estimator must be 'eac' or 'shapley'")


def explain(predict, labels, target, budget, mc_samples=2000, estimator='eac', batch_size=64, seed=0,
            hidden=64, epochs=300, device='cpu'):
    """EAC-sharedmasks attribution map (H x W) and info."""
    labels = np.asarray(labels)
    concepts = np.unique(labels)
    n = len(concepts)
    if n < 2:
        raise ValueError(f'EAC needs at least 2 concepts, got {n}')
    rng = np.random.default_rng(seed)
    z = rng.binomial(1, .5, size=(budget, n)).astype(np.float32)
    y = query(predict, target, z, labels, concepts, batch_size)  # The only real model queries
    surrogate, r2 = fit_surrogate(z, y, hidden=hidden, epochs=epochs, seed=seed, device=device)
    values = concept_values(surrogate, n, mc_samples, estimator, seed)
    heatmap = values[np.searchsorted(concepts, labels)]
    return heatmap, {'realised_evals': int(budget), 'concepts': int(n), 'surrogate_r2': r2,
                     'surrogate': f'MLP(n, {hidden}) tanh, MSE, Adam, {epochs} epochs (PIE head replaced; black-box model)',
                     'estimator': estimator, 'surrogate_samples_per_concept': int(mc_samples),
                     'concept_values': values.tolist(), 'concept_labels': concepts.tolist(),
                     'per_pixel': 'concept value on every concept pixel (no area division)'}
