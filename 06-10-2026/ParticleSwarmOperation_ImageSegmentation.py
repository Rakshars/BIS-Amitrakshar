from multiprocessing import Pool, cpu_count
import numpy as np
from PIL import Image

IMAGE = "input.jpg"
OUTPUT = "segmented.png"

PARTICLES = 20
ITERATIONS = 30
THRESHOLDS = 2


def fitness(t, hist):

    t = np.sort(t.astype(int))

    mean = np.average(
        np.arange(256),
        weights=hist
    )

    score = 0
    edges = [0] + list(t) + [256]

    for a, b in zip(edges[:-1], edges[1:]):

        h = hist[a:b]
        n = h.sum()

        if n > 0:
            m = np.average(
                np.arange(a, b),
                weights=h
            )

            score += n * (m - mean) ** 2

    return score


def pso(args):

    image, seed, particles = args

    rng = np.random.default_rng(seed)

    # FLOAT positions
    p = rng.uniform(
        1, 255,
        (particles, THRESHOLDS)
    )

    v = rng.uniform(
        -10, 10,
        (particles, THRESHOLDS)
    )

    hist = np.bincount(
        image.ravel(),
        minlength=256
    )

    best_p = p.copy()

    best_f = np.array([
        fitness(x, hist)
        for x in p
    ])

    for _ in range(ITERATIONS):

        for i in range(particles):

            r1 = rng.random(THRESHOLDS)
            r2 = rng.random(THRESHOLDS)

            v[i] = (
                0.7 * v[i]
                + 1.5 * r1 * (best_p[i] - p[i])
            )

            p[i] += v[i]

            p[i] = np.clip(
                p[i],
                1,
                254
            )

            f = fitness(p[i], hist)

            if f > best_f[i]:
                best_f[i] = f
                best_p[i] = p[i].copy()

    i = np.argmax(best_f)

    return best_f[i], best_p[i]


if __name__ == "__main__":

    image = np.array(
        Image.open(IMAGE).convert("L")
    )

    processes = min(
        cpu_count(),
        4
    )

    particles_per_process = (
        PARTICLES // processes
    )

    tasks = []

    for i in range(processes):
        tasks.append(
            (
                image,
                i,
                particles_per_process
            )
        )

    with Pool(processes) as pool:
        results = pool.map(
            pso,
            tasks
        )

    best = max(
        results,
        key=lambda x: x[0]
    )

    score, thresholds = best

    thresholds = np.sort(
        thresholds.astype(int)
    )

    # Image segmentation
    segmented = np.zeros_like(image)

    segmented[
        image > thresholds[0]
    ] = 127

    segmented[
        image > thresholds[1]
    ] = 255

    Image.fromarray(
        segmented
    ).save(OUTPUT)

    print()
    print("PSO Image Segmentation")
    print("----------------------")
    print("Processes :", processes)
    print("Particles :", PARTICLES)
    print("Thresholds:", thresholds)
    print("Output    :", OUTPUT)
