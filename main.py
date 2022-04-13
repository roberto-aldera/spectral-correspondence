# Script for spectral correspondence toy examples
import pdb
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

plt.rc('text', usetex=False)
plt.rc('font', family='serif')


def make_random_points(n=16):
    # P_mat is a point cloud with random coordinates, Q_mat is its shifted counterpart
    primary_points = 10 * np.random.rand(2, n)
    secondary_points = np.array((primary_points[0] + 2, primary_points[1] + 3))
    return primary_points, secondary_points


def make_tunnel_points():
    primary_points = np.transpose(
        np.array(([1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5])))
    secondary_points = np.array((primary_points[0] + 2, primary_points[1] + 3))
    return primary_points, secondary_points


def make_square_points():
    primary_points = np.transpose(
        np.array(([4, 0], [1, 0], [1, 1], [0, 1])))
    secondary_points = np.array((primary_points[0] + 2, primary_points[1] + 3))
    return primary_points, secondary_points


def get_principle_eigenvector_from_matches(primary_points, secondary_points):
    # List all possible correspondences
    L = []
    n = primary_points.shape[1]
    for i in range(n):
        for j in range(n):
            L.append([i, j])
    L = np.array(L)
    W = L.shape[0]
    # np.random.shuffle(L)

    C = W * np.eye(W)  # Compatibility matrix, with diagonal set to number of possible matches

    # Fill in remaining cells to compute pairwise compatibility:
    # Select points from P, each point is compared to each other point to compute distance between them
    # Also select points from Q, each point is compared to each other point to compute distance between them
    # Determine score based on distance, if d1 and d2 share a similar value, this is probably a likely match and so
    # score is high. If the value of d1 and d2 are not similar, score will be low (unlikely match)
    # Pairs are between scans, but distances are calculated on pairs of points within each scan - so (a1, a2) and
    # (b1, b2) will compare distance between a1 and b1 with a2 and b2
    # If a1 is b1, or a2 is b2, score = 0 as we can't have a point assigned to more than one other point (exception for
    # a1 is b1 AND a2 is b2)

    for idx_i in range(W):
        # Make first pair from two points, one from each scan
        a1 = L[idx_i, 0]  # point ID from set 1 -> P
        a2 = L[idx_i, 1]  # point ID from set 2 -> Q
        C[idx_i, idx_i] = 1

        for idx_j in range(idx_i + 1, W):
            # Make second pair from two points, one from each scan
            b1 = L[idx_j, 0]
            b2 = L[idx_j, 1]

            if (a1 == b1) or (a2 == b2):
                C[idx_i, idx_j] = 0  # Set score of points with same ID to 0
            else:
                # Get points from set 1 and distance between them
                p_i = primary_points[:, a1]
                p_j = primary_points[:, b1]
                d1 = np.matmul(np.transpose((p_i - p_j)), (p_i - p_j))
                # Get points from set 2 and distance between them
                q_i = secondary_points[:, a2]
                q_j = secondary_points[:, b2]
                d2 = np.matmul(np.transpose((q_i - q_j)), (q_i - q_j))
                # assign score based on difference between distances
                C[idx_i, idx_j] = 1 / (1 + np.abs(d1 - d2))
            C[idx_j, idx_i] = C[idx_i, idx_j]  # lower triangle = upper triangle

    # Find principle eigenvector
    w, v = np.linalg.eig(C)
    eigenvalues = np.diag(w)
    max_idx = np.argmax(eigenvalues)
    u1 = v[:, max_idx]
    u1 = u1 / np.linalg.norm(u1)

    # to handle cases where eigenvector values are negative due to limitation of the eig method, this happens if
    # initial vector prior to iteration is negative
    if np.any(u1 < 0):
        u1 *= -1

    return u1, L


def recover_matches(u1, L):
    # Recover matches from randomly permuted L using principle eigenvector
    W = L.shape[0]
    iteration = 0
    do_iterate = True
    searched = []
    unsearched = np.ones(W)
    M = []  # Use M for the final matches

    while do_iterate:
        print("Iteration:", iteration)
        max_match_k = 0
        max_reward_k = 0

        for i in range(W):
            for j in range(W):
                if unsearched[i] and unsearched[j]:
                    if np.square(u1[i]) >= np.square(u1[j]):
                        max_match = i
                        max_reward = np.square(u1[i])

                        if max_reward > max_reward_k:
                            max_reward_k = max_reward
                            max_match_k = max_match
        # Add the match to M
        M.append(L[max_match_k])

        for i in range(W):
            if (L[max_match_k, 0] == L[i, 0]) or (L[max_match_k, 1] == L[i, 1]):
                if i not in searched:
                    searched.append(i)

        for i in range(len(searched)):
            unsearched[searched[i]] = False

        iteration += 1

        if not np.any(unsearched) or (iteration > 50):
            do_iterate = False

    return M


def do_plotting(primary_points, secondary_points, principle_eigenvector, fig_path):
    # Plot initial points
    plt.figure(figsize=(5, 5))
    plt.plot(primary_points[0], primary_points[1], '^')
    plt.plot(secondary_points[0], secondary_points[1], '^')
    for i in range(primary_points.shape[1]):
        plt.plot([primary_points[0, i], secondary_points[0, i]], [primary_points[1, i], secondary_points[1, i]], 'k-')
    plt.grid()
    # plt.xlim([-5, 10])
    # plt.ylim([-5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Matched points")
    plt.savefig("%s%s" % (fig_path, "landmarks.png"))

    # Plot sorted values of normalised principle eigenvector in descending order
    plt.figure(figsize=(5, 5))
    plt.plot(principle_eigenvector, 'r.', label="Unsorted")
    plt.plot(np.sort(principle_eigenvector)[::-1], '.', label="Sorted")
    # plt.ylim([0, 0.5])
    plt.legend()
    plt.grid()
    plt.title("Eigenvector")
    plt.savefig("%s%s" % (fig_path, "eigenvector.png"))


def build_big_figure(results_path):
    # A perfect square
    primary_points = np.transpose(
        np.array(([1, 0], [1, 1], [0, 1], [0, 0])))
    secondary_points = np.array((primary_points[0] + 0.1, primary_points[1] + 0.3))

    principle_eigenvector, L = get_principle_eigenvector_from_matches(primary_points, secondary_points)
    fig, ax = plt.subplots(4, 2, figsize=(10, 15))
    ax[0, 0].grid()
    ax[0, 0].plot(primary_points[0], primary_points[1], '^', label="Primary")
    ax[0, 0].plot(secondary_points[0], secondary_points[1], '^', label="Secondary")
    for i in range(primary_points.shape[1]):
        ax[0, 0].plot([primary_points[0, i], secondary_points[0, i]], [primary_points[1, i], secondary_points[1, i]],
                      'k--')
    ax[0, 0].axis('equal')
    ax[0, 0].set_xlim([-1, 3])
    ax[0, 0].set_ylim([-1, 3])
    ax[0, 0].legend()
    ax[0, 0].set_title("Corresponding points: full symmetry")

    ax[0, 1].grid()
    ax[0, 1].set_ylim([0, 0.4])
    ax[0, 1].plot(principle_eigenvector, 'o', color="tab:red", label="Unsorted")
    ax[0, 1].plot(np.sort(principle_eigenvector)[::-1], '*', color="tab:blue", label="Sorted")
    ax[0, 1].legend(loc="lower right")
    ax[0, 1].set_title("Principle eigenvector elements")

    # Add some asymmetry
    primary_points = np.transpose(
        np.array(([1.25, 0], [1, 1], [0, 1], [0, 0])))
    secondary_points = np.array((primary_points[0] + 0.1, primary_points[1] + 0.3))

    principle_eigenvector, L = get_principle_eigenvector_from_matches(primary_points, secondary_points)
    ax[1, 0].grid()
    ax[1, 0].plot(primary_points[0], primary_points[1], '^', label="Primary")
    ax[1, 0].plot(secondary_points[0], secondary_points[1], '^', label="Secondary")
    for i in range(primary_points.shape[1]):
        ax[1, 0].plot([primary_points[0, i], secondary_points[0, i]], [primary_points[1, i], secondary_points[1, i]],
                      'k--')
    ax[1, 0].axis('equal')
    ax[1, 0].set_xlim([-1, 3])
    ax[1, 0].set_ylim([-1, 3])
    ax[1, 0].legend()
    ax[1, 0].set_title("Corresponding points: slight asymmetry for one point")

    ax[1, 1].grid()
    ax[1, 1].set_ylim([0, 0.4])
    ax[1, 1].plot(principle_eigenvector, 'o', color="tab:red", label="Unsorted")
    ax[1, 1].plot(np.sort(principle_eigenvector)[::-1], '*', color="tab:blue", label="Sorted")
    ax[1, 1].legend(loc="lower right")
    ax[1, 1].set_title("Principle eigenvector elements")

    # Add a little more asymmetry
    primary_points = np.transpose(
        np.array(([2, 0], [1, 1], [0, 1], [0, 0])))
    secondary_points = np.array((primary_points[0] + 0.1, primary_points[1] + 0.3))

    principle_eigenvector, L = get_principle_eigenvector_from_matches(primary_points, secondary_points)
    ax[2, 0].grid()
    ax[2, 0].plot(primary_points[0], primary_points[1], '^', label="Primary")
    ax[2, 0].plot(secondary_points[0], secondary_points[1], '^', label="Secondary")
    for i in range(primary_points.shape[1]):
        ax[2, 0].plot([primary_points[0, i], secondary_points[0, i]], [primary_points[1, i], secondary_points[1, i]],
                      'k--')
    ax[2, 0].axis('equal')
    ax[2, 0].set_xlim([-1, 3])
    ax[2, 0].set_ylim([-1, 3])
    ax[2, 0].legend()
    ax[2, 0].set_title("Corresponding points: large asymmetry for one point")

    ax[2, 1].grid()
    ax[2, 1].set_ylim([0, 0.4])
    ax[2, 1].plot(principle_eigenvector, 'o', color="tab:red", label="Unsorted")
    ax[2, 1].plot(np.sort(principle_eigenvector)[::-1], '*', color="tab:blue", label="Sorted")
    ax[2, 1].legend(loc="lower right")
    ax[2, 1].set_title("Principle eigenvector elements")

    # Add some very easy/asymmetrical data so matches are not ambiguous
    # primary_points = np.transpose(
    #     np.array(([2, 0], [-0.25, -0.25], [0.5, 1.5], [0, 0])))
    primary_points = np.transpose(
        np.array(([0, 0], [0, 2], [0.8, 2], [1.1, 0.5])))
    secondary_points = np.array((primary_points[0] + 0.1, primary_points[1] + 0.3))

    principle_eigenvector, L = get_principle_eigenvector_from_matches(primary_points, secondary_points)
    ax[3, 0].grid()
    ax[3, 0].plot(primary_points[0], primary_points[1], '^', label="Primary")
    ax[3, 0].plot(secondary_points[0], secondary_points[1], '^', label="Secondary")
    for i in range(primary_points.shape[1]):
        ax[3, 0].plot([primary_points[0, i], secondary_points[0, i]], [primary_points[1, i], secondary_points[1, i]],
                      'k--')
    ax[3, 0].axis('equal')
    ax[3, 0].set_xlim([-1, 3])
    ax[3, 0].set_ylim([-1, 3])
    ax[3, 0].legend()
    ax[3, 0].set_title("Corresponding points: no symmetry")

    ax[3, 1].grid()
    ax[3, 1].set_ylim([0, 0.4])
    ax[3, 1].plot(principle_eigenvector, 'o', color="tab:red", label="Unsorted")
    ax[3, 1].plot(np.sort(principle_eigenvector)[::-1], '*', color="tab:blue", label="Sorted")
    ax[3, 1].legend(loc="lower right")
    ax[3, 1].set_title("Principle eigenvector elements")

    # fig.suptitle("Eigenvector")
    fig.tight_layout()
    fig.savefig("%s%s" % (results_path, "big-fig.pdf"))
    plt.close()


def main():
    fig_path = "/Users/roberto/data/spectral-correspondence/"
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    results_path = fig_path + current_time + "/"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    # results_path = None  # just to stop folders being created while debugging
    print("Results will be saved to:", results_path)

    # first_point_set, second_point_set = make_random_points(n=4)
    first_point_set, second_point_set = make_square_points()
    principle_eigenvector, L = get_principle_eigenvector_from_matches(first_point_set, second_point_set)
    final_matches = recover_matches(principle_eigenvector, L)
    print("All matches:", L)
    print("Final matches:", final_matches)
    do_plotting(first_point_set, second_point_set, principle_eigenvector, results_path)

    # first_point_set, second_point_set = make_square_points()
    # principle_eigenvector, L = get_principle_eigenvector_from_matches(first_point_set, second_point_set)
    # do_plotting(first_point_set, second_point_set, principle_eigenvector, results_path)
    build_big_figure(results_path)


if __name__ == '__main__':
    main()
