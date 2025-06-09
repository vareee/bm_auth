import random
import hashlib
import numpy as np


PRIME = 2**31 - 1

def vector_to_x(vs: list[float]) -> int:
    vs = np.array(vs)
    binary_pattern = ''.join(['1' if v > 0 else '0' for v in vs])
    vec_bytes = bytes([int(binary_pattern[i:i+8], 2) for i in range(0, len(binary_pattern), 8)])
    return int.from_bytes(hashlib.sha256(vec_bytes).digest(), "big") % PRIME

def serialize_coeffs(coeffs):
    return b''.join(c.to_bytes(8, 'big') for c in coeffs)

def eval_poly(coeffs, x):
    y = 0
    for i, c in enumerate(coeffs):
        y = (y + c * pow(x, i, PRIME)) % PRIME
    return y

def lagrange_interpolate(points, p=PRIME):
    seen = set()
    unique_points = []
    for point in points:
        if isinstance(point, list):
            point = tuple(point)
        if point[0] not in seen:
            seen.add(point[0])
            unique_points.append(tuple(point))
    points = unique_points
    n = len(points)

    if n < 1:
        raise ValueError("Not enough unique points")

    coeffs = [0] * n

    for i in range(n):
        xi, yi = points[i]
        li = [1]
        denom = 1
        for j in range(n):
            if i == j:
                continue
            xj, _ = points[j]

            temp = [0] * (len(li) + 1)
            for k in range(len(li)):
                temp[k+1] = li[k]
            for k in range(len(li)):
                temp[k] = (temp[k] - li[k] * xj) % p
            li = temp
            denom = (denom * (xi - xj)) % p

        try:
            inv_denom = pow(denom, -1, p)
        except ValueError:
            raise ValueError(f"Denominator {denom} is not invertible modulo {p}")

        for k in range(n):
            li[k] = (li[k] * yi * inv_denom) % p

        for k in range(n):
            coeffs[k] = (coeffs[k] + li[k]) % p

    return coeffs

def deterministic_secret_from_biometric(encoding):
    N = min(128, len(encoding))
    vs = np.array(encoding[:N])
    binary_pattern = ''.join(['1' if v > 0 else '0' for v in vs])
    seed_data = bytes([int(binary_pattern[i:i+8], 2) for i in range(0, len(binary_pattern), 8)])
    seed_hash = hashlib.sha256(seed_data).digest()
    random.seed(seed_hash)
    return [random.randint(0, PRIME - 1) for _ in range(33)]  # degree=32

def extract_biometric_points(encoding: list[float], coeffs, point_count=10) -> list[tuple[int, int]]:
    encoding_len = len(encoding)
    chunk_size = encoding_len // point_count

    if chunk_size < 4:
        raise ValueError("Too many points requested for the given vector size")

    points = []
    for i in range(point_count):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = encoding[start:end]
        x = vector_to_x(chunk)
        y = eval_poly(coeffs, x)
        points.append((x, y))

    return points

def create_vault_from_coeffs(coeffs, biometric_data: list[float], chaff_count=100, point_count=68):
    genuine_points = extract_biometric_points(biometric_data, coeffs, point_count=point_count)

    chaff = set()
    while len(chaff) < chaff_count:
        x = random.randint(0, PRIME - 1)
        y = random.randint(0, PRIME - 1)
        if any(p[0] == x for p in genuine_points):
            continue
        chaff.add((x, y))

    full_vault = genuine_points + list(chaff)
    random.shuffle(full_vault)

    serialized = serialize_coeffs(coeffs)
    return {
        "points": full_vault,
        "hash": hashlib.sha256(serialized).hexdigest()
    }

def unlock_vault(vault, biometric_data: list[float], degree=32, trials=100, point_count=10, top_k=30):
    candidate_points = []
    encoding_len = len(biometric_data)
    chunk_size = encoding_len // point_count

    for i in range(point_count):
        start = i * chunk_size
        end = start + chunk_size
        if end > encoding_len or start >= encoding_len:
            break
        chunk = biometric_data[start:end]
        x_candidate = vector_to_x(chunk)
        candidate_points.append(x_candidate)

    vault_points = sorted(
        vault.get("points", []),
        key=lambda p: min(abs(p[0] - xc) for xc in candidate_points)
    )[:top_k]

    from itertools import combinations
    
    count = 0
    for subset in combinations(vault_points, degree + 1):
        try:
            count += 1
            if count > trials:
                return False

            coeffs = lagrange_interpolate(list(subset), PRIME)
            serialized = serialize_coeffs(coeffs)
            candidate_hash = hashlib.sha256(serialized).hexdigest()
            expected_hash = vault.get("hash")

            if candidate_hash == expected_hash:
                print("Authentication successful!")
                return True
        except Exception as e:
            continue

    print("Authentication failed after all trials")
    return False
