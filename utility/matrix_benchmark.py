"""
matrix_benchmark.py

Бенчмарк умножения квадратных матриц тремя способами:

1) Классический тройной цикл (O(n^3)) — базовый эталон.

2) Чистый Python: алгоритм Штрассена
   (7 рекурсивных умножений вместо 8,
   ~18 блочных сложений/вычитаний на уровень рекурсии).

3) Чистый Python: вариант Штрассена–Винограда
   (та же асимптотика, но меньше блочных сложений/вычитаний —
   порядка 15 вместо ~18 на каждый уровень рекурсии).

Скрипт:
- генерирует матрицы A, B размерности N с элементами в [-9, 9];
- измеряет только время одной операции умножения (без генерации/проверок);
- считает статистики по NUM_REPEATS прогонкам;
- проверяет корректность Штрассена и Штрассена–Винограда,
  сравнивая результат с классическим O(n^3) алгоритмом (один раз, вне тайминга).
"""

from __future__ import annotations

import gc
import random
import statistics
import time
from typing import List, Tuple, Callable

# =======================
# Константы эксперимента
# =======================

# Размерность матриц
N: int = 384

# Если True — на каждом повторе генерируются новые матрицы (случайные, но детерминированные через RANDOM_SEED).
# Если False — генерируются один раз и переиспользуются во всех повторах.
IS_RANDOM: bool = False

# Сколько раз повторять измерение (усреднение)
NUM_REPEATS: int = 1

# Базовый сид для генерации
RANDOM_SEED: int = 42

# Пороги переключения на обычный тройной цикл внутри рекурсивных алгоритмов
CUTOFF_STRASSEN: int = 64
CUTOFF_WINOGRAD: int = 64


# =======================
# Вспомогательные функции
# =======================

Matrix = List[List[int]]


def generate_matrix(n: int, seed: int) -> Matrix:
    """Сгенерировать n×n матрицу с целыми элементами в диапазоне [-9, 9]."""
    rng = random.Random(seed)
    return [[rng.randint(-9, 9) for _ in range(n)] for _ in range(n)]


def matmul_classic(A: Matrix, B: Matrix) -> Matrix:
    """
    Классический тройной цикл (ikj-порядок) — базовый O(n^3) алгоритм.

    Используется:
    - как эталон для проверки корректности,
    - как отдельный алгоритм для сравнения по времени.
    """
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        rowA = A[i]
        Ci = C[i]
        for k in range(n):
            aik = rowA[k]
            Bk = B[k]
            for j in range(n):
                Ci[j] += aik * Bk[j]
    return C


def add_matrix(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def sub_matrix(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]


# =======================
# Алгоритм Штрассена
# =======================

def _strassen_core(A: Matrix, B: Matrix, cutoff: int) -> Matrix:
    n = len(A)
    if n <= cutoff:
        return matmul_classic(A, B)

    m = n // 2

    a11 = [row[:m] for row in A[:m]]
    a12 = [row[m:] for row in A[:m]]
    a21 = [row[:m] for row in A[m:]]
    a22 = [row[m:] for row in A[m:]]

    b11 = [row[:m] for row in B[:m]]
    b12 = [row[m:] for row in B[:m]]
    b21 = [row[:m] for row in B[m:]]
    b22 = [row[m:] for row in B[m:]]

    # 7 рекурсивных умножений (M1..M7)
    M1 = _strassen_core(add_matrix(a11, a22), add_matrix(b11, b22), cutoff)
    M2 = _strassen_core(add_matrix(a21, a22), b11, cutoff)
    M3 = _strassen_core(a11, sub_matrix(b12, b22), cutoff)
    M4 = _strassen_core(a22, sub_matrix(b21, b11), cutoff)
    M5 = _strassen_core(add_matrix(a11, a12), b22, cutoff)
    M6 = _strassen_core(sub_matrix(a21, a11), add_matrix(b11, b12), cutoff)
    M7 = _strassen_core(sub_matrix(a12, a22), add_matrix(b21, b22), cutoff)

    # Комбинация в блоки C11..C22 (классические формулы Штрассена)
    c11 = sub_matrix(add_matrix(add_matrix(M1, M4), M7), M5)
    c12 = add_matrix(M3, M5)
    c21 = add_matrix(M2, M4)
    c22 = add_matrix(sub_matrix(add_matrix(M1, M3), M2), M6)

    # Склейка блоков
    C = [c11[i] + c12[i] for i in range(m)] + [c21[i] + c22[i] for i in range(m)]
    return C


def matmul_strassen(A: Matrix, B: Matrix, cutoff: int = CUTOFF_STRASSEN) -> Matrix:
    """Умножение матриц методом Штрассена, поддерживает произвольное n (через доп. нули)."""
    n = len(A)
    if n == 0:
        return []
    assert n == len(B) == len(A[0]) == len(B[0]), "Матрицы должны быть квадратными и одинакового размера"

    # Если n не степень двойки — дополняем нулями до ближайшей степени двойки
    if n & (n - 1):
        m = 1 << (n - 1).bit_length()
        Ap = [row + [0] * (m - n) for row in A] + [[0] * m for _ in range(m - n)]
        Bp = [row + [0] * (m - n) for row in B] + [[0] * m for _ in range(m - n)]
        Cp = _strassen_core(Ap, Bp, cutoff)
        return [row[:n] for row in Cp[:n]]
    else:
        return _strassen_core(A, B, cutoff)


# =======================
# Алгоритм Штрассена–Винограда
# (формулы с S1..S10, P1..P7)
# =======================

def _strassen_winograd_core(A: Matrix, B: Matrix, cutoff: int) -> Matrix:
    n = len(A)
    if n <= cutoff:
        return matmul_classic(A, B)

    m = n // 2

    a11 = [row[:m] for row in A[:m]]
    a12 = [row[m:] for row in A[:m]]
    a21 = [row[:m] for row in A[m:]]
    a22 = [row[m:] for row in A[m:]]

    b11 = [row[:m] for row in B[:m]]
    b12 = [row[m:] for row in B[:m]]
    b21 = [row[:m] for row in B[m:]]
    b22 = [row[m:] for row in B[m:]]

    # Промежуточные суммы/разности S1..S10 (Виноград)
    S1 = sub_matrix(b12, b22)
    S2 = add_matrix(a11, a12)
    S3 = add_matrix(a21, a22)
    S4 = sub_matrix(b21, b11)
    S5 = add_matrix(a11, a22)
    S6 = add_matrix(b11, b22)
    S7 = sub_matrix(a12, a22)
    S8 = add_matrix(b21, b22)
    S9 = sub_matrix(a11, a21)
    S10 = add_matrix(b11, b12)

    # 7 рекурсивных умножений P1..P7
    P1 = _strassen_winograd_core(a11, S1, cutoff)
    P2 = _strassen_winograd_core(S2, b22, cutoff)
    P3 = _strassen_winograd_core(S3, b11, cutoff)
    P4 = _strassen_winograd_core(a22, S4, cutoff)
    P5 = _strassen_winograd_core(S5, S6, cutoff)
    P6 = _strassen_winograd_core(S7, S8, cutoff)
    P7 = _strassen_winograd_core(S9, S10, cutoff)

    # Комбинация:
    # C11 = P5 + P4 − P2 + P6
    # C12 = P1 + P2
    # C21 = P3 + P4
    # C22 = P5 + P1 − P3 − P7
    C11 = add_matrix(sub_matrix(add_matrix(P5, P4), P2), P6)
    C12 = add_matrix(P1, P2)
    C21 = add_matrix(P3, P4)
    C22 = sub_matrix(sub_matrix(add_matrix(P5, P1), P3), P7)

    C = [C11[i] + C12[i] for i in range(m)] + [C21[i] + C22[i] for i in range(m)]
    return C


def matmul_strassen_winograd(A: Matrix, B: Matrix, cutoff: int = CUTOFF_WINOGRAD) -> Matrix:
    """Умножение матриц методом Штрассена–Винограда, поддерживает произвольное n (через доп. нули)."""
    n = len(A)
    if n == 0:
        return []
    assert n == len(B) == len(A[0]) == len(B[0]), "Матрицы должны быть квадратными и одинакового размера"

    if n & (n - 1):
        m = 1 << (n - 1).bit_length()
        Ap = [row + [0] * (m - n) for row in A] + [[0] * m for _ in range(m - n)]
        Bp = [row + [0] * (m - n) for row in B] + [[0] * m for _ in range(m - n)]
        Cp = _strassen_winograd_core(Ap, Bp, cutoff)
        return [row[:n] for row in Cp[:n]]
    else:
        return _strassen_winograd_core(A, B, cutoff)


# =======================
# Бенчмарк
# =======================

def time_algo(
    fn: Callable[[Matrix, Matrix], Matrix],
    matrices: Tuple[Tuple[Matrix, Matrix], ...]
) -> list[float]:
    """Замерить время работы fn на наборе матриц. Замеряем только fn, без генерации данных."""
    times: list[float] = []
    for A, B in matrices:
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        _ = fn(A, B)
        t1 = time.perf_counter()
        gc.enable()
        times.append(t1 - t0)
    return times


def run_benchmark() -> None:
    n = N
    print("=" * 60)
    print(f"Benchmark: n = {n}, repeats = {NUM_REPEATS}, is_random = {IS_RANDOM}")
    print(f"cutoff Strassen = {CUTOFF_STRASSEN}, cutoff Winograd = {CUTOFF_WINOGRAD}")
    print("=" * 60)

    # --- Генерация матриц для всех повторов ---
    if IS_RANDOM:
        matrices_list = []
        for r in range(NUM_REPEATS):
            seed_a = RANDOM_SEED + r
            seed_b = RANDOM_SEED + 10_000 + r
            A = generate_matrix(n, seed_a)
            B = generate_matrix(n, seed_b)
            matrices_list.append((A, B))
    else:
        A = generate_matrix(n, RANDOM_SEED)
        B = generate_matrix(n, RANDOM_SEED + 10_000)
        matrices_list = [(A, B)] * NUM_REPEATS

    matrices_list = tuple(matrices_list)

    # --- Корректность: сверяем Страссен и Виноград с классикой (на одном наборе) ---
    A0_list, B0_list = matrices_list[0]

    C_ref = matmul_classic(A0_list, B0_list)
    C_strassen = matmul_strassen(A0_list, B0_list, cutoff=CUTOFF_STRASSEN)
    C_winograd = matmul_strassen_winograd(A0_list, B0_list, cutoff=CUTOFF_WINOGRAD)

    if C_strassen != C_ref:
        raise AssertionError("Ошибка: результат Штрассена не совпадает с классическим O(n^3)")
    if C_winograd != C_ref:
        raise AssertionError("Ошибка: результат Штрассена–Винограда не совпадает с классическим O(n^3)")

    print("Проверка корректности: OK (Strassen и Strassen–Winograd совпадают с классическим алгоритмом)")

    # --- Замеры времени ---
    times_classic = time_algo(matmul_classic, matrices_list)
    times_str = time_algo(lambda X, Y: matmul_strassen(X, Y, cutoff=CUTOFF_STRASSEN), matrices_list)
    times_win = time_algo(lambda X, Y: matmul_strassen_winograd(X, Y, cutoff=CUTOFF_WINOGRAD), matrices_list)

    def report(name: str, times: list[float]) -> None:
        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        sd = statistics.pstdev(times) if len(times) > 1 else 0.0
        print(f"{name:18s}: avg={avg:.6f}s  min={mn:.6f}s  max={mx:.6f}s  std={sd:.6f}s")

    print("\nРезультаты (чистое время умножения матриц):")
    report("Classic O(n^3)", times_classic)
    report("Strassen", times_str)
    report("Strassen-Winograd", times_win)
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
