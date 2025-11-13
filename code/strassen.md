# Алгоритм Штрассена (квадратные матрицы)

Алгоритм Штрассена рекурсивно разбивает квадратные матрицы $n \times n$
на четыре блока $n/2 \times n/2$ и вычисляет произведение за счёт
семи блочных умножений (вместо восьми) и дополнительных сложений.
Теоретическая трудоёмкость в квадратном случае — $O(n^{\log_2 7})$.

Эта реализация:

- принимает две квадратные матрицы одного размера;
- при необходимости дополняет их нулями до ближайшей сверху степени двойки;
- рекурсивно применяет схему Штрассена с базой $n = 1$;
- в конце обрезает результат до исходного размера.

```python
def mat_add(A, B):
    n = len(A)
    m = len(A[0])
    return [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]


def mat_sub(A, B):
    n = len(A)
    m = len(A[0])
    return [[A[i][j] - B[i][j] for j in range(m)] for i in range(n)]


def pad_square(A, size):
    n = len(A)
    m = len(A[0])
    R = [row + [0] * (size - m) for row in A]
    for _ in range(size - n):
        R.append([0] * size)
    return R


def split_quadrants(A):
    n = len(A)
    k = n // 2
    a11 = [row[:k] for row in A[:k]]
    a12 = [row[k:] for row in A[:k]]
    a21 = [row[:k] for row in A[k:]]
    a22 = [row[k:] for row in A[k:]]
    return a11, a12, a21, a22


def join_quadrants(c11, c12, c21, c22):
    k = len(c11)
    R = [c11[i] + c12[i] for i in range(k)]
    R.extend(c21[i] + c22[i] for i in range(k))
    return R


def next_power_of_two(n):
    m = 1
    while m < n:
        m <<= 1
    return m


def strassen_recursive(A, B):
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    a11, a12, a21, a22 = split_quadrants(A)
    b11, b12, b21, b22 = split_quadrants(B)
    m1 = strassen_recursive(mat_add(a11, a22), mat_add(b11, b22))
    m2 = strassen_recursive(mat_add(a21, a22), b11)
    m3 = strassen_recursive(a11, mat_sub(b12, b22))
    m4 = strassen_recursive(a22, mat_sub(b21, b11))
    m5 = strassen_recursive(mat_add(a11, a12), b22)
    m6 = strassen_recursive(mat_sub(a21, a11), mat_add(b11, b12))
    m7 = strassen_recursive(mat_sub(a12, a22), mat_add(b21, b22))
    c11 = mat_sub(mat_add(mat_add(m1, m4), m7), m5)
    c12 = mat_add(m3, m5)
    c21 = mat_add(m2, m4)
    c22 = mat_add(mat_sub(mat_add(m1, m3), m2), m6)
    return join_quadrants(c11, c12, c21, c22)


def matmul_strassen(A, B):
    n = len(A)
    m = next_power_of_two(n)
    if m == n:
        return strassen_recursive(A, B)
    Ap = pad_square(A, m)
    Bp = pad_square(B, m)
    Cp = strassen_recursive(Ap, Bp)
    return [row[:n] for row in Cp[:n]]
