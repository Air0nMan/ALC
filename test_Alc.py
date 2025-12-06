#!/usr/bin/env python3
"""
Tests para las funciones de pseudo-inversa del TP2:

- pinvEcuacionesNormales(X, L, Y)
- pinvSVD(U, S, V, Y)
- pinvHouseHolder(Q, R, Y)
- pinvGramSchmidt(Q, R, Y)

La idea es testearlas con distintos tamaños y formas de X_t, Y_t:
- Caso 1: n > p (matriz "alta", rango completo por columnas)
- Caso 2: n < p (matriz "ancha", rango completo por filas)
- Caso 3: n = p (cuadrada e invertible)

Cada test:
- construye una W_true "real",
- genera Y_t = W_true @ X_t,
- calcula W con cada método de pseudo-inversa,
- compara contra una referencia W_ref = Y_t @ np.linalg.pinv(X_t),
- y chequea que Y_t ≈ W X_t para cada método.
"""

import numpy as np

# IMPORTAR TUS FUNCIONES DESDE alc.py (ajustá el nombre del módulo si hace falta)
from ALC import (
    rango,
    cholesky,
    traspuesta,
    QR_con_HH,
    QR_con_GS,
    svd_reducida,
    pinvSVD,
    pinvEcuacionesNormales,
    pinvGramSchmidt,
    pinvHouseHolder,
)

TOL = 1e-8


def _compute_all_W(X_t, Y_t, use_cholesky=True):
    """
    Calcula W con todos los métodos disponibles para un par (X_t, Y_t),
    reproduciendo la lógica de selección de Cholesky que mostraste.
    Devuelve un diccionario con las distintas W.
    """

    n, p = X_t.shape
    ranX = rango(X_t)

    # --- Cholesky (ecuaciones normales) ---
    L = None
    usar_normales = False

    if use_cholesky:
        if ranX == p and n > p:
            # Caso: rango completo por columnas, matriz alta
            # L, Lt = cholesky(X_t.T @ X_t) pero usando tu interfaz
            L, Lt = cholesky(traspuesta(X_t), X_t)
            usar_normales = True
        elif ranX == n:
            if n < p:
                # Caso: rango completo por filas, matriz ancha
                L, Lt = cholesky(X_t, traspuesta(X_t))
                usar_normales = True
            elif n == p:
                # Caso cuadrado e invertible
                L, Lt = cholesky(X_t)
                usar_normales = True

    # --- SVD reducida de X_t ---
    U, S, V = svd_reducida(X_t)

    # --- QR de X_t^T ---
    Qt_hh, Rt_hh = QR_con_HH(traspuesta(X_t))
    Qt_gs, Rt_gs = QR_con_GS(traspuesta(X_t))

    # --- Pseudo-inversas por distintos métodos ---
    W_svd = pinvSVD(U, S, V, Y_t)
    W_hh = pinvHouseHolder(Qt_hh, Rt_hh, Y_t)
    W_gs = pinvGramSchmidt(Qt_gs, Rt_gs, Y_t)

    if usar_normales:
        W_cholesky = pinvEcuacionesNormales(X_t, L, Y_t)
    else:
        W_cholesky = None

    return {
        "W_svd": W_svd,
        "W_cholesky": W_cholesky,
        "W_hh": W_hh,
        "W_gs": W_gs,
    }


def _check_case(name, X_t, Y_t, use_cholesky=True, tol=TOL):
    """
    Ejecuta un caso de prueba:
    - chequea que rango(X_t) sea el esperado (mín(n,p)),
    - calcula todas las W,
    - las compara contra la referencia numpy.linalg.pinv,
    - y verifica que Y_t ≈ W X_t.
    """
    print(f"\n===== {name} =====")
    n, p = X_t.shape

    # Chequear que tu rango coincide con el de numpy
    ranX_np = np.linalg.matrix_rank(X_t)
    ranX = rango(X_t)
    assert ranX == ranX_np, (
        f"[{name}] rango(X_t) != matrix_rank(X_t): "
        f"rango = {ranX}, numpy = {ranX_np}"
    )

    # Calcular todas las W con tus métodos
    W_dict = _compute_all_W(X_t, Y_t, use_cholesky=use_cholesky)

    # Referencia usando numpy (solo para tests, no en el TP final)
    W_ref = Y_t @ np.linalg.pinv(X_t)

    # Comparar cada W con la referencia y con Y_t
    for metodo, W in W_dict.items():
        if W is None:
            # Puede pasar con Cholesky si no corresponde usar ecuaciones normales
            print(f"[{name}] {metodo}: SKIP (no aplica este caso)")
            continue

        # W ≈ W_ref
        assert np.allclose(W, W_ref, atol=tol), (
            f"[{name}] {metodo}: W no coincide con la referencia."
        )

        # Y_t ≈ W X_t
        Y_rec = W @ X_t
        assert np.allclose(
            Y_t, Y_rec, atol=tol
        ), f"[{name}] {metodo}: Y != W X (reconstrucción mala)."

        print(f"[{name}] {metodo}: OK")

    print(f"===== {name}: TODOS LOS MÉTODOS OK =====")


# ---------------------------------------------------------------------------
# CASOS DE TEST
# ---------------------------------------------------------------------------

def test_caso_1_n_mayor_p():
    """
    Caso 1: n > p, rango(X) = p (matriz alta, columnas independientes).
    X_t ∈ R^{4x3}, Y_t ∈ R^{2x3}
    """
    X_t = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 3.0],
        ]
    )

    # W_true ∈ R^{2x4}
    W_true = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, -1.0, 1.0, 2.0],
        ]
    )

    Y_t = W_true @ X_t  # Y_t ∈ R^{2x3}

    _check_case("CASO 1 (n > p, rango = p)", X_t, Y_t, use_cholesky=True)


def test_caso_2_n_menor_p():
    """
    Caso 2: n < p, rango(X) = n (matriz ancha, filas independientes).
    X_t ∈ R^{3x5}, Y_t ∈ R^{2x5}
    """
    X_t = np.array(
        [
            [1.0, 0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 1.0, 0.0, 4.0],
            [1.0, 1.0, 0.0, 1.0, 1.0],
        ]
    )

    # W_true ∈ R^{2x3}
    W_true = np.array(
        [
            [2.0, -1.0, 0.0],
            [1.0, 3.0, 4.0],
        ]
    )

    Y_t = W_true @ X_t  # Y_t ∈ R^{2x5}

    _check_case("CASO 2 (n < p, rango = n)", X_t, Y_t, use_cholesky=True)


def test_caso_3_n_igual_p_cuadrada_invertible():
    """
    Caso 3: n = p, X_t cuadrada e invertible.
    X_t ∈ R^{3x3}, Y_t ∈ R^{2x3}
    """
    X_t = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [5.0, 6.0, 0.0],
        ]
    )

    # W_true ∈ R^{2x3}
    W_true = np.array(
        [
            [1.0, 0.0, -1.0],
            [2.0, 1.0, 3.0],
        ]
    )

    Y_t = W_true @ X_t  # Y_t ∈ R^{2x3}

    _check_case("CASO 3 (n = p, invertible)", X_t, Y_t, use_cholesky=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ejecutar todos los tests "a mano" si corrés: python test_pinv.py
    test_caso_1_n_mayor_p()
    test_caso_2_n_menor_p()
    test_caso_3_n_igual_p_cuadrada_invertible()
    print("\n*** TODOS LOS TESTS TERMINARON SIN ERRORES ***")
