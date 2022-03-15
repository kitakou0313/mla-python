import jax.numpy as jnp
import numpy as np


def squared_error(actual: jnp.ndarray, predict: jnp.ndarray) -> jnp.ndarray:
    """
    各配列の値の二乗誤差を計算
    """
    return (actual - predict) ** 2


def mean_squared_error(actual: jnp.ndarray, predict: jnp.ndarray) -> jnp.float64:
    """
    平均二乗誤差を返却
    """
    return jnp.mean(squared_error(actual=actual, predict=predict))


if __name__ == "__main__":
    pass
