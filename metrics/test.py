import unittest
import jax.numpy as jnp

from metrics import squared_error, mean_squared_error


class MetricsTest(unittest.TestCase):
    """
    各メトリクスのテスト
    """

    def test_squared_error(self):
        """
        squared_errorのテスト
        """
        predict = jnp.array(
            [1, 2, 3]
        )
        actual = jnp.array(
            [1, 2, 3]
        )

        res = squared_error(
            actual=actual, predict=predict)

        self.assertEqual(res.all(), jnp.array([0, 0, 0]).all())

        predict = jnp.array(
            [1, 2, 3]
        )
        actual = jnp.array(
            [1, 0, 3]
        )

        res = squared_error(
            actual=actual, predict=predict)

        self.assertEqual(res.all(), jnp.array([0, 4, 0]).all())

    def test_mean_squared_error(self):
        """
        mean_squared_errorのテスト
        """
        predict = jnp.array(
            [1, 2, 3]
        )
        actual = jnp.array(
            [1, 2, 3]
        )

        res = mean_squared_error(
            actual=actual, predict=predict)

        self.assertEqual(res, 0)

        predict = jnp.array(
            [1, 2, 3]
        )
        actual = jnp.array(
            [1, 2, 0]
        )

        res = mean_squared_error(
            actual=actual, predict=predict)

        self.assertEqual(res, 3)


if __name__ == "__main__":
    unittest.main()
