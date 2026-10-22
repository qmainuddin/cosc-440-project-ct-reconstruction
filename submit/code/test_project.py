"""Assertion-based checks supplement (do not replace) the supplied print-out tests."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from project_main import (MultiScaleVoxelModel, ScaleAttentionModel, rays_to_points,
                          ray_attenuation, make_train_step, display_volume, volume_metrics)


class ProjectTests(unittest.TestCase):
    def test_ray_endpoints_and_dtype(self):
        rays = tf.constant([[1., 0., 0., -1., .1, .1]], dtype=tf.float64)
        points, t = rays_to_points(rays, 10, np.float64(.9), np.float64(1.1))
        self.assertEqual(points.dtype, tf.float64)
        self.assertEqual(points.shape, (1, 10, 3))
        np.testing.assert_allclose(points[0, 0], [.1, .09, .09])
        np.testing.assert_allclose(points[0, -1], [-.1, .11, .11])
        np.testing.assert_allclose(np.diff(t), .2 / 9)

    def test_integral_analytic_and_gradients(self):
        values = tf.Variable([[.5, .3, .1]], dtype=tf.float32)
        with tf.GradientTape() as tape:
            answer = ray_attenuation(values, [.9, 1., 1.1], [2.], .9, 1.1)
        np.testing.assert_allclose(answer, [.12], rtol=1e-5)
        np.testing.assert_allclose(tape.gradient(answer, values), [[.1, .2, .1]], rtol=1e-5)
        # Constant field with interior-only samples integrates across all [near,far].
        result = ray_attenuation(tf.constant([[3., 3.]]), [.25, .75], [2.], 0., 1.)
        np.testing.assert_allclose(result, [6.])

    def test_supplied_index_examples_and_boundaries(self):
        model = MultiScaleVoxelModel(2., 3, 1)
        values = tf.constant([[.5, .25, .125]])
        for depth, expected in enumerate([1, 6, 84, 584, 4368, 33824, 266304, 2113664]):
            self.assertEqual(int(model.get_flattened_position(values, depth)[0]), expected)
        boundary = model.get_flattened_position(tf.constant([[0., 0., 0.], [1., 1., 1.]]), 2)
        np.testing.assert_array_equal(boundary, [0, 511])

    def test_models_update_and_checkpoint_roundtrip(self):
        for cls in (MultiScaleVoxelModel, ScaleAttentionModel):
            with self.subTest(model=cls.__name__):
                tf.keras.utils.set_random_seed(8)
                model = cls(2., 2, 1)
                query = tf.constant([[.1, -.2, .3], [-.3, .2, -.1]])
                model(query)
                optimizer = tf.keras.optimizers.Adam(.001)
                optimizer.build(model.trainable_variables)
                before = [v.numpy().copy() for v in model.trainable_variables]
                step = make_train_step(model, optimizer, 4)
                rays = tf.constant([[0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 1., 0.]])
                mse, nmse = step(rays, tf.constant([.3, .5]), tf.constant(0.), tf.constant(.5))
                self.assertTrue(np.isfinite([mse.numpy(), nmse.numpy()]).all())
                self.assertTrue(any(not np.array_equal(a, b.numpy()) for a, b in zip(before, model.trainable_variables)))
                if cls is ScaleAttentionModel:
                    # Prove the new attention weights affect the optimized loss.
                    attention_ids = {id(v) for v in model.attention.trainable_variables}
                    self.assertTrue(any(id(v) in attention_ids and not np.array_equal(a, v.numpy())
                                        for a, v in zip(before, model.trainable_variables)))
                expected = model(query).numpy()
                with tempfile.TemporaryDirectory() as tmp:
                    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
                    path = checkpoint.save(str(Path(tmp) / 'ckpt'))
                    restored = cls(2., 2, 1)
                    restored(query)
                    opt2 = tf.keras.optimizers.Adam(.001)
                    opt2.build(restored.trainable_variables)
                    tf.train.Checkpoint(model=restored, optimizer=opt2).restore(path).assert_consumed()
                    np.testing.assert_allclose(restored(query), expected, atol=1e-7)
                    self.assertEqual(int(opt2.iterations.numpy()), 1)

    def test_zero_volume_and_identical_metrics(self):
        np.testing.assert_array_equal(display_volume(np.zeros((2, 12, 12))), 0)
        raw = np.linspace(0, 1, 2 * 12 * 12, dtype=np.float32).reshape(2, 12, 12)
        metrics = volume_metrics(raw, raw.transpose(1, 2, 0))
        self.assertAlmostEqual(metrics['ssim'], 1., places=5)
        self.assertEqual(metrics['display_mse'], 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
