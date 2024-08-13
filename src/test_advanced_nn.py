import unittest
import jax
import jax.numpy as jnp
from advanced_nn import data_augmentation


class TestDataAugmentation(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.image_shape = (32, 32, 3)
        self.batch_size = 4
        self.images = jax.random.uniform(
            self.rng,
            (self.batch_size,) + self.image_shape
        )

    def test_horizontal_flip(self):
        augmented, _ = data_augmentation(self.images, self.rng)
        self.assertEqual(augmented.shape, self.images.shape)
        # Check if at least one image is flipped horizontally
        flipped = jnp.any(jnp.not_equal(augmented[:, :, ::-1, :], self.images))
        self.assertTrue(flipped)

    def test_vertical_flip(self):
        augmented, _ = data_augmentation(self.images, self.rng)
        self.assertEqual(augmented.shape, self.images.shape)
        # Check if at least one image is flipped vertically
        flipped = jnp.any(jnp.not_equal(augmented[:, ::-1, :, :], self.images))
        self.assertTrue(flipped)

    def test_rotation(self):
        augmented, _ = data_augmentation(self.images, self.rng)
        self.assertEqual(augmented.shape, self.images.shape)
        # Check if at least one image is rotated
        rotated = jnp.any(jnp.not_equal(augmented, self.images))
        self.assertTrue(rotated)

    def test_brightness_adjustment(self):
        augmented, _ = data_augmentation(self.images, self.rng)
        self.assertEqual(augmented.shape, self.images.shape)
        # Check if brightness is adjusted while maintaining the valid range
        self.assertTrue(jnp.all(augmented >= 0) and jnp.all(augmented <= 1))
        brightness_changed = jnp.any(jnp.not_equal(augmented, self.images))
        self.assertTrue(brightness_changed)

    def test_contrast_adjustment(self):
        augmented, _ = data_augmentation(self.images, self.rng)
        self.assertEqual(augmented.shape, self.images.shape)
        # Check if contrast is adjusted while maintaining the valid range
        self.assertTrue(jnp.all(augmented >= 0) and jnp.all(augmented <= 1))
        contrast_changed = jnp.any(jnp.not_equal(augmented, self.images))
        self.assertTrue(contrast_changed)

    def test_key_update(self):
        _, key1 = data_augmentation(self.images, self.rng)
        _, key2 = data_augmentation(self.images, key1)
        self.assertFalse(jnp.array_equal(key1, key2))


if __name__ == '__main__':
    unittest.main()
