import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

root = zarr.open('./dataset/dp_train_data.zarr.zip')
print(root.tree())

root = zarr.open('./example_demo_session/dataset.zarr.zip')
print(root.tree())