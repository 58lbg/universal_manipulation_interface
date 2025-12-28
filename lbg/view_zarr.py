import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

root = zarr.open('example_demo_session/dataset.zarr.zip')
print(root.tree())