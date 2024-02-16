from diffusers import DiffusionPipeline, UNet2DModel, DDPMScheduler, AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import torch
import tqdm
import numpy as np
from PIL import Image
import torch.nn.functional as F
import os
import shutil


repo_id = "google/ddpm-cat-256"
model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)
scheduler = DDPMScheduler.from_pretrained(repo_id)
seed = 3


def sample_to_PIL(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = Image.fromarray(image_processed[0])
    return image_pil

def display_sample(sample, i):
    """For jupyter notebook."""
    image_pil = sample_to_PIL(sample, i)
    # display(f"Image at step {i}")
    # display(image_pil)

def reference_direct(img, ref_img):
  """Calculate the direction vector from img to ref_img normalized to length 1."""

  res = ref_img - img
  res = F.normalize(res, dim=(0, 1, 2, 3))
  return res

def update_img(img, vect, l=10):
  """Update the image by going in the direction of vect.
  l indicates how far in the direction of vect should be went."""

  new_img = img + l * vect
  return new_img

def save_ref_img(seed, pil_list, max_save=10):
  path = f"ref_images/seed_{seed}/"
  isExist = os.path.exists(path)
  if not isExist:
    os.makedirs(path)

  for cnt, img in enumerate(pil_list):
    if cnt < max_save:
      pil = sample_to_PIL(img, cnt+1)
      pil.save(path + f"step_{cnt+1}.png")
  pil = sample_to_PIL(pil_list[-1], len(pil_list))
  pil.save(path + f"final.png")

def save_ref_vect(seed, pil_list, max_save=10):
  path = f"ref_images/seed_{seed}/"
  isExist = os.path.exists(path)
  if not isExist:
    os.makedirs(path)

  for cnt, img in enumerate(pil_list):
    if cnt < max_save:
      torch.save(img, path + f"step_{cnt+1}.pt")
  torch.save(pil_list[-1], path + f"final.pt")


def save_ref_imgs_many(seed, timesteps=100):
  torch.manual_seed(seed)
  scheduler.set_timesteps(timesteps)

  noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
  model.to("cuda")
  noisy_sample = noisy_sample.to("cuda")
  sample = noisy_sample
  pil_list = []

  for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
      # 1. predict noise residual
      with torch.no_grad():
          residual = model(sample, t).sample

      # 2. compute less noisy image and set x_t -> x_t-1
      sample = scheduler.step(residual, t, sample).prev_sample

      # 3. optionally look at image
      # if (i + 1) % 5 == 0:
      #     display_sample(sample, i + 1)
      pil_list.append(sample)
  # display_sample(sample, i)
  save_ref_img(seed, pil_list)
  save_ref_vect(seed, pil_list)
