import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from tqdm.auto import tqdm
from PIL import Image
import os

device = 'cuda'
seed = 3
# Load the SD pipeline and add a hook
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to(device)
pipe.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
pipe.scheduler.set_timesteps(30)

def hook_fn(module, input, output):
    module.output = output
    
pipe.unet.mid_block.register_forward_hook(hook_fn)

def display_sample(sample, i):
    image_pil = sample_to_PIL(sample, i)
    # display(f"Image at step {i}")
    # display(image_pil)


def reference_direction(img, reference_img, scale=1):
    """Return the direction from img to reference_img normed to length 1. (Both images are in latent space)"""

    res = reference_img - img
    res = F.normalize(res, dim=(0, 1, 2, 3)) * scale
    return res


def sample(prompt, guidance_loss_scale, guidance_scale=10,
         negative_prompt = "zoomed in, blurry, oversaturated, warped",
         num_inference_steps=30, start_latents = None,
         early_stop = 20, cfg_norm=True, cfg_decay=True, reference_latents=[]):

    image_row = []
    # If no starting point is passed, create one
    if start_latents is None:
        start_latents = torch.randn((1, 4, 64, 64), device=device)

    pipe.scheduler.set_timesteps(num_inference_steps)

    # Encode the prompt
    text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt)

    # Create our random starting point
    latents = start_latents.clone()
    latents *= pipe.scheduler.init_noise_sigma

    # Prepare the scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Loop through the sampling timesteps
    for i, t in tqdm(enumerate(pipe.scheduler.timesteps)):

        if i > early_stop: guidance_loss_scale = 0 # Early stop (optional)

        sigma = pipe.scheduler.sigmas[i]

        # Set requires grad
        if guidance_loss_scale != 0: latents = latents.detach().requires_grad_()

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)

        # Apply any scaling required by the scheduler
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual with the unet
        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform CFG
        cfg_scale = guidance_scale
        if cfg_decay: cfg_scale = 1 + guidance_scale * (1-i/num_inference_steps)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
        # Normalize (see https://enzokro.dev/blog/posts/2022-11-15-guidance-expts-1/)
        if cfg_norm:
            noise_pred = noise_pred * (torch.linalg.norm(noise_pred_uncond) / torch.linalg.norm(noise_pred))

        if guidance_loss_scale != 0:
            if len(reference_latents) > 1:  # Row of images
                cond_grad = reference_direction(latents, reference_latents[i], scale=guidance_loss_scale)
            elif len(reference_latents) == 1:  # Only one image, otherwise no reference image
                cond_grad = reference_direction(latents, reference_latents, scale=guidance_loss_scale)
            # Modify the latents based on this gradient
            latents = latents.detach() + cond_grad  #  * sigma**2

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        image_row.append(latents)
        with torch.no_grad():
            img = pipe.decode_latents(latents.detach())
            #display(pipe.numpy_to_pil(img)[0])


    # Decode the resulting latents into an image
    with torch.no_grad():
        image = pipe.decode_latents(latents.detach())

    return pipe.numpy_to_pil(image)[0], image, latents, image_row

def save_ref_img(seed, pil_list, max_save=10):
    path = f"ref_images/seed_{seed}/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    for cnt, img in enumerate(pil_list):
        if cnt < max_save:
            pil = img
            pil.save(path + f"step_{cnt+1}.png")
    pil = pil_list[-1]
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


def sample_to_PIL(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = Image.fromarray(image_processed[0])
    return image_pil


def save_ref_images(prompt, guidance_scale=10,
         negative_prompt = "zoomed in, blurry, oversaturated, warped",
         num_inference_steps=30, start_latents = None,
         early_stop = 20, cfg_norm=True, cfg_decay=True, reference_latents=[], seed=1):

    torch.manual_seed(seed)
    image_row = []
    pil_row = []
    # If no starting point is passed, create one
    if start_latents is None:
      start_latents = torch.randn((1, 4, 64, 64), device=device)

    pipe.scheduler.set_timesteps(num_inference_steps)

    # Encode the prompt
    text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt)

    # Create our random starting point
    latents = start_latents.clone()
    latents *= pipe.scheduler.init_noise_sigma

    # Prepare the scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Loop through the sampling timesteps
    for i, t in tqdm(enumerate(pipe.scheduler.timesteps)):

        sigma = pipe.scheduler.sigmas[i]

        # Set requires grad
        latents = latents.detach().requires_grad_()

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)

        # Apply any scaling required by the scheduler
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual with the unet
        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform CFG
        cfg_scale = guidance_scale
        if cfg_decay: cfg_scale = 1 + guidance_scale * (1-i/num_inference_steps)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
        # Normalize (see https://enzokro.dev/blog/posts/2022-11-15-guidance-expts-1/)
        if cfg_norm:
          noise_pred = noise_pred * (torch.linalg.norm(noise_pred_uncond) / torch.linalg.norm(noise_pred))

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        image_row.append(latents)
        with torch.no_grad():
          img = pipe.decode_latents(latents.detach())
          pil_row.append(pipe.numpy_to_pil(img)[0])
          # print(type(pipe.numpy_to_pil(img)[0]))
          # display(pipe.numpy_to_pil(img)[0])
          print()


    # Decode the resulting latents into an image
    with torch.no_grad():
        image = pipe.decode_latents(latents.detach())

    return pipe.numpy_to_pil(image)[0], image, latents, image_row, pil_row


def gen_with_ref_image(prompt, guidance_loss_scale, seed, ref_seed, guidance_scale=10,
         negative_prompt = "zoomed in, blurry, oversaturated, warped",
         num_inference_steps=30, start_latents = None,
         early_stop = 20, cfg_norm=True, cfg_decay=True, reference_latents=[]):

    torch.manual_seed(seed)
    # If no starting point is passed, create one
    if start_latents is None:
      start_latents = torch.randn((1, 4, 64, 64), device=device)

    pipe.scheduler.set_timesteps(num_inference_steps)

    # Encode the prompt
    text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt)

    # Create our random starting point
    latents = start_latents.clone()
    latents *= pipe.scheduler.init_noise_sigma

    # Prepare the scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Loop through the sampling timesteps
    for i, t in tqdm(enumerate(pipe.scheduler.timesteps)):

        if i > early_stop: guidance_loss_scale = 0 # Early stop (optional)

        sigma = pipe.scheduler.sigmas[i]

        # Set requires grad
        if guidance_loss_scale != 0: latents = latents.detach().requires_grad_()

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)

        # Apply any scaling required by the scheduler
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual with the unet
        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform CFG
        cfg_scale = guidance_scale
        if cfg_decay: cfg_scale = 1 + guidance_scale * (1-i/num_inference_steps)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
        # Normalize (see https://enzokro.dev/blog/posts/2022-11-15-guidance-expts-1/)
        if cfg_norm:
          noise_pred = noise_pred * (torch.linalg.norm(noise_pred_uncond) / torch.linalg.norm(noise_pred))

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        if guidance_loss_scale != 0:
            ref_img = torch.load(f"ref_images/seed_{ref_seed}/step_{i+1}.pt")
            cond_grad = reference_direction(latents, ref_img, scale=guidance_loss_scale)
            # elif len(reference_latents) == 1:  # Only one image, otherwise no reference image
            #   cond_grad = reference_direction(latents, reference_latents, scale=guidance_loss_scale)
            # Modify the latents based on this gradient
            latents = latents.detach() + cond_grad  #  * sigma**2



    # Decode the resulting latents into an image
    with torch.no_grad():
        image = pipe.decode_latents(latents.detach())

    return pipe.numpy_to_pil(image)[0]

def gen_400_imgs():
    """Generate the 400 referenced images."""

    guidance_loss_scale=200
    early_stop=1

    for seed in range(20):
        for ref_seed in range(20):
            img = gen_with_ref_image(prompt="Photograph of a cat", guidance_loss_scale=guidance_loss_scale, early_stop=early_stop, seed=seed, ref_seed=ref_seed)

            folder_path = f"res_images_addRef{early_stop}_refStepSize{guidance_loss_scale}/"
            isExist = os.path.exists(folder_path)
            if not isExist:
                os.makedirs(folder_path)
                img.save(folder_path + f"seed_{seed}_referredTo_{ref_seed}.png")


