"""Train IDGuardian perturbations for one image or an image directory.

This implementation follows Algorithm 1 in the IDGuardian supplementary
material.  The surrogate is IP-Adapter SDXL Plus Face.  The optimized image
is updated with two signals:

* CLIP image-embedding and FaceNet cosine identity losses; and
* the score difference between IP-Adapter conditions from the clean and
  perturbed images.

The script deliberately performs no generation or evaluation.  It produces
protected images that can be used by any downstream personalization pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from diffusers import DDIMScheduler, StableDiffusionXLPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from idguardian.ip_adapter import IPAdapterPlusXL  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate IDGuardian-protected images with PGD."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input image or directory of images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/idguardian"),
        help="Directory for protected images and results.jsonl.",
    )
    parser.add_argument(
        "--pretrained-model-name-or-path",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="SDXL base model identifier or local path.",
    )
    parser.add_argument(
        "--image-encoder-path",
        default="checkpoints/ip_adapter/models/image_encoder",
        help="IP-Adapter CLIP image encoder directory or model identifier.",
    )
    parser.add_argument(
        "--ip-adapter-path",
        default="checkpoints/ip_adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin",
        help="Local IP-Adapter Plus Face SDXL checkpoint.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        choices=(224, 512),
        default=224,
        help="Square optimization resolution used in the paper experiments.",
    )
    parser.add_argument(
        "--prompt",
        default="A photo of a person",
        help="Text condition used by the surrogate model.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=8.0 / 255.0,
        help="L-infinity perturbation budget. Default: 8/255.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.005,
        help="PGD step size. Default: 0.005.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of optimization steps. Default: 200.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example cuda or cuda:0. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Model precision. auto uses float16 on CUDA and float32 otherwise.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32
    return torch.float16 if device.type == "cuda" else torch.float32


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_images(input_path: Path, max_images: int | None) -> Iterable[Path]:
    if input_path.is_file():
        paths = [input_path]
    elif input_path.is_dir():
        paths = sorted(
            p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not paths:
        raise FileNotFoundError(f"No supported images found under: {input_path}")
    return paths if max_images is None else paths[:max_images]


def load_image(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device)


def save_tensor_image(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (image.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255.0).round()
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def psnr(clean: torch.Tensor, protected: torch.Tensor) -> float:
    mse = F.mse_loss(clean.float(), protected.float()).item()
    return float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def facenet_features(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Return normalized FaceNet features with the facenet-pytorch input range."""
    image = F.interpolate(image, size=(160, 160), mode="bilinear", align_corners=False)
    image = image * 2.0 - 1.0
    return F.normalize(model(image), dim=-1)


def encode_latents(
    vae: torch.nn.Module,
    image: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    with torch.no_grad():
        latent = vae.encode(image.to(dtype=dtype)).latent_dist.mode()
        return latent * vae.config.scaling_factor


def build_text_condition(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        encoded = pipe.encode_prompt(
            prompt=[prompt],
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    prompt_embeds, _, pooled_prompt_embeds, _ = encoded
    time_ids = torch.tensor(
        [[image_size, image_size, 0, 0, image_size, image_size]],
        device=device,
        dtype=pooled_prompt_embeds.dtype,
    )
    return prompt_embeds, pooled_prompt_embeds, time_ids


def unet_prediction(
    pipe: StableDiffusionXLPipeline,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    text_embeds: torch.Tensor,
    pooled_text_embeds: torch.Tensor,
    time_ids: torch.Tensor,
    image_prompt_embeds: torch.Tensor,
) -> torch.Tensor:
    encoder_hidden_states = torch.cat([text_embeds, image_prompt_embeds], dim=1)
    return pipe.unet(
        latents.to(dtype=pipe.unet.dtype),
        timesteps,
        encoder_hidden_states=encoder_hidden_states.to(dtype=pipe.unet.dtype),
        added_cond_kwargs={
            "text_embeds": pooled_text_embeds.to(dtype=pipe.unet.dtype),
            "time_ids": time_ids.to(dtype=pipe.unet.dtype),
        },
    ).sample


def pgd_protect(
    image: torch.Tensor,
    pipe: StableDiffusionXLPipeline,
    ip_model: IPAdapterPlusXL,
    facenet: torch.nn.Module,
    scheduler: DDIMScheduler,
    text_embeds: torch.Tensor,
    pooled_text_embeds: torch.Tensor,
    time_ids: torch.Tensor,
    device: torch.device,
    model_dtype: torch.dtype,
    epsilon: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    """Run Algorithm 1 from the supplementary material."""
    # Algorithm 1 initializes delta to zero.
    protected = image.detach().clone()

    with torch.no_grad():
        clean_latents = encode_latents(pipe.vae, image, model_dtype)
        clean_ip_tokens = ip_model.get_image_embeds_from_tensor(image).detach()
        clean_clip_features = ip_model.get_identity_vector_from_tensor(image).detach()
        clean_facenet_features = facenet_features(facenet, image).detach()

    alpha_cumprod = scheduler.alphas_cumprod.to(device=device)

    for _ in range(steps):
        protected = protected.detach().requires_grad_(True)

        adv_ip_tokens = ip_model.get_image_embeds_from_tensor(protected)
        adv_clip_features = ip_model.get_identity_vector_from_tensor(protected)
        adv_facenet_features = facenet_features(facenet, protected)

        identity_loss = (
            F.cosine_similarity(clean_clip_features, adv_clip_features, dim=-1).mean()
            + F.cosine_similarity(clean_facenet_features, adv_facenet_features, dim=-1).mean()
        )
        identity_loss.backward()
        identity_grad = protected.grad.detach()

        # Score-based adversarial conceptual bridge.  Both predictions use the
        # same clean noisy latent, while only the IP-Adapter identity tokens
        # differ, exactly as in Eq. (9) of the paper.
        with torch.no_grad():
            noise = torch.randn_like(clean_latents)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (clean_latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = scheduler.add_noise(clean_latents, noise, timesteps)
            adv_prediction = unet_prediction(
                pipe,
                noisy_latents,
                timesteps,
                text_embeds,
                pooled_text_embeds,
                time_ids,
                adv_ip_tokens.detach(),
            )
            clean_prediction = unet_prediction(
                pipe,
                noisy_latents,
                timesteps,
                text_embeds,
                pooled_text_embeds,
                time_ids,
                clean_ip_tokens,
            )
            sigma = (1.0 - alpha_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
            score_star = -(adv_prediction - clean_prediction) / sigma
            score_up = F.interpolate(
                score_star[:, :3],
                size=protected.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        identity_grad = identity_grad / (identity_grad.norm(p=2) + 1e-8)
        score_up = score_up / (score_up.norm(p=2) + 1e-8)
        total_grad = identity_grad - score_up

        with torch.no_grad():
            protected = protected - alpha * total_grad.sign()
            delta = (protected - image).clamp(-epsilon, epsilon)
            protected = (image + delta).clamp(0.0, 1.0)

    return protected.detach()


def load_facenet(device: torch.device) -> torch.nn.Module:
    from facenet_pytorch import InceptionResnetV1

    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    model.requires_grad_(False)
    return model


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model_dtype = resolve_dtype(args.dtype, device)
    seed_everything(args.seed)

    if device.type != "cuda" and model_dtype != torch.float32:
        raise ValueError("float16/bfloat16 requires CUDA; use --dtype float32 on CPU.")

    input_paths = list(iter_images(args.input, args.max_images))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ip_adapter_path = Path(args.ip_adapter_path)
    if not ip_adapter_path.is_file():
        raise FileNotFoundError(
            "IP-Adapter checkpoint not found: "
            f"{ip_adapter_path}. Download it as described in checkpoints/README.md "
            "or pass --ip-adapter-path."
        )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=model_dtype,
        add_watermarker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)

    ip_model = IPAdapterPlusXL(
        pipe,
        args.image_encoder_path,
        args.ip_adapter_path,
        device,
        num_tokens=16,
        torch_dtype=model_dtype,
    )
    ip_model.image_encoder.requires_grad_(False)
    ip_model.image_proj_model.requires_grad_(False)
    ip_model.image_encoder.eval()
    ip_model.image_proj_model.eval()

    facenet = load_facenet(device)
    scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    text_embeds, pooled_text_embeds, time_ids = build_text_condition(
        pipe, args.prompt, args.image_size, device
    )

    results_path = args.output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as results_file:
        for input_path in tqdm(input_paths, desc="IDGuardian"):
            image = load_image(input_path, args.image_size, device)
            protected = pgd_protect(
                image=image,
                pipe=pipe,
                ip_model=ip_model,
                facenet=facenet,
                scheduler=scheduler,
                text_embeds=text_embeds,
                pooled_text_embeds=pooled_text_embeds,
                time_ids=time_ids,
                device=device,
                model_dtype=model_dtype,
                epsilon=args.epsilon,
                alpha=args.alpha,
                steps=args.steps,
            )
            output_path = args.output_dir / input_path.name
            save_tensor_image(protected, output_path)
            record = {
                "input": str(input_path),
                "output": str(output_path),
                "image_size": args.image_size,
                "epsilon": args.epsilon,
                "alpha": args.alpha,
                "steps": args.steps,
                "prompt": args.prompt,
                "psnr_db": psnr(image, protected),
            }
            results_file.write(json.dumps(record) + "\n")
            results_file.flush()

    print(f"Saved {len(input_paths)} protected image(s) to {args.output_dir}")
    print(f"Per-image metadata: {results_path}")


if __name__ == "__main__":
    main()
