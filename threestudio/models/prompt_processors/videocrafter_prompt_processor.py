import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *

from omegaconf import OmegaConf

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], "threestudio", 'models', 'guidance', 'videocrafter'))
from threestudio.models.guidance.videocrafter.utils.utils import instantiate_from_config
from threestudio.models.guidance.videocrafter.scripts.evaluation.funcs import load_model_checkpoint


@threestudio.register("videocrafter-prompt-processor")
class VideoCrafterPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pass

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def destroy_text_encoder(self) -> None:
        # del self.tokenizer
        # del self.text_encoder
        cleanup()

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B 77 768"], Float[Tensor, "B 77 768"]]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        # Tokenize text and get embeddings
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
            uncond_text_embeddings = self.text_encoder(
                uncond_tokens.input_ids.to(self.device)
            )[0]
        return text_embeddings, uncond_text_embeddings

    ###

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir, cfg):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        config = OmegaConf.load(cfg.config)
        model_config = config.pop("model", OmegaConf.create())
        model = instantiate_from_config(model_config)
        model = load_model_checkpoint(model, pretrained_model_name_or_path)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        
        embedding_module = model.cond_stage_model
        with torch.no_grad():
            text_embeddings = model.cond_stage_model.encode(prompts)

        for prompt, embedding in zip(prompts, text_embeddings):
            torch.save(
                embedding,
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        del model
