"""ESM-based protein sequence encoder."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from representation.interface import Encoder


class ESMEncoder(Encoder):
    """Lazy ESM encoder with a deterministic stub fallback."""

    CANONICAL_VOCAB = tuple("ACDEFGHIKLMNPQRSTVWY") + ("X",)

    def __init__(self, config: dict[str, Any], logger: Any | None = None) -> None:
        self.backend_preference = str(config.get("esm_backend", "auto")).strip().lower()
        self.pooling = str(config.get("esm_pooling", "mean")).strip().lower()
        self.embedding_dim = int(config.get("esm_embedding_dim", 320))
        self.allow_stub_fallback = bool(config.get("allow_stub_fallback", True))
        self.model_name = str(config.get("esm_model_name", "esm2_t6_8M_UR50D")).strip()
        self._backend_name: str | None = None
        self._backend_reason: str | None = None
        self._projection: np.ndarray | None = None
        self._transformers_model: Any | None = None
        self._transformers_tokenizer: Any | None = None
        self._esm_model: Any | None = None
        self._esm_alphabet: Any | None = None
        self._esm_batch_converter: Any | None = None
        self._char_index = {char: index for index, char in enumerate(self.CANONICAL_VOCAB)}
        super().__init__(
            name="esm",
            cache_dir=config.get("cache_dir", "data/processed/caches"),
            batch_size=int(config.get("batch_size", 32)),
            cache_enabled=bool(config.get("cache_enabled", True)),
            logger=logger,
        )

    def config_summary(self) -> dict[str, Any]:
        return {
            "backend_preference": self.backend_preference,
            "pooling": self.pooling,
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "allow_stub_fallback": self.allow_stub_fallback,
        }

    def _ensure_backend(self) -> None:
        if self._backend_name is not None:
            return

        if self.backend_preference == "stub":
            self._backend_name = "stub"
            self._backend_reason = "Stub backend was explicitly requested."
            return

        if self.backend_preference in {"auto", "transformers"} and importlib.util.find_spec("transformers") is not None:
            self._backend_name = "transformers"
            self._backend_reason = "Transformers package detected."
            return

        if self.backend_preference in {"auto", "esm"} and importlib.util.find_spec("esm") is not None:
            self._backend_name = "esm_package"
            self._backend_reason = "fair-esm package detected."
            return

        if self.allow_stub_fallback:
            self._backend_name = "stub"
            self._backend_reason = "Requested real ESM backend is unavailable; using deterministic stub backend."
            if self.logger is not None:
                self.logger.warning(self._backend_reason)
            return

        raise RuntimeError(
            "Real ESM dependencies are unavailable. Install 'esm' or 'transformers' with local model weights, "
            "or enable allow_stub_fallback."
        )

    def _transformers_repo_id(self) -> str:
        if "/" in self.model_name:
            return self.model_name
        return f"facebook/{self.model_name}"

    def _esm_builder_name(self) -> str:
        if "/" in self.model_name:
            return self.model_name.split("/")[-1]
        return self.model_name

    def _ensure_transformers_model(self) -> None:
        if self._transformers_model is not None and self._transformers_tokenizer is not None:
            return

        import torch  # type: ignore
        from transformers import AutoTokenizer, EsmModel  # type: ignore

        repo_id = self._transformers_repo_id()
        self._transformers_tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self._transformers_model = EsmModel.from_pretrained(repo_id)
        self._transformers_model.eval()
        self.embedding_dim = int(self._transformers_model.config.hidden_size)
        if self.logger is not None:
            self.logger.info("Loaded transformers ESM backend '%s' with hidden size %s.", repo_id, self.embedding_dim)

    def _ensure_esm_package_model(self) -> None:
        if self._esm_model is not None and self._esm_batch_converter is not None:
            return

        import esm  # type: ignore

        builder_name = self._esm_builder_name()
        builder = getattr(esm.pretrained, builder_name, None)
        if builder is None:
            raise ValueError(f"fair-esm does not expose pretrained builder '{builder_name}'.")
        self._esm_model, self._esm_alphabet = builder()
        self._esm_model.eval()
        self._esm_batch_converter = self._esm_alphabet.get_batch_converter()
        self.embedding_dim = int(getattr(self._esm_model, "embed_dim"))
        if self.logger is not None:
            self.logger.info(
                "Loaded fair-esm backend '%s' with hidden size %s.",
                builder_name,
                self.embedding_dim,
            )

    def _stub_projection(self) -> np.ndarray:
        if self._projection is None:
            base_dim = len(self.CANONICAL_VOCAB) * 2 + 1
            rng = np.random.default_rng(7)
            self._projection = (
                rng.standard_normal((base_dim, self.embedding_dim)).astype(np.float32) / np.sqrt(base_dim)
            )
        return self._projection

    def _validate_sequences(self, sequences: tuple[str, ...]) -> None:
        illegal = sorted(
            {
                char
                for sequence in sequences
                for char in sequence
                if char not in self._char_index
            }
        )
        if illegal:
            raise ValueError(f"ESMEncoder encountered unsupported residue(s): {illegal}")

    def _stub_encode(self, sequences: tuple[str, ...]) -> np.ndarray:
        self._validate_sequences(sequences)
        vocab_size = len(self.CANONICAL_VOCAB)
        base_dim = vocab_size * 2 + 1
        projection = self._stub_projection()
        base_features = np.zeros((len(sequences), base_dim), dtype=np.float32)
        for row_index, sequence in enumerate(sequences):
            length = max(len(sequence), 1)
            for position, char in enumerate(sequence):
                char_index = self._char_index[char]
                base_features[row_index, char_index] += 1.0 / length
                base_features[row_index, vocab_size + char_index] += (position + 1) / length
            base_features[row_index, -1] = float(len(sequence)) / 1024.0
        embedded = base_features @ projection
        if self.pooling == "cls":
            embedded *= 0.95
        return embedded.astype(np.float32, copy=False)

    def _transformers_encode(self, sequences: tuple[str, ...]) -> np.ndarray:
        self._validate_sequences(sequences)
        self._ensure_transformers_model()

        import torch  # type: ignore

        encoded = self._transformers_tokenizer(
            list(sequences),
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_special_tokens_mask=True,
        )
        with torch.no_grad():
            outputs = self._transformers_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
        hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            valid_mask = encoded["attention_mask"].bool()
            special_mask = encoded["special_tokens_mask"].bool()
            residue_mask = valid_mask & (~special_mask)
            pooled_rows = []
            for row_index in range(hidden.shape[0]):
                row_mask = residue_mask[row_index]
                if not bool(row_mask.any()):
                    row_mask = valid_mask[row_index]
                pooled_rows.append(hidden[row_index][row_mask].mean(dim=0))
            pooled = torch.stack(pooled_rows, dim=0)
        return pooled.detach().cpu().numpy().astype(np.float32, copy=False)

    def _esm_package_encode(self, sequences: tuple[str, ...]) -> np.ndarray:
        self._validate_sequences(sequences)
        self._ensure_esm_package_model()

        import torch  # type: ignore

        batch = [(f"sequence_{index}", sequence) for index, sequence in enumerate(sequences)]
        _, _, batch_tokens = self._esm_batch_converter(batch)
        with torch.no_grad():
            outputs = self._esm_model(
                batch_tokens,
                repr_layers=[self._esm_model.num_layers],
                return_contacts=False,
            )
        hidden = outputs["representations"][self._esm_model.num_layers]
        if self.pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            pooled_rows = []
            for row_index, sequence in enumerate(sequences):
                pooled_rows.append(hidden[row_index, 1 : len(sequence) + 1].mean(dim=0))
            pooled = torch.stack(pooled_rows, dim=0)
        return pooled.detach().cpu().numpy().astype(np.float32, copy=False)

    def _encode_uncached_batch(self, sequences: tuple[str, ...]) -> np.ndarray:
        self._ensure_backend()
        if self._backend_name == "transformers":
            return self._transformers_encode(sequences)
        if self._backend_name == "esm_package":
            return self._esm_package_encode(sequences)
        if self._backend_name == "stub":
            return self._stub_encode(sequences)
        raise RuntimeError(f"Unsupported ESM backend '{self._backend_name}'.")

    def get_dim(self) -> int:
        return self.embedding_dim

    def backend_info(self) -> dict[str, Any]:
        self._ensure_backend()
        return {
            "backend": self._backend_name,
            "reason": self._backend_reason,
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "pooling": self.pooling,
        }
