import mteb
from mteb.encoder_interface import PromptType
from evals.ai import AIEmbeddingModel, AIEmbeddingType, ai_embedding
import numpy as np
import asyncio


class MTEBEmbeddingModel:
    ai_model: AIEmbeddingModel

    def __init__(self, ai_model: AIEmbeddingModel) -> None:
        self.ai_model = ai_model

    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        match prompt_type:
            case PromptType.query:
                embedding_type = AIEmbeddingType.QUERY
            case PromptType.passage:
                embedding_type = AIEmbeddingType.DOCUMENT
            case None:
                embedding_type = AIEmbeddingType.DOCUMENT

        async def get_scores() -> np.ndarray:
            results = await ai_embedding(
                self.ai_model,
                sentences,
                embedding_type,
            )
            return np.array(results)

        scores = asyncio.get_event_loop().run_until_complete(get_scores())
        return scores
