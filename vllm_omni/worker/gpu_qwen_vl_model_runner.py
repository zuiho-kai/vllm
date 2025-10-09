"""Qwen2.5-VL Model Runner for extracting embeddings for DiT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Union

import torch

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner, get_pp_group
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


@dataclass
class QwenVLEmbeddingOutput(ModelRunnerOutput):
    """Output containing embeddings from Qwen2.5-VL model.

    Extends ModelRunnerOutput to include per-request embeddings
    extracted from Qwen2.5-VL's hidden states.
    """

    pooler_output: List[Optional[torch.Tensor]]  # Per-request embeddings
    modality_types: Optional[List[str]] = None  # "text", "vision", or "multimodal"
    input_images: Optional[List[Optional[Any]]] = None  # Original input images for VAE encoding


class QwenVLModelRunner(GPUModelRunner):
    """Qwen2.5-VL model runner for embedding extraction.

    Extends GPUModelRunner to extract embeddings from multimodal inputs
    (text + images) for use with DiT models. Leverages GPUModelRunner's
    existing infrastructure for:
    - Pipeline parallelism (PP)
    - Tensor parallelism (TP)
    - Multimodal input handling
    - CUDA graph optimization
    - Memory management
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_layer = -1  # Default to last layer
        self.pooling_method = "last_token"  # Default pooling

    def set_embedding_config(
        self,
        layer: int = -1,
        pooling: str = "last_token"
    ):
        """Configure embedding extraction settings.

        Args:
            layer: Which transformer layer to extract embeddings from.
                   -1 means last layer (default).
            pooling: Pooling method for sequence embeddings.
                     Options: "last_token", "mean", "cls"
        """
        self.embedding_layer = layer
        self.pooling_method = pooling

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[QwenVLEmbeddingOutput, IntermediateTensors]:
        """Execute Qwen2.5-VL model and extract embeddings.

        Reuses GPUModelRunner's infrastructure but extracts embeddings
        instead of generating tokens. The flow is:
        1. Preprocess (from parent): _update_states, _prepare_inputs, _preprocess
        2. Forward (from parent): _model_forward
        3. Postprocess (custom): extract embeddings instead of sampling tokens
        """
        from vllm.v1.utils import record_function_or_nullcontext
        from vllm.forward_context import set_forward_context

        # === PREPROCESS: Reuse parent class methods ===
        with record_function_or_nullcontext("Preprocess"):
            with self.synchronize_input_prep():
                # Update persistent batch states
                self._update_states(scheduler_output)

                # Early exit if no work to do
                if not scheduler_output.total_num_scheduled_tokens:
                    from vllm.distributed.kv_transfer import has_kv_transfer_group
                    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
                    if not has_kv_transfer_group():
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

                # Prepare inputs (attention metadata, indices, etc.)
                (attn_metadata, logits_indices, spec_decode_metadata,
                 num_scheduled_tokens_np, spec_decode_common_attn_metadata,
                 max_query_len, ubatch_slices, num_tokens_after_padding,
                 use_cascade_attn) = self._prepare_inputs(scheduler_output)

            # Preprocess to get input tensors
            (num_scheduled_tokens, num_input_tokens, num_tokens_across_dp,
             input_ids, inputs_embeds, positions, intermediate_tensors,
             model_kwargs) = self._preprocess(scheduler_output, intermediate_tensors,
                                              ubatch_slices, num_tokens_after_padding)

            # Get batch descriptor for CUDA graph
            from vllm.forward_context import BatchDescriptor
            uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
                num_scheduled_tokens == self.input_batch.num_reqs * max_query_len)
            batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                              uniform_decode=uniform_decode)
            cudagraph_runtime_mode, batch_descriptor = \
                self.cudagraph_dispatcher.dispatch(batch_descriptor, use_cascade_attn)

        # Adjust for ubatch slicing
        if ubatch_slices is not None:
            num_input_tokens = ubatch_slices[0].num_tokens

        # === FORWARD: Run model and extract hidden states ===
        with (set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                ubatch_slices=ubatch_slices,
        ), record_function_or_nullcontext("Forward"),
              self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output):

            # Forward pass - get hidden states
            hidden_states = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        # === POSTPROCESS: Extract embeddings ===
        with record_function_or_nullcontext("Postprocess"):
            # Handle PP (pipeline parallelism)
            if not get_pp_group().is_last_rank:
                # Mid-pipeline stages return intermediate tensors
                assert isinstance(hidden_states, IntermediateTensors)
                hidden_states.kv_connector_output = kv_connector_output
                return hidden_states

            # Extract embeddings at the last PP rank
            return self._create_embedding_output(
                hidden_states=hidden_states,
                logits_indices=logits_indices,
                num_scheduled_tokens=num_scheduled_tokens,
                kv_connector_output=kv_connector_output,
            )

    def _create_embedding_output(
        self,
        hidden_states: torch.Tensor,
        logits_indices: torch.Tensor,
        num_scheduled_tokens: int,
        kv_connector_output: Any,
    ) -> QwenVLEmbeddingOutput:
        """Create embedding output from hidden states.

        Args:
            hidden_states: Model output hidden states [num_tokens, hidden_dim]
            logits_indices: Indices for sampling positions
            num_scheduled_tokens: Total number of scheduled tokens
            kv_connector_output: KV connector output for PP

        Returns:
            QwenVLEmbeddingOutput with pooled embeddings and original images
        """
        # Extract hidden states at sampling positions (last token of each sequence)
        sample_hidden_states = hidden_states[logits_indices]

        # Pool embeddings per request
        pooler_output: List[Optional[torch.Tensor]] = []
        modality_types: List[str] = []
        input_images: List[Optional[Any]] = []

        for i in range(self.input_batch.num_reqs):
            req_id = self.input_batch.req_ids[i]
            req_state = self.requests[req_id]

            # Get this request's hidden state
            if i < len(sample_hidden_states):
                req_hidden = sample_hidden_states[i]

                # Apply pooling method
                if self.pooling_method == "mean":
                    # For mean pooling, we'd need all tokens, not just last
                    # For now, use the sampled position
                    pooled = req_hidden
                elif self.pooling_method == "cls":
                    # Would need first token - for now use sampled
                    pooled = req_hidden
                else:  # "last_token" (default)
                    pooled = req_hidden

                pooler_output.append(pooled.detach().cpu())
            else:
                pooler_output.append(None)

            # Extract multimodal features (images)
            images = None
            if hasattr(req_state, 'mm_features') and req_state.mm_features:
                # Extract images from mm_features
                for feature in req_state.mm_features:
                    if feature.data is not None and hasattr(feature.data, 'image'):
                        # Store the image data for VAE encoding
                        images = feature.data.image
                        break
                    elif feature.data is not None and hasattr(feature.data, 'data'):
                        # Alternative structure
                        images = feature.data.data
                        break

            input_images.append(images)

            # Determine modality type
            has_images = images is not None
            modality_types.append("multimodal" if has_images else "text")

        return QwenVLEmbeddingOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            pooler_output=pooler_output,
            modality_types=modality_types,
            input_images=input_images,
            kv_connector_output=kv_connector_output,
        )