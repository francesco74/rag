"""
Optimized ONNX Reranker with Deadlock Protection & Debug Logging
Safe for use with Celery/Multiprocessing
"""

import os
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

# Only load heavy dependencies when needed
try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

log = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Reranking result with score and index."""
    index: int
    score: float
    text: str = ""


class ONNXReranker:
    """
    Optimized ONNX Reranker with:
    - Aggressive batching (process 32+ docs at once)
    - Deadlock protection (Sequential execution for Celery compatibility)
    - Detailed Debug Logging
    """
    
    def __init__(
        self, 
        model_folder: str,
        batch_size: int = 32,
        max_length: int = 512,
        num_threads: int = 1  # Default to 1 for safety in Celery
    ):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "Missing dependencies. Install with:\n"
                "pip install onnxruntime transformers"
            )
        
        self.model_folder = Path(model_folder)
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_threads = num_threads
        
        log.info("="*50)
        log.info(f"Initializing ONNX Reranker")
        log.info(f"  - Model: {model_folder}")
        log.info(f"  - Batch Size: {batch_size}")
        log.info(f"  - Max Length: {max_length}")
        log.info(f"  - Threads: {num_threads} (Sequential Mode)")
        log.info("="*50)
        
        self._load_tokenizer()
        self._load_model()
    
    def _load_tokenizer(self):
        """Load tokenizer from model folder."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_folder),
                local_files_only=True
            )
            log.info("✓ Tokenizer loaded successfully")
        except Exception as e:
            log.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_model(self):
        """Load ONNX model with safe execution options."""
        possible_names = [
            "model.onnx", "model_quantized.onnx", 
            "model_int8.onnx", "onnx/model.onnx"
        ]
        
        model_path = None
        for name in possible_names:
            candidate = self.model_folder / name
            if candidate.exists():
                model_path = candidate
                log.info(f"✓ Found model file: {name}")
                break
        
        if not model_path:
            raise FileNotFoundError(f"ONNX model not found in {self.model_folder}")
        
        # Configure ONNX Runtime
        sess_options = ort.SessionOptions()
        
        # --- CRITICAL FIX FOR CELERY DEADLOCKS ---
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            start_load = time.time()
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            log.info(f"✓ ONNX model loaded safely in {time.time() - start_load:.2f}s")
            
            # Log expected inputs
            for inp in self.session.get_inputs():
                log.debug(f"  Model Input: {inp.name} - {inp.shape}")
            
        except Exception as e:
            log.error(f"Failed to load ONNX model: {e}")
            raise
    
    def _tokenize_batch(self, query: str, documents: List[str]) -> dict:
        """Tokenize query-document pairs in batch."""
        pairs = [[query, doc] for doc in documents]
        
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        )
        
        return {
            'input_ids': encoded['input_ids'].astype(np.int64),
            'attention_mask': encoded['attention_mask'].astype(np.int64),
            'token_type_ids': encoded.get('token_type_ids', 
                np.zeros_like(encoded['input_ids'])).astype(np.int64)
        }
    
    def _run_inference(self, inputs: dict) -> np.ndarray:
        """Run ONNX inference on batch."""
        input_names = [inp.name for inp in self.session.get_inputs()]
        
        onnx_inputs = {
            name: inputs[name] 
            for name in input_names 
            if name in inputs
        }
        
        outputs = self.session.run(None, onnx_inputs)
        scores = outputs[0]
        
        if len(scores.shape) > 1:
            scores = scores[:, 0]
            
        return scores
    
    def rerank(
        self, 
        query: str, 
        documents: List[str],
        return_documents: bool = False
    ) -> List[RerankResult]:
        """
        Rerank documents using optimized batching.
        """
        if not documents:
            log.warning("Rerank called with empty document list")
            return []
        
        num_docs = len(documents)
        log.info(f"Reranking {num_docs} documents against query: '{query[:30]}...'")
        
        all_scores = np.zeros(num_docs, dtype=np.float32)
        total_start = time.time()
        
        # Process in batches
        for batch_idx, i in enumerate(range(0, num_docs, self.batch_size)):
            batch_docs = documents[i:i + self.batch_size]
            batch_size_actual = len(batch_docs)
            
            log.debug(f"Processing Batch {batch_idx+1}: Docs {i} to {i+batch_size_actual}")
            
            try:
                # Tokenize
                t_start = time.time()
                inputs = self._tokenize_batch(query, batch_docs)
                t_dur = time.time() - t_start
                
                # Inference
                i_start = time.time()
                batch_scores = self._run_inference(inputs)
                i_dur = time.time() - i_start
                
                all_scores[i:i + batch_size_actual] = batch_scores
                
                log.debug(f"  ↳ Batch {batch_idx+1}: Tok={t_dur:.3f}s, Inf={i_dur:.3f}s | Scores: Min={batch_scores.min():.3f} Max={batch_scores.max():.3f}")
                
            except Exception as e:
                log.error(f"Batch {batch_idx+1} failed: {e}", exc_info=True)
                raise
        
        total_dur = time.time() - total_start
        docs_per_sec = num_docs / total_dur if total_dur > 0 else 0
        
        log.info(f"Reranking complete: {total_dur:.3f}s ({docs_per_sec:.1f} docs/sec)")
        
        # Create results
        results = [
            RerankResult(
                index=idx,
                score=float(score),
                text=documents[idx] if return_documents else ""
            )
            for idx, score in enumerate(all_scores)
        ]
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        if results:
            log.info(f"Top Result: Score {results[0].score:.4f} (Index {results[0].index})")
            
            top_k = min(10, len(results))
            log.info(f"--- Top {top_k} Reranked Documents ---")
            for i, res in enumerate(results[:top_k]):
                # Grab a snippet of the document for context (handle safely)
                snippet = documents[res.index][:80].replace('\n', ' ')
                log.debug(f"  Rank {i+1:2d} | Score: {res.score: .4f} | Index: {res.index:3d} | Text: {snippet}...")
            
        return results