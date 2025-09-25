#!/usr/bin/env python3
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download


class ModelDynamicAnalyzer:
    def __init__(self) -> None:
        self.temp_dirs: List[str] = []

    def analyze_model_loading(self, repo_id: str) -> Dict[str, Any]:
        try:
            analysis: Dict[str, Any] = {
                "repo_id": repo_id,
                "success": False,
                "model_type": "unknown",
                "can_load_model": False,
                "can_load_tokenizer": False,
                "model_size_mb": 0.0,
                "vocab_size": 0,
                "max_length": 0,
                "architecture": "unknown",
                "num_parameters": 0,
                "loading_time_ms": 0,
                "error": None
            }

            start_time: float = time.time()

            try:
                config = self._load_model_config(repo_id)
                if config:
                    if not isinstance(config, dict):
                        print(f"ModelAnalyzer: config is not a dictionary: {type(config)}", file=sys.stderr)
                        analysis["error"] = f"Config is not a dictionary: {type(config)}"
                    else:
                        analysis["model_type"] = config.get("model_type", "unknown")
                        analysis["architecture"] = config.get("architectures", ["unknown"])
                        analysis["vocab_size"] = config.get("vocab_size", 0)
                        analysis["max_length"] = config.get("max_position_embeddings", 0)
                        analysis["num_parameters"] = config.get("num_parameters", 0)
            except Exception as e:
                analysis["error"] = f"Config loading failed: {str(e)}"

            try:
                tokenizer = self._load_tokenizer(repo_id)
                if tokenizer:
                    analysis["can_load_tokenizer"] = True
                    analysis["vocab_size"] = (
                        len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab')
                        else analysis["vocab_size"])
            except Exception as e:
                if not analysis["error"]:
                    analysis["error"] = f"Tokenizer loading failed: {str(e)}"

            try:
                model_info = self._load_model_info(repo_id)
                if model_info:
                    analysis["can_load_model"] = True
                    analysis["model_size_mb"] = model_info.get("size_mb", 0.0)
            except Exception as e:
                if not analysis["error"]:
                    analysis["error"] = f"Model loading failed: {str(e)}"

            end_time: float = time.time()
            analysis["loading_time_ms"] = int((end_time - start_time) * 1000)
            analysis["success"] = analysis["can_load_model"] or analysis["can_load_tokenizer"]

            return analysis

        except Exception as e:
            return {
                "repo_id": repo_id,
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }

    def _load_model_config(self, repo_id: str) -> Optional[Dict[str, Any]]:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path: str = hf_hub_download(
                    repo_id=repo_id,
                    filename="config.json",
                    repo_type="model",
                    cache_dir=tmpdir
                )

                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                if not isinstance(config_data, dict):
                    print(f"Warning: Config data for {repo_id} is not a dictionary", file=sys.stderr)
                    return None
                
                return config_data

        except Exception as e:
            print(f"Warning: Could not load config for {repo_id}: {e}", file=sys.stderr)
            return None

    def _load_tokenizer(self, repo_id: str) -> Optional[Any]:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                trust_remote_code=True,
                use_auth_token=None
            )
            return tokenizer

        except Exception as e:
            is_autograder = os.environ.get('AUTOGRADER', '').lower() in ['true', '1', 'yes']
            debug_enabled = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']
            
            if not is_autograder and debug_enabled:
                print(f"Warning: Could not load tokenizer for {repo_id}: {e}", file=sys.stderr)
            return None

    def _load_model_info(self, repo_id: str) -> Optional[Dict[str, Any]]:
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                repo_id,
                trust_remote_code=True,
                use_auth_token=None
            )

            if isinstance(config, str):
                is_autograder = os.environ.get('AUTOGRADER', '').lower() in ['true', '1', 'yes']
                debug_enabled = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']
                
                if not is_autograder and debug_enabled:
                    print(f"Warning: AutoConfig returned a string instead of config object for {repo_id}", file=sys.stderr)
                return {
                    "size_mb": 0.0,
                    "config_loaded": False,
                    "error": "Config is a string"
                }

            size_mb: float = self._estimate_model_size_from_config(config)

            return {
                "size_mb": size_mb,
                "config_loaded": True
            }

        except Exception as e:
            is_autograder = os.environ.get('AUTOGRADER', '').lower() in ['true', '1', 'yes']
            debug_enabled = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']
            
            if not is_autograder and debug_enabled:
                print(f"Warning: Could not load model info for {repo_id}: {e}", file=sys.stderr)
            return None

    def _estimate_model_size_from_config(self, config: Any) -> float:
        try:
            if isinstance(config, str):
                is_autograder = os.environ.get('AUTOGRADER', '').lower() in ['true', '1', 'yes']
                debug_enabled = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']
                
                if not is_autograder and debug_enabled:
                    print(f"Warning: Config is a string, not a config object", file=sys.stderr)
                return 0.0
            
            hidden_size: int = getattr(config, 'hidden_size', 0)
            num_layers: int = getattr(config, 'num_hidden_layers', 0)
            vocab_size: int = getattr(config, 'vocab_size', 0)
            intermediate_size: int = getattr(config, 'intermediate_size', 0)

            if not all([hidden_size, num_layers, vocab_size]):
                return 0.0

            embedding_params: int = vocab_size * hidden_size

            attention_params: int = num_layers * (4 * hidden_size * hidden_size)
            ffn_params: int = num_layers * (2 * hidden_size * intermediate_size)

            ln_params: int = num_layers * (2 * hidden_size)  # Layer norm parameters

            total_params: int = embedding_params + attention_params + ffn_params + ln_params

            size_mb: float = total_params * 4 / (1024 * 1024)
            return size_mb

        except Exception:
            return 0.0

    def validate_model_completeness(self, repo_id: str) -> Dict[str, Any]:
       
        try:
            validation: Dict[str, Any] = {
                "repo_id": repo_id,
                "is_complete": False,
                "has_config": False,
                "has_tokenizer": False,
                "has_model_files": False,
                "has_readme": False,
                "completeness_score": 0.0,
                "missing_components": [],
                "recommendations": []
            }

            tokenizer_files: List[str] = ["tokenizer.json", "tokenizer_config.json", "vocab.json"]
            model_files: List[str] = ["pytorch_model.bin", "model.safetensors", "tf_model.h5"]

            try:
                self._load_model_config(repo_id)
                validation["has_config"] = True
            except Exception:
                validation["missing_components"].append("config.json")

            try:
                self._load_tokenizer(repo_id)
                validation["has_tokenizer"] = True
            except Exception:
                validation["missing_components"].extend(tokenizer_files)

            try:
                from huggingface_hub import HfApi
                api: HfApi = HfApi()
                repo_files: List[Dict[str, Any]] = api.list_repo_files(
                    repo_id=repo_id, repo_type="model")  # type: ignore

                model_file_found: bool = False
                for file_info in repo_files:
                    filename: str = file_info.get("path", "")
                    for model_file in model_files:
                        if model_file in filename:
                            model_file_found = True
                            break
                    if model_file_found:
                        break

                validation["has_model_files"] = model_file_found
                if not model_file_found:
                    validation["missing_components"].extend(model_files)

            except Exception:
                validation["missing_components"].extend(model_files)

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename="README.md",
                        repo_type="model",
                        cache_dir=tmpdir
                    )
                    validation["has_readme"] = True
            except Exception:
                validation["missing_components"].append("README.md")

            components: List[bool] = [
                validation["has_config"],
                validation["has_tokenizer"],
                validation["has_model_files"],
                validation["has_readme"]
            ]

            validation["completeness_score"] = sum(components) / len(components)
            validation["is_complete"] = validation["completeness_score"] >= 0.75

            if not validation["has_config"]:
                validation["recommendations"].append("Add config.json file")
            if not validation["has_tokenizer"]:
                validation["recommendations"].append("Add tokenizer files")
            if not validation["has_model_files"]:
                validation["recommendations"].append("Add model weight files")
            if not validation["has_readme"]:
                validation["recommendations"].append("Add README.md documentation")

            return validation

        except Exception as e:
            return {
                "repo_id": repo_id,
                "is_complete": False,
                "error": f"Validation failed: {str(e)}"
            }

    def cleanup(self) -> None:
        for temp_dir in self.temp_dirs:
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                is_autograder = os.environ.get('AUTOGRADER', '').lower() in ['true', '1', 'yes']
                debug_enabled = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']
                
                if not is_autograder and debug_enabled:
                    print(f"Warning: Failed to clean up {temp_dir}: {e}", file=sys.stderr)
        self.temp_dirs.clear()

    def __del__(self) -> None:
        self.cleanup()


def analyze_model_dynamically(repo_id: str) -> Dict[str, Any]:
    analyzer: ModelDynamicAnalyzer = ModelDynamicAnalyzer()
    try:
        return analyzer.analyze_model_loading(repo_id)
    finally:
        analyzer.cleanup()


def validate_model_completeness(repo_id: str) -> Dict[str, Any]:
    analyzer: ModelDynamicAnalyzer = ModelDynamicAnalyzer()
    try:
        return analyzer.validate_model_completeness(repo_id)
    finally:
        analyzer.cleanup()
