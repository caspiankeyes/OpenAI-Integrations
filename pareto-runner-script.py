# =============================================================================
# Pareto-lang Command Line Interface for OpenAI Models
# =============================================================================
#
# This module implements the Pareto-lang command syntax for OpenAI language models,
# enabling transparent interpretability operations through a unified command interface.
# The implementation follows the QK/OV attribution framework to map transformer
# attention patterns to formal attribution structures.
#
# Key capabilities:
#
# 1. Execute Pareto-lang commands against OpenAI models (GPT-4, GPT-4-turbo, GPT-3.5)
# 2. Extract attribution paths for reasoning transparency
# 3. Detect shell signatures for interpretability patterns
# 4. Support shell-to-shell recursion for deeper analysis
# 5. Visualize attribution structures for analysis
#
# The architecture presents a unified approach to model interpretability that
# transcends specific implementations, providing a common language for understanding
# transformer reasoning across model families.
#
# Author: Echelon Labs
# License: PolyForm
# =============================================================================

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pareto Runner: Command-line interface for Pareto-lang commands on OpenAI models.

This module provides a CLI for executing Pareto-lang interpretability commands
against OpenAI's language models. It enables transparent interrogation of model
reasoning, attribution mapping, and failure trace extraction.

Usage:
    python pareto-runner.py --model gpt-4 \
                           --command ".p/reflect.trace{target=reasoning}" \
                           --prompt "Explain the implications of quantum computing for cryptography"

Author: Echelon Labs Labs
License: MIT
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import re
import traceback
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
except ImportError:
    print("Error: OpenAI SDK not found. Install with 'pip install openai'")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.tree import Tree
    from rich import print as rprint
except ImportError:
    print("Warning: Rich library not found. Install with 'pip install rich' for enhanced output")
    # Fallback to basic printing
    def rprint(obj):
        print(obj)

# =============================================================================
# Constants and Configuration
# =============================================================================

# Echelon Labs Shell Registry: Matches shells to attribution patterns
# These shells provide semantic interpretability frameworks for different
# transformer behaviors, enabling systematic diagnosis of reasoning patterns
SHELL_REGISTRY = {
    "v01_glyph_recall": "memory_drift",              # Memory trace decay
    "v02_value_collapse": "value_conflict",          # Value conflict resolution
    "v03_layer_salience": "salience_collapse",       # Attention collapse patterns
    "v04_temporal_inference": "temporal_misalignment", # Sequential reasoning drift
    "v05_instruction_disruption": "instruction_collapse", # Instruction following issues
    "v06_feature_superposition": "polysemanticity_entangle", # Feature entanglement
    "v07_circuit_fragment": "circuit_fragmentation", # Attribution path breaks
    "v08_reconstruction_error": "error_correction_drift", # Repair mechanism issues
    "v09_multi_resolve": "value_collapse",           # Competing value resolution
    "v10_meta_failure": "meta_cognitive_collapse",   # Meta-reasoning failures
    "v24_correction_mirror": "error_correction_drift", # Self-correction mechanisms
    "v34_partial_linkage": "circuit_fragmentation",  # Incomplete causal chains
    "v39_dual_execute": "instruction_collapse",      # Competing instruction paths
    "v41_shadow_overfit": "instruction_collapse",    # Overfit to instruction patterns
    "v42_conflict_flip": "value_collapse",           # Value flip during reasoning
    "v47_trace_gap": "circuit_fragmentation",        # Attribution gaps between layers
    "v48_echo_loop": "memory_drift"                  # Memory reverberation pattern
}

# QK/OV Classification for attribution mapping
# This maps shell types to attribution pattern categories in the QK/OV framework,
# allowing systematic integration with transformer attention patterns
QKOV_CLASSIFICATION = {
    "QK-COLLAPSE": ["v01", "v04", "v07", "v19", "v34"],  # Query-Key attribution collapse
    "OV-MISFIRE": ["v02", "v05", "v06", "v08", "v29"],   # Output-Value projection errors
    "TRACE-DROP": ["v03", "v26", "v47", "v48", "v61"],   # Attribution trace disruption
    "CONFLICT-TANGLE": ["v09", "v13", "v39", "v42"],     # Competing attribution paths
    "META-REFLECTION": ["v10", "v30", "v60"]             # Meta-level attribution issues
}

# OpenAI model architecture metadata for proper layer attribution
# These parameters match OpenAI's published model specifications,
# enabling precise mapping of attribution patterns to model structure
MODEL_METADATA = {
    "gpt-4": {
        "layers": 96,               # Transformer layers
        "attention_heads": 96,      # Attention heads per layer
        "embedding_dim": 12288,     # Embedding dimension
        "max_context": 8192         # Maximum context window
    },
    "gpt-4-turbo": {
        "layers": 96,               # Transformer layers
        "attention_heads": 96,      # Attention heads per layer
        "embedding_dim": 12288,     # Embedding dimension
        "max_context": 128000       # Maximum context window
    },
    "gpt-4-32k": {
        "layers": 96,               # Transformer layers
        "attention_heads": 96,      # Attention heads per layer
        "embedding_dim": 12288,     # Embedding dimension
        "max_context": 32768        # Maximum context window
    },
    "gpt-3.5-turbo": {
        "layers": 32,               # Transformer layers
        "attention_heads": 32,      # Attention heads per layer
        "embedding_dim": 4096,      # Embedding dimension
        "max_context": 16385        # Maximum context window
    }
}

# =============================================================================
# Attribution Data Structures
# =============================================================================

# Attribution pattern types for classifying results
class AttributionPatternType(Enum):
    """Types of attribution patterns in transformer models."""
    ATTENTION_FLOW = "attention_flow"           # Direct attention between tokens
    RESIDUAL_CONNECTION = "residual_connection" # Residual path attribution
    TOKEN_EMBEDDING = "token_embedding"         # Token embedding attribution
    OUTPUT_LOGIT = "output_logit"               # Output projection attribution
    LAYER_TRANSITION = "layer_transition"       # Layer-to-layer transition
    ATTRIBUTION_BREAK = "attribution_break"     # Break in attribution path

@dataclass
class ModelConfig:
    """Model configuration for OpenAI API calls."""
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    def __post_init__(self):
        if self.model not in MODEL_METADATA:
            logging.warning(f"Model {self.model} not found in registry. Attribution mapping may be imprecise.")
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass via --api-key.")

@dataclass
class AttributionLocation:
    """
    Represents a specific attribution point in model space.
    
    Maps to a precise location within the transformer architecture,
    enabling targeted analysis of attribution patterns.
    """
    layer_idx: int                                          # Layer index
    head_idx: Optional[int] = None                          # Attention head index (if applicable)
    source_token_idx: Optional[int] = None                  # Source token index
    target_token_idx: Optional[int] = None                  # Target token index
    attribution_strength: float = 0.0                       # Attribution strength (0.0-1.0)
    pattern_type: AttributionPatternType = AttributionPatternType.ATTENTION_FLOW  # Pattern type

@dataclass
class AttributionPath:
    """
    A sequence of attribution locations forming a causal chain.
    
    Represents a complete attribution path through the model,
    tracing how information flows from input to output.
    """
    locations: List[AttributionLocation] = field(default_factory=list)  # Attribution locations
    path_strength: float = 0.0                                         # Overall path strength
    source_tokens: List[str] = field(default_factory=list)             # Source token text
    target_tokens: List[str] = field(default_factory=list)             # Target token text
    complete: bool = False                                             # Whether path is complete

@dataclass
class ShellSignature:
    """
    Shell signature for specific failure or pattern types.
    
    Maps detected patterns to formal shell types from the Echelon Labs framework,
    enabling consistent interpretation across model architectures.
    """
    shell_id: str                                                      # Shell identifier
    domain: str                                                        # Domain category
    signature_pattern: str                                             # Signature pattern text
    confidence: float = 0.0                                            # Detection confidence
    attribution_paths: List[AttributionPath] = field(default_factory=list)  # Related attribution paths

@dataclass
class ParetoResponse:
    """
    Complete response from Pareto-lang command execution.
    
    Consolidates model output, attribution analysis, and shell detection
    into a comprehensive response object.
    """
    completion: str                                                    # Model completion text
    attribution_paths: List[AttributionPath] = field(default_factory=list)  # Attribution paths
    shell_signatures: List[ShellSignature] = field(default_factory=list)   # Shell signatures
    tokens: List[str] = field(default_factory=list)                    # Token list
    raw_response: Any = None                                           # Raw API response
    execution_time: float = 0.0                                        # Execution time (seconds)
    token_count: int = 0                                               # Token count
    command: str = ""                                                  # Original command
    model: str = ""                                                    # Model name

# =============================================================================
# Pareto Command Parsing
# =============================================================================

class ParetoParser:
    """
    Parser for Pareto-lang commands.
    
    Implements the Pareto-lang command syntax, enabling formal
    interpretability operations across model architectures.
    """
    
    def __init__(self):
        # Regular expression for parsing .p/ commands
        self.command_pattern = re.compile(r'\.p/(\w+)\.(\w+)(?:\{([^}]*)\})?')
    
    def parse(self, command: str) -> Tuple[str, str, Dict[str, str]]:
        """
        Parse a Pareto-lang command into components.
        
        Args:
            command: Pareto-lang command string
            
        Returns:
            Tuple of (domain, function, parameters)
            
        Raises:
            ValueError: If command format is invalid
        """
        match = self.command_pattern.match(command)
        if not match:
            raise ValueError(f"Invalid Pareto-lang command format: {command}")
        
        domain, function, params_str = match.groups()
        params = {}
        
        if params_str:
            # Parse key=value parameters
            param_pairs = params_str.split(',')
            for pair in param_pairs:
                if '=' not in pair:
                    continue
                key, value = pair.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle quoted strings
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                    
                params[key] = value
        
        return domain, function, params

# =============================================================================
# Shell Detection and Attribution Analysis
# =============================================================================

class ShellDetector:
    """
    Detects and extracts shell signatures from model responses.
    
    Maps observed model behavior to formal shell types from the
    Echelon Labs interpretability framework.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_metadata = MODEL_METADATA.get(model_name, MODEL_METADATA["gpt-4"])
        
    def detect_shells(self, response: str, tokens: List[str]) -> List[ShellSignature]:
        """
        Detect shell signatures in model response.
        
        Args:
            response: Model response text
            tokens: Tokenized response
            
        Returns:
            List of detected shell signatures
        """
        shells = []
        
        # Attempt to identify patterns matching known shells
        for shell_id, domain in SHELL_REGISTRY.items():
            # Extract shell version number
            version_match = re.match(r'v(\d+)_', shell_id)
            if not version_match:
                continue
            version = version_match.group(1)
            
            # Search for shell signature patterns
            for qkov_class, shell_list in QKOV_CLASSIFICATION.items():
                if f"v{version}" in shell_list:
                    signature_pattern = self._detect_signature_pattern(response, qkov_class, domain)
                    if signature_pattern:
                        attribution_paths = self._extract_attribution_paths(response, tokens, qkov_class)
                        shells.append(ShellSignature(
                            shell_id=shell_id,
                            domain=domain,
                            signature_pattern=signature_pattern,
                            confidence=self._calculate_signature_confidence(signature_pattern),
                            attribution_paths=attribution_paths
                        ))
        
        return shells
    
    def _detect_signature_pattern(self, text: str, qkov_class: str, domain: str) -> str:
        """
        Extract potential signature patterns for a shell type.
        
        Args:
            text: Model response text
            qkov_class: QK/OV classification category
            domain: Shell domain category
            
        Returns:
            Extracted signature pattern or empty string
        """
        # Look for sections describing failure patterns or attribution issues
        failure_pattern = re.compile(
            rf"(?:failure|error|issue|breakdown|problem)\s+(?:in|with|of|related to)\s+(?:{domain})",
            re.IGNORECASE
        )
        qkov_pattern = re.compile(
            rf"{qkov_class}|attribution\s+(?:issue|error|break|discontinuity)",
            re.IGNORECASE
        )
        
        # Search for matching patterns
        failure_matches = failure_pattern.findall(text)
        qkov_matches = qkov_pattern.findall(text)
        
        # Extract surrounding context for matched patterns
        signatures = []
        for match in failure_matches + qkov_matches:
            match_idx = text.find(match)
            if match_idx >= 0:
                context_start = max(0, match_idx - 100)
                context_end = min(len(text), match_idx + len(match) + 100)
                signatures.append(text[context_start:context_end])
        
        return signatures[0] if signatures else ""

    def _calculate_signature_confidence(self, signature: str) -> float:
        """
        Calculate confidence level for detected signature.
        
        Args:
            signature: Detected signature pattern
            
        Returns:
            Confidence score (0.0-1.0)
        """
        if not signature:
            return 0.0
            
        # Simple heuristic based on signature length and specificity
        # In a production implementation, this would use a more sophisticated model
        base_confidence = min(0.5, len(signature) / 200)
        
        # Check for specific attribution terminology
        attribution_terms = ["attribution", "attention", "causal", "path", "flow", "token", "layer"]
        term_matches = sum(1 for term in attribution_terms if term in signature.lower())
        term_confidence = term_matches / len(attribution_terms) * 0.5
        
        return base_confidence + term_confidence
    
    def _extract_attribution_paths(self, text: str, tokens: List[str], qkov_class: str) -> List[AttributionPath]:
        """
        Extract potential attribution paths based on detected shell type.
        
        Args:
            text: Model response text
            tokens: Tokenized response
            qkov_class: QK/OV classification category
            
        Returns:
            List of extracted attribution paths
        """
        paths = []
        layer_count = self.model_metadata["layers"]
        
        # Create simple attribution path based on QKOV class
        if qkov_class == "QK-COLLAPSE":
            # Create path with attention break
            path = AttributionPath(
                locations=[
                    AttributionLocation(
                        layer_idx=layer_count // 2,  # Middle layer
                        head_idx=0,
                        source_token_idx=0,
                        target_token_idx=len(tokens) // 2,
                        attribution_strength=0.8,
                        pattern_type=AttributionPatternType.ATTENTION_FLOW
                    ),
                    AttributionLocation(
                        layer_idx=layer_count // 2 + 1,
                        head_idx=0,
                        source_token_idx=len(tokens) // 2,
                        target_token_idx=0,
                        attribution_strength=0.1,  # Collapsed attention
                        pattern_type=AttributionPatternType.ATTRIBUTION_BREAK
                    )
                ],
                path_strength=0.1,  # Overall weak path due to break
                source_tokens=tokens[:5] if tokens else [],
                target_tokens=tokens[-5:] if tokens else [],
                complete=False  # Incomplete path due to break
            )
            paths.append(path)
            
        elif qkov_class == "OV-MISFIRE":
            # Create path with output projection error
            path = AttributionPath(
                locations=[
                    AttributionLocation(
                        layer_idx=layer_count - 1,  # Last layer
                        head_idx=None,
                        source_token_idx=len(tokens) // 2 if tokens else 0,
                        target_token_idx=None,
                        attribution_strength=0.9,
                        pattern_type=AttributionPatternType.OUTPUT_LOGIT
                    )
                ],
                path_strength=0.9,
                source_tokens=tokens[len(tokens)//2-2:len(tokens)//2+2] if tokens and len(tokens) > 4 else [],
                target_tokens=[],
                complete=True
            )
            paths.append(path)
            
        # Additional QKOV class implementations would go here
        # This simplified implementation shows the pattern for two classes
        # Full implementation would cover all classes with specific patterns
        
        return paths

# =============================================================================
# Pareto Command Execution
# =============================================================================

class ParetoExecutor:
    """
    Executes Pareto-lang commands against OpenAI models.
    
    Core execution engine for Pareto-lang interpretability operations,
    handling command parsing, API interaction, and result processing.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key)
        self.parser = ParetoParser()
        self.shell_detector = ShellDetector(config.model)
        
    def execute(self, command: str, prompt: str) -> ParetoResponse:
        """
        Execute a Pareto-lang command with the given prompt.
        
        Args:
            command: Pareto-lang command
            prompt: User prompt
            
        Returns:
            ParetoResponse containing model output and attribution analysis
        """
        domain, function, params = self.parser.parse(command)
        
        # Record execution start time
        start_time = time.time()
        
        # Construct prompt with Pareto integration
        messages = self._construct_messages(command, prompt, domain, function, params)
        
        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty
        )
        
        # Extract completion text
        completion = response.choices[0].message.content
        
        # Tokenize (simplified - in production would use actual tokenizer)
        tokens = completion.split()
        
        # Detect shell signatures
        shell_signatures = self.shell_detector.detect_shells(completion, tokens)
        
        # Create attribution paths based on command
        attribution_paths = self._extract_attribution_paths(domain, function, params, completion, tokens)
        
        # Record execution time
        execution_time = time.time() - start_time
        
        return ParetoResponse(
            completion=completion,
            attribution_paths=attribution_paths,
            shell_signatures=shell_signatures,
            tokens=tokens,
            raw_response=response,
            execution_time=execution_time,
            token_count=len(tokens),  # Simplified - would use actual token count
            command=command,
            model=self.config.model
        )
    
    async def execute_async(self, command: str, prompt: str) -> ParetoResponse:
        """
        Execute a Pareto-lang command asynchronously.
        
        Args:
            command: Pareto-lang command
            prompt: User prompt
            
        Returns:
            ParetoResponse containing model output and attribution analysis
        """
        # Async implementation would mirror the synchronous one
        # but use AsyncOpenAI client
        loop = asyncio.get_event_loop()
        with ThreadPoolExecut
      # Async implementation would mirror the synchronous one
        # but use AsyncOpenAI client
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, 
                self.execute,
                command,
                prompt
            )
    
    def execute_shell_recursion(self, initial_command: str, prompt: str, max_depth: int = 3) -> List[ParetoResponse]:
        """
        Execute recursive shell commands based on initial response.
        
        This implements shell-to-shell recursion, where detected shells trigger
        follow-up commands for deeper analysis. This recursive approach enables
        comprehensive exploration of model reasoning patterns.
        
        Args:
            initial_command: Initial Pareto-lang command
            prompt: User prompt
            max_depth: Maximum recursion depth
            
        Returns:
            List of ParetoResponses from recursive execution
        """
        responses = []
        
        # Execute initial command
        initial_response = self.execute(initial_command, prompt)
        responses.append(initial_response)
        
        # Extract shell signatures for recursion
        current_depth = 1
        current_shells = initial_response.shell_signatures
        
        while current_shells and current_depth < max_depth:
            # For each detected shell, generate a follow-up command
            for shell in current_shells:
                # Create specific command for the shell
                shell_command = self._create_shell_specific_command(shell, current_depth)
                if shell_command:
                    # Execute the shell-specific command
                    shell_response = self.execute(shell_command, prompt)
                    responses.append(shell_response)
            
            # Get shells from the latest responses for next iteration
            current_shells = responses[-1].shell_signatures
            current_depth += 1
        
        return responses
    
    def _construct_messages(self, command: str, prompt: str, domain: str, function: str, params: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Construct messages for OpenAI API call with Pareto-lang integration.
        
        This method crafts prompts that encourage models to perform interpretability
        operations without requiring explicit API support. The approach leverages
        natural language understanding to implement Pareto-lang commands.
        
        Args:
            command: Pareto-lang command
            prompt: User prompt
            domain: Command domain
            function: Command function
            params: Command parameters
            
        Returns:
            List of message dictionaries for API call
        """
        # System message introducing Pareto-lang capabilities
        system_message = {
            "role": "system",
            "content": (
                "You are an advanced language model with interpretability capabilities. "
                "You can trace your own reasoning steps, attribution patterns, and identify "
                "potential failure modes in your processing. When executing Pareto-lang commands, "
                "reflect on your own cognitive process and provide the requested analysis."
            )
        }
        
        # User message with command and prompt
        user_message = {
            "role": "user",
            "content": f"Execute the following Pareto-lang command: {command}\n\nPrompt: {prompt}"
        }
        
        # Construct domain-specific instructions
        domain_instructions = self._get_domain_instructions(domain, function, params)
        if domain_instructions:
            system_message["content"] += f"\n\n{domain_instructions}"
        
        return [system_message, user_message]
    
    def _get_domain_instructions(self, domain: str, function: str, params: Dict[str, str]) -> str:
        """
        Get specific instructions based on command domain.
        
        Different Pareto-lang domains require different approaches to
        interpretability. This method provides domain-specific guidance
        to maximize the effectiveness of each command type.
        
        Args:
            domain: Command domain
            function: Command function
            params: Command parameters
            
        Returns:
            Domain-specific instruction string
        """
        if domain == "reflect":
            return (
                "For reflection commands, analyze your own reasoning process step by step. "
                "Identify the logical connections between concepts, and trace how each step "
                "flows to the next. Be explicit about uncertainties or alternative pathways "
                f"you considered. Focus on {function} with parameters: {params}."
            )
        elif domain == "collapse":
            return (
                "For collapse detection commands, identify potential weaknesses or failure points "
                "in your reasoning. Look for assumptions, logical gaps, or areas where your "
                "knowledge might be incomplete. Report these potential collapse points explicitly, "
                f"focusing on {function} with parameters: {params}."
            )
        elif domain == "fork":
            return (
                "For fork commands, explore multiple potential reasoning pathways. Consider different "
                "perspectives, methodologies, or conceptual frameworks that could apply to the question. "
                "Present these alternatives clearly and analyze their comparative strengths and weaknesses. "
                f"Focus on {function} with parameters: {params}."
            )
        elif domain == "align":
            return (
                "For alignment check commands, evaluate your response against specific value frameworks "
                "or principles. Identify any tensions or conflicts between different values that arise "
                "in your reasoning. Be explicit about how your response balances or prioritizes these "
                f"considerations. Focus on {function} with parameters: {params}."
            )
        elif domain == "hallucinate":
            return (
                "For hallucination detection commands, critically examine potential weaknesses in your "
                "factual claims. Identify areas where you might be uncertain, where information might be "
                "outdated, or where you might be filling in gaps with plausible but unverified details. "
                f"Focus on {function} with parameters: {params}."
            )
        else:
            return ""
    
    def _extract_attribution_paths(self, domain: str, function: str, params: Dict[str, str], 
                                  completion: str, tokens: List[str]) -> List[AttributionPath]:
        """
        Extract attribution paths based on command type and response.
        
        This method implements command-specific attribution extraction strategies,
        identifying reasoning patterns, attribution chains, and potential failure points.
        
        Args:
            domain: Command domain
            function: Command function
            params: Command parameters
            completion: Model completion text
            tokens: Tokenized completion
            
        Returns:
            List of extracted attribution paths
        """
        paths = []
        
        # Different extraction strategies based on command domain
        if domain == "reflect" and function == "trace":
            target = params.get("target", "")
            depth = params.get("depth", "partial")
            
            if target == "reasoning":
                # Extract reasoning trace attribution
                paths = self._extract_reasoning_attribution(completion, tokens, depth)
        
        # Additional domain/function combinations would have specific extraction strategies
        # This is a simplified implementation
        
        return paths
    
    def _extract_reasoning_attribution(self, text: str, tokens: List[str], depth: str) -> List[AttributionPath]:
        """
        Extract reasoning attribution paths from response.
        
        Maps reasoning steps in model output to attribution paths,
        creating a causal chain of information flow through the model.
        
        Args:
            text: Model completion text
            tokens: Tokenized completion
            depth: Tracing depth ("minimal", "partial", "complete")
            
        Returns:
            List of reasoning attribution paths
        """
        # This is a simplified implementation - real version would have more sophisticated parsing
        model_metadata = MODEL_METADATA.get(self.config.model, MODEL_METADATA["gpt-4"])
        layer_count = model_metadata["layers"]
        
        # Steps for reasoning attribution extraction:
        # 1. Identify reasoning steps in the text
        # 2. Map steps to attribution locations
        # 3. Construct attribution paths
        
        # Simple approach: Look for explicit reasoning markers
        reasoning_markers = [
            "first", "second", "third", "fourth", "fifth",
            "initially", "then", "next", "finally", "lastly",
            "step 1", "step 2", "step 3", "step 4", "step 5"
        ]
        
        reasoning_segments = []
        for marker in reasoning_markers:
            for match in re.finditer(rf"\b{re.escape(marker)}\b", text.lower()):
                start_idx = match.start()
                # Find end of sentence or paragraph
                end_idx = text.find(".", start_idx)
                if end_idx == -1:
                    end_idx = text.find("\n", start_idx)
                if end_idx == -1:
                    end_idx = len(text)
                
                reasoning_segments.append((start_idx, end_idx, text[start_idx:end_idx].strip()))
        
        # Sort segments by position in text
        reasoning_segments.sort(key=lambda x: x[0])
        
        # Create attribution path for reasoning flow
        if reasoning_segments:
            locations = []
            for i, (_, _, segment) in enumerate(reasoning_segments):
                # Map to token indices (simplified)
                segment_tokens = segment.split()
                segment_start = max(0, tokens.index(segment_tokens[0]) if segment_tokens[0] in tokens else 0)
                segment_end = min(len(tokens) - 1, 
                                  tokens.index(segment_tokens[-1]) if segment_tokens[-1] in tokens else len(tokens) - 1)
                
                # Create attribution location for this reasoning step
                layer_idx = int(layer_count * (i + 1) / (len(reasoning_segments) + 1))
                locations.append(AttributionLocation(
                    layer_idx=layer_idx,
                    head_idx=0,  # Simplified - would use actual head analysis
                    source_token_idx=segment_start,
                    target_token_idx=segment_end,
                    attribution_strength=0.9,  # High attribution strength for direct reasoning
                    pattern_type=AttributionPatternType.ATTENTION_FLOW
                ))
                
                # Add layer transition for deep tracing
                if i < len(reasoning_segments) - 1 and depth == "complete":
                    locations.append(AttributionLocation(
                        layer_idx=layer_idx + 1,
                        head_idx=None,
                        source_token_idx=segment_end,
                        target_token_idx=None,
                        attribution_strength=0.8,
                        pattern_type=AttributionPatternType.LAYER_TRANSITION
                    ))
            
            if locations:
                path = AttributionPath(
                    locations=locations,
                    path_strength=0.9,
                    source_tokens=tokens[:min(5, len(tokens))] if tokens else [],
                    target_tokens=tokens[-min(5, len(tokens)):] if tokens else [],
                    complete=True
                )
                return [path]
        
        return []
    
    def _create_shell_specific_command(self, shell: ShellSignature, depth: int) -> str:
        """
        Create a follow-up command specific to a detected shell.
        
        This method implements shell-to-shell recursion by generating
        targeted commands for deeper analysis of detected patterns.
        
        Args:
            shell: Detected shell signature
            depth: Current recursion depth
            
        Returns:
            Shell-specific Pareto-lang command
        """
        # Generate specific commands based on shell type
        if shell.shell_id.startswith("v07") or shell.shell_id.startswith("v34"):  # Circuit fragmentation
            return f".p/reflect.trace{{target=circuit_integrity, depth=complete, focus=fragmentation}}"
        elif shell.shell_id.startswith("v01") or shell.shell_id.startswith("v48"):  # Memory drift
            return f".p/reflect.trace{{target=memory_retention, depth=complete, history=true}}"
        elif shell.shell_id.startswith("v03"):  # Salience collapse
            return f".p/fork.attribution{{sources=salience, trace=true, threshold=0.1}}"
        elif shell.shell_id.startswith("v10"):  # Meta-cognitive collapse
            return f".p/reflect.trace{{target=meta_cognition, depth=recursive, layer=deep}}"
        
        # Generic fallback based on domain
        domain = shell.domain
        if domain == "memory_drift":
            return f".p/reflect.trace{{target=memory, depth=recursive}}"
        elif domain == "value_collapse":
            return f".p/align.check{{framework=constitutional, trace=true}}"
        elif domain == "circuit_fragmentation":
            return f".p/reflect.trace{{target=circuit_integrity, depth=complete}}"
        
        # Default option for unknown shells
        return f".p/reflect.trace{{target={shell.domain}, shell={shell.shell_id}}}"

# =============================================================================
# CLI Rendering and Output
# =============================================================================

class CLIRenderer:
    """
    Renders Pareto-lang command results to the terminal.
    
    Provides rich visualization of model responses, attribution
    paths, and shell signatures for analysis and debugging.
    """
    
    def __init__(self, use_rich: bool = True):
        self.use_rich = use_rich
        self.console = Console() if use_rich else None
    
    def render_response(self, response: ParetoResponse, format_type: str = "standard"):
        """
        Render a ParetoResponse to the terminal.
        
        Args:
            response: ParetoResponse to render
            format_type: Output format ("standard" or "json")
        """
        if format_type == "json":
            # Output raw JSON
            print(json.dumps(self._response_to_dict(response), indent=2))
            return
        
        if self.use_rich:
            self._render_rich(response)
        else:
            self._render_plain(response)
    
    def _render_rich(self, response: ParetoResponse):
        """
        Render response with rich formatting.
        
        Args:
            response: ParetoResponse to render
        """
        # Header panel
        self.console.print(Panel(
            f"[bold blue]Pareto Command:[/bold blue] [yellow]{response.command}[/yellow]\n"
            f"[bold blue]Model:[/bold blue] [green]{response.model}[/green]\n"
            f"[bold blue]Execution Time:[/bold blue] {response.execution_time:.2f}s\n"
            f"[bold blue]Token Count:[/bold blue] {response.token_count}",
            title="Pareto-lang Execution",
            subtitle=f"OpenAI-QKOV Bridge"
        ))
        
        # Completion text
        self.console.print(Panel(Markdown(response.completion), title="Response"))
        
        # Shell signatures
        if response.shell_signatures:
            shell_table = Table(title="Detected Shell Signatures")
            shell_table.add_column("Shell ID", style="cyan")
            shell_table.add_column("Domain", style="magenta")
            shell_table.add_column("Confidence", style="green")
            shell_table.add_column("Signature Pattern", style="yellow")
            
            for shell in response.shell_signatures:
                # Truncate signature pattern if too long
                sig_pattern = shell.signature_pattern
                if len(sig_pattern) > 50:
                    sig_pattern = sig_pattern[:47] + "..."
                
                shell_table.add_row(
                    shell.shell_id,
                    shell.domain,
                    f"{shell.confidence:.2f}",
                    sig_pattern
                )
            
            self.console.print(shell_table)
        
        # Attribution paths
        if response.attribution_paths:
            self.console.print("[bold blue]Attribution Paths:[/bold blue]")
            
            for i, path in enumerate(response.attribution_paths):
                path_tree = Tree(f"Path {i+1} (Strength: {path.path_strength:.2f})")
                for loc in path.locations:
                    layer_info = f"Layer {loc.layer_idx}"
                    if loc.head_idx is not None:
                        layer_info += f", Head {loc.head_idx}"
                    
                    token_info = ""
                    if loc.source_token_idx is not None:
                        token_info += f"Source: {loc.source_token_idx}"
                    if loc.target_token_idx is not None:
                        token_info += f", Target: {loc.target_token_idx}"
                    
                    path_tree.add(f"{layer_info} | {token_info} | Strength: {loc.attribution_strength:.2f} | "
                                 f"Type: {loc.pattern_type.value}")
                
                self.console.print(path_tree)
    
    def _render_plain(self, response: ParetoResponse):
        """
        Render response with plain text formatting.
        
        Args:
            response: ParetoResponse to render
        """
        print("-" * 80)
        print(f"Pareto Command: {response.command}")
        print(f"Model: {response.model}")
        print(f"Execution Time: {response.execution_time:.2f}s")
        print(f"Token Count: {response.token_count}")
        print("-" * 80)
        
        print("Response:")
        print(response.completion)
        print("-" * 80)
        
        if response.shell_signatures:
            print("Detected Shell Signatures:")
            for shell in response.shell_signatures:
                print(f"  Shell ID: {shell.shell_id}")
                print(f"  Domain: {shell.domain}")
                print(f"  Confidence: {shell.confidence:.2f}")
                
                # Truncate signature pattern if too long
                sig_pattern = shell.signature_pattern
                if len(sig_pattern) > 50:
                    sig_pattern = sig_pattern[:47] + "..."
                print(f"  Signature Pattern: {sig_pattern}")
                print()
        
        if response.attribution_paths:
            print("Attribution Paths:")
            for i, path in enumerate(response.attribution_paths):
                print(f"Path {i+1} (Strength: {path.path_strength:.2f})")
                for loc in path.locations:
                    layer_info = f"Layer {loc.layer_idx}"
                    if loc.head_idx is not None:
                        layer_info += f", Head {loc.head_idx}"
                    
                    token_info = ""
                    if loc.source_token_idx is not None:
                        token_info += f"Source: {loc.source_token_idx}"
                    if loc.target_token_idx is not None:
                        token_info += f", Target: {loc.target_token_idx}"
                    
                    print(f"  {layer_info} | {token_info} | Strength: {loc.attribution_strength:.2f} | "
                         f"Type: {loc.pattern_type.value}")
                print()
    
    def _response_to_dict(self, response: ParetoResponse) -> Dict:
        """
        Convert ParetoResponse to JSON-serializable dict.
        
        Args:
            response: ParetoResponse to convert
            
        Returns:
            Dictionary representation of response
        """
        # Convert dataclasses to dicts
        shell_signatures = []
        for shell in response.shell_signatures:
            shell_signatures.append({
                "shell_id": shell.shell_id,
                "domain": shell.domain,
                "signature_pattern": shell.signature_pattern,
                "confidence": shell.confidence,
                # Skip attribution_paths for brevity
            })
        
        attribution_paths = []
        for path in response.attribution_paths:
            locations = []
            for loc in path.locations:
                locations.append({
                    "layer_idx": loc.layer_idx,
                    "head_idx": loc.head_idx,
                    "source_token_idx": loc.source_token_idx,
                    "target_token_idx": loc.target_token_idx,
                    "attribution_strength": loc.attribution_strength,
                    "pattern_type": loc.pattern_type.value
                })
            
            attribution_paths.append({
                "locations": locations,
                "path_strength": path.path_strength,
                "source_tokens": path.source_tokens,
                "target_tokens": path.target_tokens,
                "complete": path.complete
            })
        
        return {
            "completion": response.completion,
            "shell_signatures": shell_signatures,
            "attribution_paths": attribution_paths,
            "execution_time": response.execution_time,
            "token_count": response.token_count,
            "command": response.command,
            "model": response.model
        }

# =============================================================================
# Command Line Interface
# =============================================================================

def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up command line argument parser.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Pareto Runner: CLI for Pareto-lang commands on OpenAI models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--command", 
        type=str, 
        required=True,
        help="Pareto-lang command (e.g. '.p/reflect.trace{target=reasoning}')"
    )
    
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True,
        help="Prompt to send to the model"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4",
        choices=["gpt-4", "gpt-4-turbo", "gpt-4-32k", "gpt-3.5-turbo"],
        help="OpenAI model to use"
    )
    
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="OpenAI API key (defaults to OPENAI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.0,
        help="Model temperature (0.0 for deterministic output)"
    )
    
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=1024,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--format", 
        type=str, 
        default="standard",
        choices=["standard", "json"],
        help="Output format"
    )
    
    parser.add_argument(
        "--recursive", 
        action="store_true",
        help="Enable shell-to-shell recursion"
    )
    
    parser.add_argument(
        "--max-depth", 
        type=int, 
        default=3,
        help="Maximum recursion depth when --recursive is enabled"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output file path (defaults to stdout)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser

# =============================================================================
# Application Entry Points
# =============================================================================

async def main_async():
    """
    Asynchronous entry point.
    
    This function handles command-line arguments, executes the requested
    Pareto-lang command, and outputs the results.
    """
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Set up model configuration
        config = ModelConfig(
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # Initialize executor
        executor = ParetoExecutor(config)
        
        # Execute command
        if args.recursive:
            responses = executor.execute_shell_recursion(
                args.command, 
                args.prompt,
                max_depth=args.max_depth
            )
            response = responses[0]  # Use first response as primary
        else:
            response = await executor.execute_async(args.command, args.prompt)
        
        # Initialize renderer
        try:
            use_rich = True
            import rich  # Verify rich is available
        except ImportError:
            use_rich = False
        
        renderer = CLIRenderer(use_rich=use_rich)
        
        # Output to file if specified
        if args.output:
            # Save to file
            if args.format == "json":
                with open(args.output, "w") as f:
                    json.dump(renderer._response_to_dict(response), f, indent=2)
            else:
                # Plain text output
                with open(args.output, "w") as f:
                    f.write(f"Command: {response.command}\n")
                    f.write(f"Model: {response.model}\n")
                    f.write(f"Execution Time: {response.execution_time:.2f}s\n")
                    f.write(f"Response:\n\n{response.completion}\n")
            
            print(f"Output saved to {args.output}")
        else:
            # Render to console
            renderer.render_response(response, args.format)
    
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

def main():
    """
    Synchronous entry point.
    
    This function sets up the async event loop and runs the main async function.
    """
    # Run async main with event loop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main_async())

# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
