"""
Core components for the Scientific Agent System.
"""

from .error_handling import (
    LogLevel, LogCategory, ErrorSeverity, ErrorCategory,
    StructuredLog, StructuredError, AgentError,
    SystemLogger, logger, RetryStrategy,
    with_retry, create_error
)

from .llm_client import (
    LLMProvider, LLMConfig, LLMClient,
    GeminiProvider, AnthropicProvider, OpenAIProvider
)

from .knowledge_graph import (
    NodeType, EdgeType, KGNode, KGEdge, KnowledgeGraph,
    create_paper_node, create_concept_node, create_function_node, create_mapping_node
)

from .orchestrator import (
    PipelineStage, PipelineEvent, PipelineResult,
    PipelineOrchestrator
)

from .resource_estimator import (
    ComputeLevel, ResourceEstimate, ResourceEstimator, estimate_resources
)

from .metrics import (
    MetricType, MetricDataPoint, AggregatedMetric, MetricsCollector,
    get_metrics_collector, record_metric
)

from .checkpointing import (
    CheckpointStage, CheckpointMetadata, Checkpoint, CheckpointManager,
    get_checkpoint_manager
)

from .bubblewrap_sandbox import (
    IsolationLevel, SandboxConfig, ExecutionResult,
    SandboxBackend, SubprocessBackend, DockerBackend,
    BubblewrapBackend, QEMUBackend, SandboxManager,
    get_sandbox_manager, execute_in_sandbox
)

from .qemu_backend import (
    QEMUAccelerator, VMState, ExecutionMode,
    QEMUVMConfig, QEMUExecutionResult,
    QEMUImageManager, QEMUMonitor, QEMUVirtualMachine,
    QEMUPool, QEMUBackendImpl, create_qemu_backend
)

from . import agent_prompts

__all__ = [
    # Error handling
    'LogLevel', 'LogCategory', 'ErrorSeverity', 'ErrorCategory',
    'StructuredLog', 'StructuredError', 'AgentError',
    'SystemLogger', 'logger', 'RetryStrategy',
    'with_retry', 'create_error',

    # LLM
    'LLMProvider', 'LLMConfig', 'LLMClient',
    'GeminiProvider', 'AnthropicProvider', 'OpenAIProvider',

    # Knowledge Graph
    'NodeType', 'EdgeType', 'KGNode', 'KGEdge', 'KnowledgeGraph',
    'create_paper_node', 'create_concept_node', 'create_function_node', 'create_mapping_node',

    # Orchestrator
    'PipelineStage', 'PipelineEvent', 'PipelineResult',
    'PipelineOrchestrator',

    # Resource Estimation
    'ComputeLevel', 'ResourceEstimate', 'ResourceEstimator', 'estimate_resources',

    # Metrics
    'MetricType', 'MetricDataPoint', 'AggregatedMetric', 'MetricsCollector',
    'get_metrics_collector', 'record_metric',

    # Checkpointing
    'CheckpointStage', 'CheckpointMetadata', 'Checkpoint', 'CheckpointManager',
    'get_checkpoint_manager',

    # Sandbox
    'IsolationLevel', 'SandboxConfig', 'ExecutionResult',
    'SandboxBackend', 'SubprocessBackend', 'DockerBackend',
    'BubblewrapBackend', 'QEMUBackend', 'SandboxManager',
    'get_sandbox_manager', 'execute_in_sandbox',

    # QEMU Backend
    'QEMUAccelerator', 'VMState', 'ExecutionMode',
    'QEMUVMConfig', 'QEMUExecutionResult',
    'QEMUImageManager', 'QEMUMonitor', 'QEMUVirtualMachine',
    'QEMUPool', 'QEMUBackendImpl', 'create_qemu_backend',

    # Prompts
    'agent_prompts'
]
