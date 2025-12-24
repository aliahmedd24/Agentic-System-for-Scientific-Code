"""
Metrics Collector - Tracks accuracy, performance, and quality metrics.

This module provides:
- Mapping accuracy tracking
- Execution success rates
- Performance timing
- Quality scoring
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics


class MetricType(Enum):
    """Types of metrics tracked."""
    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    SUCCESS_RATE = "success_rate"
    TIMING = "timing"


@dataclass
class MetricDataPoint:
    """Single metric data point."""
    timestamp: datetime
    metric_type: MetricType
    name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedMetric:
    """Aggregated metric statistics."""
    name: str
    count: int
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    last_value: float
    last_updated: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "mean": round(self.mean, 4),
            "std_dev": round(self.std_dev, 4),
            "min": round(self.min_value, 4),
            "max": round(self.max_value, 4),
            "last_value": round(self.last_value, 4),
            "last_updated": self.last_updated.isoformat()
        }


class MetricsCollector:
    """
    Collects and aggregates metrics for pipeline analysis.

    Tracks:
    - Mapping accuracy (concept-to-code match quality)
    - Execution success rates
    - Agent performance timing
    - Overall quality scores
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.

        Args:
            max_history: Maximum data points to keep per metric
        """
        self._data_points: Dict[str, List[MetricDataPoint]] = {}
        self._max_history = max_history
        self._start_time = datetime.now()

    def record(
        self,
        metric_type: MetricType,
        name: str,
        value: float,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Record a metric data point.

        Args:
            metric_type: Type of metric
            name: Metric name (e.g., "mapping_confidence")
            value: Numeric value
            metadata: Optional additional context
        """
        key = f"{metric_type.value}.{name}"

        if key not in self._data_points:
            self._data_points[key] = []

        point = MetricDataPoint(
            timestamp=datetime.now(),
            metric_type=metric_type,
            name=name,
            value=value,
            metadata=metadata or {}
        )

        self._data_points[key].append(point)

        # Trim if exceeds max history
        if len(self._data_points[key]) > self._max_history:
            self._data_points[key] = self._data_points[key][-self._max_history:]

    def record_accuracy(self, name: str, value: float, **metadata) -> None:
        """Record an accuracy metric."""
        self.record(MetricType.ACCURACY, name, value, metadata)

    def record_timing(self, name: str, duration_ms: float, **metadata) -> None:
        """Record a timing metric."""
        self.record(MetricType.TIMING, name, duration_ms, metadata)

    def record_success(self, name: str, success: bool, **metadata) -> None:
        """Record a success/failure metric."""
        self.record(MetricType.SUCCESS_RATE, name, 1.0 if success else 0.0, metadata)

    def record_quality(self, name: str, score: float, **metadata) -> None:
        """Record a quality score metric."""
        self.record(MetricType.QUALITY, name, score, metadata)

    def get_metric(self, metric_type: MetricType, name: str) -> Optional[AggregatedMetric]:
        """
        Get aggregated statistics for a metric.

        Args:
            metric_type: Type of metric
            name: Metric name

        Returns:
            AggregatedMetric with statistics, or None if no data
        """
        key = f"{metric_type.value}.{name}"

        if key not in self._data_points or not self._data_points[key]:
            return None

        points = self._data_points[key]
        values = [p.value for p in points]

        return AggregatedMetric(
            name=name,
            count=len(values),
            mean=statistics.mean(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
            min_value=min(values),
            max_value=max(values),
            last_value=values[-1],
            last_updated=points[-1].timestamp
        )

    def get_all_metrics(self) -> Dict[str, AggregatedMetric]:
        """Get all aggregated metrics."""
        result = {}

        for key in self._data_points:
            parts = key.split(".", 1)
            if len(parts) == 2:
                metric_type = MetricType(parts[0])
                name = parts[1]
                metric = self.get_metric(metric_type, name)
                if metric:
                    result[key] = metric

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        metrics = self.get_all_metrics()

        summary = {
            "collection_start": self._start_time.isoformat(),
            "total_data_points": sum(len(pts) for pts in self._data_points.values()),
            "metric_count": len(metrics),
            "metrics": {}
        }

        for key, metric in metrics.items():
            summary["metrics"][key] = metric.to_dict()

        # Calculate key statistics
        accuracy_metrics = [m for k, m in metrics.items() if "accuracy" in k.lower()]
        if accuracy_metrics:
            summary["overall_accuracy"] = round(
                statistics.mean(m.mean for m in accuracy_metrics), 4
            )

        success_metrics = [m for k, m in metrics.items() if "success" in k.lower()]
        if success_metrics:
            summary["overall_success_rate"] = round(
                statistics.mean(m.mean for m in success_metrics), 4
            )

        timing_metrics = [m for k, m in metrics.items() if "timing" in k.lower()]
        if timing_metrics:
            summary["avg_timing_ms"] = round(
                statistics.mean(m.mean for m in timing_metrics), 2
            )

        return summary

    def record_mapping_result(
        self,
        concept_name: str,
        code_element: str,
        confidence: float,
        was_correct: Optional[bool] = None
    ) -> None:
        """
        Record a concept-to-code mapping result.

        Args:
            concept_name: Name of the paper concept
            code_element: Matched code element
            confidence: Confidence score (0-1)
            was_correct: Optional ground truth validation
        """
        self.record_accuracy(
            "mapping_confidence",
            confidence,
            concept=concept_name,
            code_element=code_element
        )

        if was_correct is not None:
            self.record_success(
                "mapping_correct",
                was_correct,
                concept=concept_name,
                code_element=code_element,
                confidence=confidence
            )

    def record_execution_result(
        self,
        test_name: str,
        success: bool,
        execution_time_ms: float,
        error: str = None
    ) -> None:
        """
        Record a test execution result.

        Args:
            test_name: Name of the test
            success: Whether execution succeeded
            execution_time_ms: Execution time in milliseconds
            error: Optional error message
        """
        self.record_success(
            "test_execution",
            success,
            test_name=test_name,
            error=error
        )

        self.record_timing(
            "test_execution",
            execution_time_ms,
            test_name=test_name,
            success=success
        )

    def record_agent_operation(
        self,
        agent_name: str,
        operation: str,
        duration_ms: float,
        success: bool = True
    ) -> None:
        """
        Record an agent operation.

        Args:
            agent_name: Name of the agent
            operation: Operation name
            duration_ms: Duration in milliseconds
            success: Whether operation succeeded
        """
        self.record_timing(
            f"agent.{agent_name}.{operation}",
            duration_ms,
            agent=agent_name,
            operation=operation
        )

        self.record_success(
            f"agent.{agent_name}",
            success,
            operation=operation,
            duration_ms=duration_ms
        )

    def record_pipeline_stage(
        self,
        stage: str,
        duration_ms: float,
        success: bool,
        data: Dict[str, Any] = None
    ) -> None:
        """
        Record a pipeline stage completion.

        Args:
            stage: Stage name
            duration_ms: Duration in milliseconds
            success: Whether stage succeeded
            data: Optional stage-specific data
        """
        self.record_timing(
            f"pipeline.{stage}",
            duration_ms,
            stage=stage,
            success=success,
            **(data or {})
        )

        self.record_success(
            f"pipeline.{stage}",
            success,
            duration_ms=duration_ms
        )

    def calculate_accuracy_score(self) -> float:
        """
        Calculate overall accuracy score based on multiple factors.

        Returns:
            Score from 0-1 representing overall accuracy
        """
        weights = {
            "mapping_confidence": 0.4,
            "mapping_correct": 0.3,
            "test_execution": 0.3
        }

        total_weight = 0
        weighted_sum = 0

        for name, weight in weights.items():
            for key, points in self._data_points.items():
                if name in key and points:
                    values = [p.value for p in points]
                    weighted_sum += statistics.mean(values) * weight
                    total_weight += weight
                    break

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def reset(self) -> None:
        """Reset all metrics."""
        self._data_points.clear()
        self._start_time = datetime.now()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def record_metric(metric_type: MetricType, name: str, value: float, **metadata) -> None:
    """Convenience function to record a metric."""
    get_metrics_collector().record(metric_type, name, value, metadata)
