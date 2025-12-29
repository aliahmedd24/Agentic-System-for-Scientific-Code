"""
Tests for Metrics Collector - Performance and accuracy tracking.
"""

import pytest
from datetime import datetime


# ============================================================================
# MetricType Enum Tests
# ============================================================================

class TestMetricType:
    """Tests for MetricType enum."""

    def test_all_metric_types_exist(self):
        """Test all expected metric types are defined."""
        from core.metrics import MetricType

        assert MetricType.ACCURACY.value == "accuracy"
        assert MetricType.PERFORMANCE.value == "performance"
        assert MetricType.QUALITY.value == "quality"
        assert MetricType.SUCCESS_RATE.value == "success_rate"
        assert MetricType.TIMING.value == "timing"


# ============================================================================
# MetricDataPoint Tests
# ============================================================================

class TestMetricDataPoint:
    """Tests for MetricDataPoint model."""

    def test_datapoint_creation(self):
        """Test creating a data point."""
        from core.metrics import MetricDataPoint, MetricType

        point = MetricDataPoint(
            timestamp=datetime.now(),
            metric_type=MetricType.ACCURACY,
            name="test_metric",
            value=0.85
        )

        assert point.metric_type == MetricType.ACCURACY
        assert point.name == "test_metric"
        assert point.value == 0.85
        assert point.metadata == {}

    def test_datapoint_with_metadata(self):
        """Test data point with metadata."""
        from core.metrics import MetricDataPoint, MetricType

        point = MetricDataPoint(
            timestamp=datetime.now(),
            metric_type=MetricType.TIMING,
            name="execution",
            value=150.5,
            metadata={"test_name": "test_parser", "file_count": 10}
        )

        assert point.metadata["test_name"] == "test_parser"
        assert point.metadata["file_count"] == 10


# ============================================================================
# AggregatedMetric Tests
# ============================================================================

class TestAggregatedMetric:
    """Tests for AggregatedMetric model."""

    def test_aggregated_metric_creation(self):
        """Test creating aggregated metric."""
        from core.metrics import AggregatedMetric

        metric = AggregatedMetric(
            name="test_metric",
            count=5,
            mean=0.8,
            std_dev=0.1,
            min_value=0.6,
            max_value=0.95,
            last_value=0.85,
            last_updated=datetime.now()
        )

        assert metric.name == "test_metric"
        assert metric.count == 5
        assert metric.mean == 0.8
        assert metric.std_dev == 0.1


# ============================================================================
# MetricsCollector Basic Tests
# ============================================================================

class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_collector_creation(self):
        """Test creating collector with defaults."""
        from core.metrics import MetricsCollector

        collector = MetricsCollector()

        assert collector._max_history == 1000
        assert collector._data_points == {}

    def test_collector_custom_max_history(self):
        """Test collector with custom max history."""
        from core.metrics import MetricsCollector

        collector = MetricsCollector(max_history=100)

        assert collector._max_history == 100

    def test_record_metric(self, metrics_collector):
        """Test recording a single metric."""
        from core.metrics import MetricType

        metrics_collector.record(MetricType.ACCURACY, "test", 0.9)

        # Verify it was recorded
        metric = metrics_collector.get_metric(MetricType.ACCURACY, "test")
        assert metric is not None
        assert metric.last_value == 0.9

    def test_record_multiple_metrics(self, metrics_collector):
        """Test recording multiple metric values."""
        from core.metrics import MetricType

        metrics_collector.record(MetricType.ACCURACY, "test", 0.8)
        metrics_collector.record(MetricType.ACCURACY, "test", 0.9)
        metrics_collector.record(MetricType.ACCURACY, "test", 0.85)

        metric = metrics_collector.get_metric(MetricType.ACCURACY, "test")
        assert metric.count == 3
        assert metric.last_value == 0.85

    def test_record_with_metadata(self, metrics_collector):
        """Test recording metric with metadata."""
        from core.metrics import MetricType

        metrics_collector.record(
            MetricType.TIMING,
            "execution",
            150.0,
            {"test_name": "test_parser"}
        )

        # Metadata is stored with the data point


# ============================================================================
# Convenience Recording Methods Tests
# ============================================================================

class TestRecordingMethods:
    """Tests for convenience recording methods."""

    def test_record_accuracy(self, metrics_collector):
        """Test record_accuracy method."""
        from core.metrics import MetricType

        metrics_collector.record_accuracy("mapping_confidence", 0.85)

        metric = metrics_collector.get_metric(MetricType.ACCURACY, "mapping_confidence")
        assert metric is not None
        assert metric.last_value == 0.85

    def test_record_timing(self, metrics_collector):
        """Test record_timing method."""
        from core.metrics import MetricType

        metrics_collector.record_timing("test_execution", 1500.0)

        metric = metrics_collector.get_metric(MetricType.TIMING, "test_execution")
        assert metric is not None
        assert metric.last_value == 1500.0

    def test_record_success_true(self, metrics_collector):
        """Test record_success with success=True."""
        from core.metrics import MetricType

        metrics_collector.record_success("test_run", True)

        metric = metrics_collector.get_metric(MetricType.SUCCESS_RATE, "test_run")
        assert metric.last_value == 1.0

    def test_record_success_false(self, metrics_collector):
        """Test record_success with success=False."""
        from core.metrics import MetricType

        metrics_collector.record_success("test_run", False)

        metric = metrics_collector.get_metric(MetricType.SUCCESS_RATE, "test_run")
        assert metric.last_value == 0.0

    def test_record_quality(self, metrics_collector):
        """Test record_quality method."""
        from core.metrics import MetricType

        metrics_collector.record_quality("code_quality", 0.75)

        metric = metrics_collector.get_metric(MetricType.QUALITY, "code_quality")
        assert metric is not None
        assert metric.last_value == 0.75


# ============================================================================
# Aggregation Tests
# ============================================================================

class TestMetricAggregation:
    """Tests for metric aggregation."""

    def test_get_metric_not_found(self, metrics_collector):
        """Test getting non-existent metric."""
        from core.metrics import MetricType

        metric = metrics_collector.get_metric(MetricType.ACCURACY, "nonexistent")

        assert metric is None

    def test_aggregation_mean(self, metrics_collector):
        """Test mean calculation."""
        from core.metrics import MetricType

        metrics_collector.record(MetricType.ACCURACY, "test", 0.8)
        metrics_collector.record(MetricType.ACCURACY, "test", 0.9)
        metrics_collector.record(MetricType.ACCURACY, "test", 0.7)

        metric = metrics_collector.get_metric(MetricType.ACCURACY, "test")
        assert abs(metric.mean - 0.8) < 0.001  # (0.8 + 0.9 + 0.7) / 3 = 0.8

    def test_aggregation_min_max(self, metrics_collector):
        """Test min/max calculation."""
        from core.metrics import MetricType

        metrics_collector.record(MetricType.ACCURACY, "test", 0.8)
        metrics_collector.record(MetricType.ACCURACY, "test", 0.9)
        metrics_collector.record(MetricType.ACCURACY, "test", 0.7)

        metric = metrics_collector.get_metric(MetricType.ACCURACY, "test")
        assert metric.min_value == 0.7
        assert metric.max_value == 0.9

    def test_aggregation_std_dev(self, metrics_collector):
        """Test standard deviation calculation."""
        from core.metrics import MetricType

        # Record values with known std dev
        values = [80, 90, 70, 85, 95]
        for v in values:
            metrics_collector.record(MetricType.TIMING, "test", v)

        metric = metrics_collector.get_metric(MetricType.TIMING, "test")
        assert metric.std_dev > 0  # Should have some variance

    def test_aggregation_std_dev_single_value(self, metrics_collector):
        """Test std_dev with single value is 0."""
        from core.metrics import MetricType

        metrics_collector.record(MetricType.ACCURACY, "test", 0.8)

        metric = metrics_collector.get_metric(MetricType.ACCURACY, "test")
        assert metric.std_dev == 0.0

    def test_get_all_metrics(self, populated_metrics_collector):
        """Test getting all aggregated metrics."""
        all_metrics = populated_metrics_collector.get_all_metrics()

        assert len(all_metrics) > 0
        assert "accuracy.mapping_confidence" in all_metrics

    def test_get_summary(self, populated_metrics_collector):
        """Test getting summary."""
        summary = populated_metrics_collector.get_summary()

        assert "collection_start" in summary
        assert "total_data_points" in summary
        assert "metric_count" in summary
        assert "metrics" in summary


# ============================================================================
# History Limit Tests
# ============================================================================

class TestHistoryLimit:
    """Tests for history limiting."""

    def test_max_history_enforced(self):
        """Test that max history is enforced."""
        from core.metrics import MetricsCollector, MetricType

        collector = MetricsCollector(max_history=5)

        # Record more than max_history
        for i in range(10):
            collector.record(MetricType.ACCURACY, "test", float(i))

        # Should only keep last 5
        metric = collector.get_metric(MetricType.ACCURACY, "test")
        assert metric.count == 5

    def test_max_history_keeps_latest(self):
        """Test that oldest values are removed."""
        from core.metrics import MetricsCollector, MetricType

        collector = MetricsCollector(max_history=3)

        collector.record(MetricType.ACCURACY, "test", 1.0)
        collector.record(MetricType.ACCURACY, "test", 2.0)
        collector.record(MetricType.ACCURACY, "test", 3.0)
        collector.record(MetricType.ACCURACY, "test", 4.0)  # 1.0 should be removed

        metric = collector.get_metric(MetricType.ACCURACY, "test")
        assert metric.min_value == 2.0  # 1.0 was removed


# ============================================================================
# Domain-Specific Recording Tests
# ============================================================================

class TestDomainMethods:
    """Tests for domain-specific recording methods."""

    def test_record_mapping_result(self, metrics_collector):
        """Test recording a mapping result."""
        from core.metrics import MetricType

        metrics_collector.record_mapping_result(
            concept_name="attention",
            code_element="MultiHeadAttention",
            confidence=0.85
        )

        metric = metrics_collector.get_metric(MetricType.ACCURACY, "mapping_confidence")
        assert metric is not None
        assert metric.last_value == 0.85

    def test_record_mapping_result_with_validation(self, metrics_collector):
        """Test recording mapping with validation."""
        from core.metrics import MetricType

        metrics_collector.record_mapping_result(
            concept_name="attention",
            code_element="MultiHeadAttention",
            confidence=0.85,
            was_correct=True
        )

        success_metric = metrics_collector.get_metric(MetricType.SUCCESS_RATE, "mapping_correct")
        assert success_metric is not None
        assert success_metric.last_value == 1.0

    def test_record_execution_result(self, metrics_collector):
        """Test recording an execution result."""
        from core.metrics import MetricType

        metrics_collector.record_execution_result(
            test_name="test_attention",
            success=True,
            execution_time_ms=1500.0
        )

        timing = metrics_collector.get_metric(MetricType.TIMING, "test_execution")
        assert timing is not None

        success = metrics_collector.get_metric(MetricType.SUCCESS_RATE, "test_passed")
        assert success is not None
        assert success.last_value == 1.0

    def test_record_agent_operation(self, metrics_collector):
        """Test recording an agent operation."""
        from core.metrics import MetricType

        metrics_collector.record_agent_operation(
            agent_name="PaperParser",
            operation="parse",
            duration_ms=500.0,
            success=True
        )

        # Check timing was recorded
        timing = metrics_collector.get_metric(MetricType.TIMING, "agent_operation")
        assert timing is not None

    def test_record_pipeline_stage(self, metrics_collector):
        """Test recording a pipeline stage."""
        from core.metrics import MetricType

        metrics_collector.record_pipeline_stage(
            stage="paper_parsed",
            duration_ms=2000.0,
            success=True,
            data={"concepts_found": 5}
        )

        timing = metrics_collector.get_metric(MetricType.TIMING, "pipeline_stage")
        assert timing is not None


# ============================================================================
# Accuracy Score Tests
# ============================================================================

class TestAccuracyScore:
    """Tests for accuracy score calculation."""

    def test_calculate_accuracy_score_empty(self, metrics_collector):
        """Test accuracy score with no data."""
        score = metrics_collector.calculate_accuracy_score()

        assert score == 0.0

    def test_calculate_accuracy_score(self, metrics_collector):
        """Test accuracy score calculation."""
        from core.metrics import MetricType

        # Add mapping confidence metrics
        metrics_collector.record_accuracy("mapping_confidence", 0.8)
        metrics_collector.record_accuracy("mapping_confidence", 0.9)

        # Add success metrics
        metrics_collector.record_success("test_passed", True)
        metrics_collector.record_success("test_passed", True)
        metrics_collector.record_success("test_passed", False)

        score = metrics_collector.calculate_accuracy_score()

        assert 0.0 <= score <= 1.0


# ============================================================================
# Reset Tests
# ============================================================================

class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_all(self, populated_metrics_collector):
        """Test reset clears all metrics."""
        populated_metrics_collector.reset()

        all_metrics = populated_metrics_collector.get_all_metrics()
        assert len(all_metrics) == 0


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_record_zero_value(self, metrics_collector):
        """Test recording zero value."""
        from core.metrics import MetricType

        metrics_collector.record(MetricType.ACCURACY, "test", 0.0)

        metric = metrics_collector.get_metric(MetricType.ACCURACY, "test")
        assert metric.last_value == 0.0

    def test_record_negative_value(self, metrics_collector):
        """Test recording negative value (for error margins etc)."""
        from core.metrics import MetricType

        metrics_collector.record(MetricType.PERFORMANCE, "error_margin", -0.5)

        metric = metrics_collector.get_metric(MetricType.PERFORMANCE, "error_margin")
        assert metric.last_value == -0.5

    def test_record_very_large_value(self, metrics_collector):
        """Test recording very large value."""
        from core.metrics import MetricType

        metrics_collector.record(MetricType.TIMING, "long_operation", 1e9)

        metric = metrics_collector.get_metric(MetricType.TIMING, "long_operation")
        assert metric.last_value == 1e9

    def test_metric_names_with_special_chars(self, metrics_collector):
        """Test metric names with dots and underscores."""
        from core.metrics import MetricType

        metrics_collector.record(MetricType.ACCURACY, "paper.parser.confidence", 0.8)

        metric = metrics_collector.get_metric(MetricType.ACCURACY, "paper.parser.confidence")
        assert metric is not None
