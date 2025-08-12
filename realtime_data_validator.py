#!/usr/bin/env python3
"""
Real-Time Data Validation and Quality Control System
Ensures data integrity and quality for live tennis data streams
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re
from collections import defaultdict, deque
import statistics

# Import WebSocket components
from websocket_tennis_client import LiveMatchEvent
from realtime_ml_pipeline import LiveDataBuffer

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataQualityMetric(Enum):
    """Data quality metrics"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"


@dataclass
class ValidationIssue:
    """Data validation issue"""
    severity: ValidationSeverity
    metric: DataQualityMetric
    field: str
    message: str
    value: Any
    expected: Any
    timestamp: datetime
    match_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['severity'] = self.severity.value
        data['metric'] = self.metric.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    match_id: int
    timestamp: datetime
    overall_score: float
    metric_scores: Dict[str, float]
    issues: List[ValidationIssue]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['issues'] = [issue.to_dict() for issue in self.issues]
        return data


class LiveDataValidator:
    """Validates live tennis match data for quality and integrity"""
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        self.data_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        self.validation_stats = {
            'total_validations': 0,
            'issues_found': 0,
            'quality_scores': deque(maxlen=1000)
        }
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize data validation rules"""
        return {
            'event_key': {
                'required': True,
                'type': int,
                'min_value': 1,
                'max_value': 99999999
            },
            'event_date': {
                'required': True,
                'type': str,
                'pattern': r'^\d{4}-\d{2}-\d{2}$',
                'date_format': '%Y-%m-%d'
            },
            'event_time': {
                'required': True,
                'type': str,
                'pattern': r'^\d{2}:\d{2}$'
            },
            'first_player': {
                'required': True,
                'type': str,
                'min_length': 2,
                'max_length': 100,
                'pattern': r'^[A-Za-z\.\s\-\']+$'
            },
            'second_player': {
                'required': True,
                'type': str,
                'min_length': 2,
                'max_length': 100,
                'pattern': r'^[A-Za-z\.\s\-\']+$'
            },
            'first_player_key': {
                'required': True,
                'type': int,
                'min_value': 1,
                'max_value': 999999
            },
            'second_player_key': {
                'required': True,
                'type': int,
                'min_value': 1,
                'max_value': 999999
            },
            'final_result': {
                'required': True,
                'type': str,
                'pattern': r'^\d+\s*-\s*\d+$'
            },
            'game_result': {
                'required': True,
                'type': str,
                'pattern': r'^\d+\s*-\s*\d+$'
            },
            'tournament_name': {
                'required': True,
                'type': str,
                'min_length': 3,
                'max_length': 200
            },
            'tournament_key': {
                'required': True,
                'type': int,
                'min_value': 1,
                'max_value': 99999
            },
            'event_status': {
                'required': True,
                'type': str,
                'allowed_values': ['Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5', 'Finished', 'Live', 'Postponed', 'Cancelled']
            },
            'live': {
                'required': True,
                'type': bool
            }
        }
    
    def validate_live_event(self, event: LiveMatchEvent) -> DataQualityReport:
        """Validate a live match event and return quality report"""
        start_time = datetime.now()
        issues = []
        
        # Convert event to dict for validation
        event_data = {
            'event_key': event.event_key,
            'event_date': event.event_date,
            'event_time': event.event_time,
            'first_player': event.first_player,
            'second_player': event.second_player,
            'first_player_key': event.first_player_key,
            'second_player_key': event.second_player_key,
            'final_result': event.final_result,
            'game_result': event.game_result,
            'tournament_name': event.tournament_name,
            'tournament_key': event.tournament_key,
            'event_status': event.status,
            'live': event.live
        }
        
        # Field-level validation
        issues.extend(self._validate_fields(event_data, event.event_key))
        
        # Cross-field validation
        issues.extend(self._validate_cross_fields(event_data, event.event_key))
        
        # Historical consistency validation
        issues.extend(self._validate_historical_consistency(event, event.event_key))
        
        # Business logic validation
        issues.extend(self._validate_business_logic(event, event.event_key))
        
        # Timeliness validation
        issues.extend(self._validate_timeliness(event, event.event_key))
        
        # Calculate quality scores
        metric_scores = self._calculate_metric_scores(issues)
        overall_score = self._calculate_overall_score(metric_scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues)
        
        # Update statistics
        self.validation_stats['total_validations'] += 1
        self.validation_stats['issues_found'] += len(issues)
        self.validation_stats['quality_scores'].append(overall_score)
        
        # Store event in history
        self.data_history[event.event_key].append(event)
        
        report = DataQualityReport(
            match_id=event.event_key,
            timestamp=start_time,
            overall_score=overall_score,
            metric_scores=metric_scores,
            issues=issues,
            recommendations=recommendations
        )
        
        # Log quality issues
        if overall_score < 0.7:
            logger.warning(f"Low data quality for match {event.event_key}: {overall_score:.2f}")
        
        return report
    
    def _validate_fields(self, data: Dict[str, Any], match_id: int) -> List[ValidationIssue]:
        """Validate individual fields according to rules"""
        issues = []
        
        for field, rules in self.validation_rules.items():
            value = data.get(field)
            
            # Required field check
            if rules.get('required', False) and (value is None or value == ''):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    metric=DataQualityMetric.COMPLETENESS,
                    field=field,
                    message=f"Required field '{field}' is missing or empty",
                    value=value,
                    expected="non-empty value",
                    timestamp=datetime.now(),
                    match_id=match_id
                ))
                continue
            
            if value is None:
                continue
            
            # Type validation
            expected_type = rules.get('type')
            if expected_type and not isinstance(value, expected_type):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    metric=DataQualityMetric.VALIDITY,
                    field=field,
                    message=f"Field '{field}' has wrong type",
                    value=type(value).__name__,
                    expected=expected_type.__name__,
                    timestamp=datetime.now(),
                    match_id=match_id
                ))
                continue
            
            # String validations
            if isinstance(value, str):
                # Length validation
                min_length = rules.get('min_length')
                if min_length and len(value) < min_length:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        metric=DataQualityMetric.VALIDITY,
                        field=field,
                        message=f"Field '{field}' is too short",
                        value=len(value),
                        expected=f"at least {min_length} characters",
                        timestamp=datetime.now(),
                        match_id=match_id
                    ))
                
                max_length = rules.get('max_length')
                if max_length and len(value) > max_length:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        metric=DataQualityMetric.VALIDITY,
                        field=field,
                        message=f"Field '{field}' is too long",
                        value=len(value),
                        expected=f"at most {max_length} characters",
                        timestamp=datetime.now(),
                        match_id=match_id
                    ))
                
                # Pattern validation
                pattern = rules.get('pattern')
                if pattern and not re.match(pattern, value):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        metric=DataQualityMetric.VALIDITY,
                        field=field,
                        message=f"Field '{field}' does not match expected pattern",
                        value=value,
                        expected=pattern,
                        timestamp=datetime.now(),
                        match_id=match_id
                    ))
                
                # Allowed values validation
                allowed_values = rules.get('allowed_values')
                if allowed_values and value not in allowed_values:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        metric=DataQualityMetric.VALIDITY,
                        field=field,
                        message=f"Field '{field}' has invalid value",
                        value=value,
                        expected=f"one of {allowed_values}",
                        timestamp=datetime.now(),
                        match_id=match_id
                    ))
                
                # Date format validation
                date_format = rules.get('date_format')
                if date_format:
                    try:
                        datetime.strptime(value, date_format)
                    except ValueError:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            metric=DataQualityMetric.VALIDITY,
                            field=field,
                            message=f"Field '{field}' has invalid date format",
                            value=value,
                            expected=date_format,
                            timestamp=datetime.now(),
                            match_id=match_id
                        ))
            
            # Numeric validations
            if isinstance(value, (int, float)):
                min_value = rules.get('min_value')
                if min_value is not None and value < min_value:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        metric=DataQualityMetric.VALIDITY,
                        field=field,
                        message=f"Field '{field}' is below minimum value",
                        value=value,
                        expected=f"at least {min_value}",
                        timestamp=datetime.now(),
                        match_id=match_id
                    ))
                
                max_value = rules.get('max_value')
                if max_value is not None and value > max_value:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        metric=DataQualityMetric.VALIDITY,
                        field=field,
                        message=f"Field '{field}' exceeds maximum value",
                        value=value,
                        expected=f"at most {max_value}",
                        timestamp=datetime.now(),
                        match_id=match_id
                    ))
        
        return issues
    
    def _validate_cross_fields(self, data: Dict[str, Any], match_id: int) -> List[ValidationIssue]:
        """Validate relationships between fields"""
        issues = []
        
        # Player keys should be different
        if data.get('first_player_key') == data.get('second_player_key'):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                metric=DataQualityMetric.CONSISTENCY,
                field='player_keys',
                message="Player keys should be different",
                value=f"{data.get('first_player_key')} == {data.get('second_player_key')}",
                expected="different values",
                timestamp=datetime.now(),
                match_id=match_id
            ))
        
        # Player names should be different
        if data.get('first_player') == data.get('second_player'):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                metric=DataQualityMetric.CONSISTENCY,
                field='player_names',
                message="Player names should be different",
                value=f"{data.get('first_player')} == {data.get('second_player')}",
                expected="different names",
                timestamp=datetime.now(),
                match_id=match_id
            ))
        
        # Score format validation
        final_result = data.get('final_result', '')
        game_result = data.get('game_result', '')
        
        if final_result and game_result:
            try:
                final_parts = final_result.replace(' ', '').split('-')
                game_parts = game_result.replace(' ', '').split('-')
                
                if len(final_parts) == 2 and len(game_parts) == 2:
                    final1, final2 = int(final_parts[0]), int(final_parts[1])
                    game1, game2 = int(game_parts[0]), int(game_parts[1])
                    
                    # Game scores should not exceed set scores
                    if game1 > final1 or game2 > final2:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            metric=DataQualityMetric.CONSISTENCY,
                            field='scores',
                            message="Game scores exceed set scores",
                            value=f"Games: {game_result}, Sets: {final_result}",
                            expected="game scores <= set scores",
                            timestamp=datetime.now(),
                            match_id=match_id
                        ))
            except ValueError:
                pass  # Invalid score format - already caught in field validation
        
        return issues
    
    def _validate_historical_consistency(self, event: LiveMatchEvent, match_id: int) -> List[ValidationIssue]:
        """Validate consistency with historical data for the same match"""
        issues = []
        
        if match_id not in self.data_history:
            return issues
        
        history = list(self.data_history[match_id])
        if not history:
            return issues
        
        last_event = history[-1]
        
        # Player names should remain consistent
        if event.first_player != last_event.first_player:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                metric=DataQualityMetric.CONSISTENCY,
                field='first_player',
                message="First player name changed during match",
                value=event.first_player,
                expected=last_event.first_player,
                timestamp=datetime.now(),
                match_id=match_id
            ))
        
        if event.second_player != last_event.second_player:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                metric=DataQualityMetric.CONSISTENCY,
                field='second_player',
                message="Second player name changed during match",
                value=event.second_player,
                expected=last_event.second_player,
                timestamp=datetime.now(),
                match_id=match_id
            ))
        
        # Tournament should remain consistent
        if event.tournament_name != last_event.tournament_name:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                metric=DataQualityMetric.CONSISTENCY,
                field='tournament_name',
                message="Tournament name changed during match",
                value=event.tournament_name,
                expected=last_event.tournament_name,
                timestamp=datetime.now(),
                match_id=match_id
            ))
        
        # Score progression validation
        issues.extend(self._validate_score_progression(event, last_event, match_id))
        
        return issues
    
    def _validate_score_progression(self, current: LiveMatchEvent, previous: LiveMatchEvent, match_id: int) -> List[ValidationIssue]:
        """Validate that scores progress logically"""
        issues = []
        
        try:
            # Parse current and previous scores
            curr_final = current.final_result.replace(' ', '').split('-')
            prev_final = previous.final_result.replace(' ', '').split('-')
            
            if len(curr_final) == 2 and len(prev_final) == 2:
                curr_sets = [int(curr_final[0]), int(curr_final[1])]
                prev_sets = [int(prev_final[0]), int(prev_final[1])]
                
                # Set scores should only increase
                if curr_sets[0] < prev_sets[0] or curr_sets[1] < prev_sets[1]:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        metric=DataQualityMetric.CONSISTENCY,
                        field='final_result',
                        message="Set scores decreased",
                        value=current.final_result,
                        expected=f"scores >= {previous.final_result}",
                        timestamp=datetime.now(),
                        match_id=match_id
                    ))
                
                # Total set difference should be reasonable
                total_sets_change = (curr_sets[0] + curr_sets[1]) - (prev_sets[0] + prev_sets[1])
                if total_sets_change > 1:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        metric=DataQualityMetric.CONSISTENCY,
                        field='final_result',
                        message="Unusual set score jump",
                        value=f"Change: {total_sets_change} sets",
                        expected="at most 1 set change",
                        timestamp=datetime.now(),
                        match_id=match_id
                    ))
        
        except ValueError:
            pass  # Invalid score format - handled elsewhere
        
        return issues
    
    def _validate_business_logic(self, event: LiveMatchEvent, match_id: int) -> List[ValidationIssue]:
        """Validate tennis-specific business logic"""
        issues = []
        
        # Live matches should have reasonable scores
        if event.live:
            try:
                final_parts = event.final_result.replace(' ', '').split('-')
                if len(final_parts) == 2:
                    sets1, sets2 = int(final_parts[0]), int(final_parts[1])
                    
                    # No player should have more than 3 sets in most tournaments
                    if sets1 > 3 or sets2 > 3:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            metric=DataQualityMetric.ACCURACY,
                            field='final_result',
                            message="Unusually high set count",
                            value=event.final_result,
                            expected="sets <= 3 for most matches",
                            timestamp=datetime.now(),
                            match_id=match_id
                        ))
                    
                    # Match should be finished if one player has enough sets
                    if sets1 >= 2 or sets2 >= 2:
                        total_sets = sets1 + sets2
                        if total_sets >= 3 and event.status != 'Finished':
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                metric=DataQualityMetric.CONSISTENCY,
                                field='event_status',
                                message="Match appears to be finished but status is not 'Finished'",
                                value=event.status,
                                expected="'Finished'",
                                timestamp=datetime.now(),
                                match_id=match_id
                            ))
            
            except ValueError:
                pass
        
        # Finished matches should have a winner
        if event.status == 'Finished' and not event.winner:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                metric=DataQualityMetric.COMPLETENESS,
                field='winner',
                message="Finished match should have a winner",
                value=event.winner,
                expected="player name",
                timestamp=datetime.now(),
                match_id=match_id
            ))
        
        return issues
    
    def _validate_timeliness(self, event: LiveMatchEvent, match_id: int) -> List[ValidationIssue]:
        """Validate data timeliness"""
        issues = []
        
        # Check if event timestamp is reasonable
        now = datetime.now()
        event_time = event.timestamp
        
        # Event shouldn't be too far in the future
        if event_time > now + timedelta(minutes=5):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                metric=DataQualityMetric.TIMELINESS,
                field='timestamp',
                message="Event timestamp is in the future",
                value=event_time.isoformat(),
                expected="current time or past",
                timestamp=datetime.now(),
                match_id=match_id
            ))
        
        # Event shouldn't be too old for live matches
        if event.live and event_time < now - timedelta(hours=12):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                metric=DataQualityMetric.TIMELINESS,
                field='timestamp',
                message="Live event timestamp is very old",
                value=event_time.isoformat(),
                expected="recent timestamp for live events",
                timestamp=datetime.now(),
                match_id=match_id
            ))
        
        return issues
    
    def _calculate_metric_scores(self, issues: List[ValidationIssue]) -> Dict[str, float]:
        """Calculate quality scores for each metric"""
        metric_counts = defaultdict(int)
        total_by_metric = defaultdict(int)
        
        # Count issues by metric and severity
        for issue in issues:
            metric = issue.metric.value
            total_by_metric[metric] += 1
            
            # Weight by severity
            if issue.severity == ValidationSeverity.CRITICAL:
                metric_counts[metric] += 4
            elif issue.severity == ValidationSeverity.ERROR:
                metric_counts[metric] += 3
            elif issue.severity == ValidationSeverity.WARNING:
                metric_counts[metric] += 2
            else:  # INFO
                metric_counts[metric] += 1
        
        # Calculate scores (1.0 = perfect, 0.0 = worst)
        scores = {}
        for metric in DataQualityMetric:
            metric_name = metric.value
            
            if metric_name in metric_counts:
                # Calculate score based on issue severity
                issue_weight = metric_counts[metric_name]
                max_possible_weight = total_by_metric[metric_name] * 4  # All critical
                
                if max_possible_weight > 0:
                    scores[metric_name] = max(0.0, 1.0 - (issue_weight / max_possible_weight))
                else:
                    scores[metric_name] = 1.0
            else:
                scores[metric_name] = 1.0  # No issues = perfect score
        
        return scores
    
    def _calculate_overall_score(self, metric_scores: Dict[str, float]) -> float:
        """Calculate overall quality score"""
        if not metric_scores:
            return 1.0
        
        # Weighted average of metric scores
        weights = {
            'completeness': 0.25,
            'accuracy': 0.25,
            'consistency': 0.25,
            'timeliness': 0.15,
            'validity': 0.10
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, score in metric_scores.items():
            weight = weights.get(metric, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 1.0
    
    def _generate_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate recommendations based on validation issues"""
        recommendations = []
        
        # Count issues by type
        issue_counts = defaultdict(int)
        for issue in issues:
            issue_counts[issue.metric] += 1
        
        # Generate recommendations
        if issue_counts[DataQualityMetric.COMPLETENESS] > 0:
            recommendations.append("Ensure all required fields are provided in the data feed")
        
        if issue_counts[DataQualityMetric.CONSISTENCY] > 0:
            recommendations.append("Check for consistency issues in match data progression")
        
        if issue_counts[DataQualityMetric.VALIDITY] > 0:
            recommendations.append("Validate data formats and value ranges at the source")
        
        if issue_counts[DataQualityMetric.TIMELINESS] > 0:
            recommendations.append("Review data timestamp handling and processing delays")
        
        if issue_counts[DataQualityMetric.ACCURACY] > 0:
            recommendations.append("Verify tennis scoring rules and business logic validation")
        
        # Severity-based recommendations
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            recommendations.append("URGENT: Address critical data quality issues immediately")
        
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        if len(error_issues) > 5:
            recommendations.append("High number of errors detected - consider data source review")
        
        return recommendations
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        stats = self.validation_stats.copy()
        
        if self.validation_stats['quality_scores']:
            scores = list(self.validation_stats['quality_scores'])
            stats['average_quality_score'] = statistics.mean(scores)
            stats['min_quality_score'] = min(scores)
            stats['max_quality_score'] = max(scores)
        
        stats['matches_tracked'] = len(self.data_history)
        
        return stats


class DataQualityMonitor:
    """Monitors data quality over time and alerts on degradation"""
    
    def __init__(self, alert_threshold: float = 0.7):
        self.validator = LiveDataValidator()
        self.alert_threshold = alert_threshold
        self.quality_history = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []
        
        # Quality trend tracking
        self.quality_trends = {
            'last_hour': deque(maxlen=60),
            'last_day': deque(maxlen=1440),
            'last_week': deque(maxlen=10080)
        }
        
        self.last_alert_time = {}
        self.alert_cooldown = timedelta(minutes=30)
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for quality alerts"""
        self.alert_callbacks.append(callback)
    
    def process_event(self, event: LiveMatchEvent) -> DataQualityReport:
        """Process event and monitor quality"""
        report = self.validator.validate_live_event(event)
        
        # Track quality over time
        self._update_quality_trends(report.overall_score)
        
        # Check for quality degradation
        self._check_quality_alerts(report)
        
        return report
    
    def _update_quality_trends(self, score: float):
        """Update quality trend tracking"""
        timestamp = datetime.now()
        
        self.quality_history.append((timestamp, score))
        
        # Update trend buckets
        self.quality_trends['last_hour'].append(score)
        self.quality_trends['last_day'].append(score)
        self.quality_trends['last_week'].append(score)
    
    def _check_quality_alerts(self, report: DataQualityReport):
        """Check if quality alerts should be triggered"""
        now = datetime.now()
        
        # Low quality score alert
        if report.overall_score < self.alert_threshold:
            alert_key = f"low_quality_{report.match_id}"
            if self._should_send_alert(alert_key, now):
                self._send_alert("Low Data Quality Detected", {
                    'match_id': report.match_id,
                    'quality_score': report.overall_score,
                    'threshold': self.alert_threshold,
                    'issues_count': len(report.issues),
                    'timestamp': now.isoformat()
                })
        
        # Quality degradation trend alert
        if len(self.quality_trends['last_hour']) >= 30:
            recent_avg = statistics.mean(list(self.quality_trends['last_hour'])[-10:])
            older_avg = statistics.mean(list(self.quality_trends['last_hour'])[-30:-10])
            
            if recent_avg < older_avg - 0.2:  # 20% degradation
                alert_key = "quality_degradation"
                if self._should_send_alert(alert_key, now):
                    self._send_alert("Data Quality Degradation Detected", {
                        'recent_average': recent_avg,
                        'previous_average': older_avg,
                        'degradation': older_avg - recent_avg,
                        'timestamp': now.isoformat()
                    })
    
    def _should_send_alert(self, alert_key: str, now: datetime) -> bool:
        """Check if alert should be sent based on cooldown"""
        if alert_key not in self.last_alert_time:
            self.last_alert_time[alert_key] = now
            return True
        
        time_since_last = now - self.last_alert_time[alert_key]
        if time_since_last > self.alert_cooldown:
            self.last_alert_time[alert_key] = now
            return True
        
        return False
    
    def _send_alert(self, alert_type: str, data: Dict[str, Any]):
        """Send quality alert to registered callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, data)
            except Exception as e:
                logger.error(f"Error in quality alert callback: {e}")
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality monitoring summary"""
        summary = {
            'validator_stats': self.validator.get_validation_statistics(),
            'current_trends': {},
            'alert_threshold': self.alert_threshold,
            'total_events_monitored': len(self.quality_history)
        }
        
        # Calculate trend averages
        for trend_name, scores in self.quality_trends.items():
            if scores:
                summary['current_trends'][trend_name] = {
                    'average_score': statistics.mean(scores),
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'sample_count': len(scores)
                }
        
        return summary


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    monitor = DataQualityMonitor()
    
    # Add alert callback
    def handle_quality_alert(alert_type: str, data: Dict[str, Any]):
        print(f"QUALITY ALERT: {alert_type}")
        print(f"Data: {json.dumps(data, indent=2)}")
    
    monitor.add_alert_callback(handle_quality_alert)
    
    # Example validation (would normally be integrated with WebSocket pipeline)
    from websocket_tennis_client import LiveMatchEvent
    
    # Create sample event
    sample_event = LiveMatchEvent(
        event_key=12345,
        event_date="2024-01-15",
        event_time="14:30",
        first_player="Rafael Nadal",
        first_player_key=1001,
        second_player="Novak Djokovic",
        second_player_key=1002,
        final_result="1 - 0",
        game_result="3 - 2",
        serve="First Player",
        winner=None,
        status="Set 1",
        event_type="ATP Masters",
        tournament_name="Miami Open",
        tournament_key=5001,
        tournament_round="Quarterfinals",
        tournament_season="2024",
        live=True,
        scores=[{"score_first": "1", "score_second": "0", "score_set": "1"}],
        point_by_point=[],
        statistics=[],
        timestamp=datetime.now()
    )
    
    # Process event
    report = monitor.process_event(sample_event)
    
    print(f"Quality Report:")
    print(f"Overall Score: {report.overall_score:.2f}")
    print(f"Issues Found: {len(report.issues)}")
    
    for issue in report.issues:
        print(f"  - {issue.severity.value.upper()}: {issue.message}")
    
    print(f"Recommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")