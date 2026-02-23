"""
Offline Evaluation & Reporting.

Computes:
- Recall@5 / Recall@10 (does correct match appear in Top-K?)
- Precision@1 for AUTO_MATCH subset (is auto-confirm safe?)
- No-Match performance (false auto-matches)
- Drift monitoring by sport/league/category
"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from core.models import (
    Decision, EvalResult, GateResult, MatchRecord,
    MappingSuggestion, FeedbackRecord,
)
from core.inference import InferenceEngine

logger = logging.getLogger(__name__)


class EvalDataPoint:
    """A single evaluation example with ground truth."""
    
    def __init__(
        self,
        op_match: MatchRecord,
        true_b365_id: Optional[str],   # None = NO_MATCH case
        decision: Decision,
        is_swapped: bool = False,
    ):
        self.op_match = op_match
        self.true_b365_id = true_b365_id
        self.decision = decision
        self.is_swapped = is_swapped


class Evaluator:
    """
    Evaluate the inference engine against labeled data.
    """
    
    def __init__(self, engine: InferenceEngine):
        self.engine = engine
    
    def evaluate(
        self,
        eval_data: List[EvalDataPoint],
        model_version: str = "baseline",
    ) -> EvalResult:
        """
        Run full evaluation suite using batch inference for speed.
        """
        logger.info(f"Evaluating {len(eval_data)} examples...")

        op_matches = [dp.op_match for dp in eval_data]
        suggestions = self.engine.predict_batch(op_matches)
        predictions: List[Tuple[EvalDataPoint, MappingSuggestion]] = list(
            zip(eval_data, suggestions)
        )
        
        # Compute metrics
        recall_5 = self._recall_at_k(predictions, k=5)
        recall_10 = self._recall_at_k(predictions, k=10)
        precision_1 = self._precision_at_1(predictions)
        auto_precision = self._auto_match_precision(predictions)
        auto_rate = self._auto_match_rate(predictions)
        no_match_fp = self._no_match_false_positive_rate(predictions)
        
        # Per-sport breakdown
        per_sport = self._metrics_by_group(predictions, group_fn=lambda dp: dp.op_match.sport)
        per_category = self._metrics_by_group(
            predictions,
            group_fn=lambda dp: ",".join(sorted(dp.op_match.category_tags)) or "NONE"
        )
        
        result = EvalResult(
            model_version=model_version,
            dataset_size=len(eval_data),
            recall_at_5=recall_5,
            recall_at_10=recall_10,
            precision_at_1=precision_1,
            auto_match_precision=auto_precision,
            auto_match_rate=auto_rate,
            no_match_false_positive_rate=no_match_fp,
            per_sport_metrics=per_sport,
            per_category_metrics=per_category,
        )
        
        logger.info(f"Results: R@5={recall_5:.4f}, R@10={recall_10:.4f}, "
                     f"P@1={precision_1:.4f}, AutoP={auto_precision:.4f}")
        
        return result
    
    # ── Core Metrics ──
    
    def _recall_at_k(
        self,
        predictions: List[Tuple[EvalDataPoint, MappingSuggestion]],
        k: int,
    ) -> float:
        """
        Recall@K: For cases where a match EXISTS, does the true B365 match
        appear in the Top-K candidates?
        """
        match_exists = [
            (dp, sg) for dp, sg in predictions
            if dp.decision in (Decision.MATCH, Decision.SWAPPED)
        ]
        
        if not match_exists:
            return 0.0
        
        hits = 0
        for dp, sg in match_exists:
            candidate_ids = [c.b365_match_id for c in sg.candidates_top5[:k]]
            if dp.true_b365_id in candidate_ids:
                hits += 1
        
        return hits / len(match_exists)
    
    def _precision_at_1(
        self,
        predictions: List[Tuple[EvalDataPoint, MappingSuggestion]],
    ) -> float:
        """
        Precision@1: When we rank a candidate as #1, is it correct?
        Only for match-exists cases.
        """
        match_exists = [
            (dp, sg) for dp, sg in predictions
            if dp.decision in (Decision.MATCH, Decision.SWAPPED)
            and sg.candidates_top5
        ]
        
        if not match_exists:
            return 0.0
        
        correct = 0
        for dp, sg in match_exists:
            top1_id = sg.candidates_top5[0].b365_match_id
            if top1_id == dp.true_b365_id:
                correct += 1
        
        return correct / len(match_exists)
    
    def _auto_match_precision(
        self,
        predictions: List[Tuple[EvalDataPoint, MappingSuggestion]],
    ) -> float:
        """
        AUTO_MATCH Precision: When we auto-confirm, are we correct?
        This is the SAFETY metric — must be >= 98%.
        
        Correct = AUTO_MATCH and Top-1 is the true match.
        Incorrect = AUTO_MATCH but Top-1 is wrong, OR auto-matching a NO_MATCH case.
        """
        auto_matches = [
            (dp, sg) for dp, sg in predictions
            if sg.gate_decision == GateResult.AUTO_MATCH
        ]
        
        if not auto_matches:
            return 1.0  # No auto-matches = no errors
        
        correct = 0
        for dp, sg in auto_matches:
            if dp.decision == Decision.NO_MATCH:
                # Auto-matched a no-match case = FALSE POSITIVE
                continue
            
            top1_id = sg.candidates_top5[0].b365_match_id
            if top1_id == dp.true_b365_id:
                correct += 1
        
        return correct / len(auto_matches)
    
    def _auto_match_rate(
        self,
        predictions: List[Tuple[EvalDataPoint, MappingSuggestion]],
    ) -> float:
        """What fraction of predictions are auto-matched (vs need_review)?"""
        if not predictions:
            return 0.0
        auto = sum(1 for _, sg in predictions if sg.gate_decision == GateResult.AUTO_MATCH)
        return auto / len(predictions)
    
    def _no_match_false_positive_rate(
        self,
        predictions: List[Tuple[EvalDataPoint, MappingSuggestion]],
    ) -> float:
        """
        For NO_MATCH cases: how often does the system auto-match?
        This should be ~0.
        """
        no_match_cases = [
            (dp, sg) for dp, sg in predictions
            if dp.decision == Decision.NO_MATCH
        ]
        
        if not no_match_cases:
            return 0.0
        
        false_auto = sum(
            1 for _, sg in no_match_cases
            if sg.gate_decision == GateResult.AUTO_MATCH
        )
        
        return false_auto / len(no_match_cases)
    
    # ── Per-Group Metrics ──
    
    def _metrics_by_group(
        self,
        predictions: List[Tuple[EvalDataPoint, MappingSuggestion]],
        group_fn,
    ) -> Dict[str, dict]:
        """Compute metrics broken down by group (sport, category, etc.)."""
        groups = defaultdict(list)
        for dp, sg in predictions:
            key = group_fn(dp)
            groups[key].append((dp, sg))
        
        result = {}
        for key, group_preds in groups.items():
            match_exists = [
                (dp, sg) for dp, sg in group_preds
                if dp.decision in (Decision.MATCH, Decision.SWAPPED)
            ]
            
            r5 = self._recall_at_k(group_preds, 5)
            p1 = self._precision_at_1(group_preds)
            
            auto_count = sum(
                1 for _, sg in group_preds
                if sg.gate_decision == GateResult.AUTO_MATCH
            )
            
            result[key] = {
                "total": len(group_preds),
                "match_exists": len(match_exists),
                "recall_at_5": round(r5, 4),
                "precision_at_1": round(p1, 4),
                "auto_match_count": auto_count,
                "auto_match_rate": round(auto_count / len(group_preds), 4) if group_preds else 0,
            }
        
        return result


class ReportGenerator:
    """Generate formatted evaluation reports."""
    
    @staticmethod
    def generate_report(
        eval_result: EvalResult,
        previous_result: Optional[EvalResult] = None,
        output_path: str = None,
    ) -> str:
        """
        Generate a human-readable evaluation report.
        Optionally compares with a previous result for drift monitoring.
        """
        lines = [
            "=" * 70,
            "  AI MATCH MAPPING ENGINE — EVALUATION REPORT",
            "=" * 70,
            f"  Model Version:  {eval_result.model_version}",
            f"  Dataset Size:   {eval_result.dataset_size}",
            f"  Evaluated At:   {eval_result.evaluated_at.isoformat()}",
            "=" * 70,
            "",
            "── CORE METRICS ──────────────────────────────────",
            "",
        ]
        
        def fmt_metric(name, value, prev_value=None, higher_is_better=True):
            line = f"  {name:<35} {value:.4f}"
            if prev_value is not None:
                delta = value - prev_value
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
                good = (delta > 0) == higher_is_better
                indicator = "✓" if good else "⚠"
                line += f"  ({arrow} {abs(delta):.4f} {indicator})"
            return line
        
        prev = previous_result
        lines.append(fmt_metric(
            "Recall@5", eval_result.recall_at_5,
            prev.recall_at_5 if prev else None
        ))
        lines.append(fmt_metric(
            "Recall@10", eval_result.recall_at_10,
            prev.recall_at_10 if prev else None
        ))
        lines.append(fmt_metric(
            "Precision@1", eval_result.precision_at_1,
            prev.precision_at_1 if prev else None
        ))
        lines.append(fmt_metric(
            "AUTO_MATCH Precision (Safety)", eval_result.auto_match_precision,
            prev.auto_match_precision if prev else None
        ))
        lines.append(fmt_metric(
            "AUTO_MATCH Rate", eval_result.auto_match_rate,
            prev.auto_match_rate if prev else None
        ))
        lines.append(fmt_metric(
            "No-Match False Positive Rate", eval_result.no_match_false_positive_rate,
            prev.no_match_false_positive_rate if prev else None,
            higher_is_better=False,
        ))
        
        # Success criteria check
        lines.extend(["", "── SUCCESS CRITERIA ──────────────────────────────", ""])
        checks = [
            ("Recall@5 >= 0.90", eval_result.recall_at_5 >= 0.90),
            ("AUTO_MATCH Precision >= 0.98", eval_result.auto_match_precision >= 0.98),
            ("No-Match FP Rate < 0.02", eval_result.no_match_false_positive_rate < 0.02),
        ]
        for check_name, passed in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            lines.append(f"  {check_name:<40} {status}")
        
        # Per-sport breakdown
        if eval_result.per_sport_metrics:
            lines.extend(["", "── PER-SPORT BREAKDOWN ──────────────────────────", ""])
            lines.append(f"  {'Sport':<20} {'Count':>6} {'R@5':>8} {'P@1':>8} {'Auto%':>8}")
            lines.append("  " + "-" * 55)
            for sport, m in sorted(eval_result.per_sport_metrics.items()):
                lines.append(
                    f"  {sport:<20} {m['total']:>6} "
                    f"{m['recall_at_5']:>8.4f} {m['precision_at_1']:>8.4f} "
                    f"{m['auto_match_rate']:>8.4f}"
                )
        
        # Per-category breakdown
        if eval_result.per_category_metrics:
            lines.extend(["", "── PER-CATEGORY BREAKDOWN ───────────────────────", ""])
            lines.append(f"  {'Category':<20} {'Count':>6} {'R@5':>8} {'P@1':>8} {'Auto%':>8}")
            lines.append("  " + "-" * 55)
            for cat, m in sorted(eval_result.per_category_metrics.items()):
                lines.append(
                    f"  {cat:<20} {m['total']:>6} "
                    f"{m['recall_at_5']:>8.4f} {m['precision_at_1']:>8.4f} "
                    f"{m['auto_match_rate']:>8.4f}"
                )
        
        lines.extend(["", "=" * 70])
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            # Also save JSON version
            json_path = output_path.replace(".txt", ".json")
            with open(json_path, "w") as f:
                json.dump(eval_result.model_dump(), f, indent=2, default=str)
            logger.info(f"Report saved: {output_path}")
        
        return report
