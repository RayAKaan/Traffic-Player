from src.model.action_recommender import ActionRecommendationLayer
from src.model.cause_reasoning import CauseReasoningLayer, ContextMemory
from src.model.explanation_engine import ExplanationEngine
from src.model.risk_intelligence import RiskSafetyLayer


def test_cause_risk_action_outputs_contract():
    current = {
        "avgSpeed": 10.0,
        "flowRatePerMin": 18.0,
        "laneQueue": {"lane_1": 4.0},
        "laneDensity": {"lane_1": 0.7},
        "frameMetricsLaneSummary": {
            "lane_1": {
                "avgExitRate": 0.4,
                "avgTurnRatio": 0.5,
                "avgQueueLength": 4.0,
                "avgNearVehicleDistance": 1.8,
            }
        },
        "predictiveMetrics": {
            "lane_1": {
                "t_plus_3s": {"queue_length": 5.0},
                "t_plus_8s": {"queue_length": 6.0},
            }
        },
    }
    predictive = {
        "predicted_metrics": {
            "lane_1": {
                "t_plus_3s": {"queue_length": 5.0},
                "t_plus_8s": {"queue_length": 6.0},
            }
        }
    }
    context = ContextMemory(time_since_last_green=60.0)

    cause = CauseReasoningLayer().analyze(current, predictive, context)
    risk = RiskSafetyLayer().assess({**current, "frameMetricsSample": [{"avg_speed": 8.0}, {"avg_speed": 14.0}]})
    action = ActionRecommendationLayer().recommend(predictive, cause, risk)
    explanation = ExplanationEngine().render(current, predictive["predicted_metrics"], cause, risk, action, 0.7)

    assert "primary_cause" in cause
    assert "risk_score" in risk and 0.0 <= risk["risk_score"] <= 1.0
    assert "recommended_action" in action
    assert "CURRENT STATE:" in explanation["explanation"]
